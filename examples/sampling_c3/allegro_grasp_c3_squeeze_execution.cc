#include "allegro_grasp_c3_squeeze_execution.h"

#include <algorithm>

#include <Eigen/QR>

#include <drake/math/rotation_matrix.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_forces.h>
#include <drake/systems/framework/context.h>

#include "allegro_grasp_c3_squeeze_contact.h"
#include "cube_kinematics.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::math::RotationMatrix;
using drake::multibody::JacobianWrtVariable;
using Eigen::Vector3d;
using Eigen::VectorXd;

}  // namespace

OscExecutorResult ComputeOscExecutorTorque(
    const OscExecutorRequest& request,
    DesiredVelocityFilterState* desired_velocity_filter) {
  const int hand_velocity_count = request.plant.num_velocities(request.hand_model);
  const VectorXd gravity = request.plant.GetVelocitiesFromArray(
      request.hand_model,
      -request.plant.CalcGravityGeneralizedForces(request.context));
  const VectorXd desired_velocity = UpdateDesiredVelocityFilter(
      request.q_desired, request.dt, request.desired_velocity_filter_tau,
      desired_velocity_filter);
  const VectorXd pd = ComputeJointPdTorque(
      request.q_desired, request.q_measured, request.v_measured,
      desired_velocity, request.kp, request.kd);

  const RotationMatrix<double> R_WC = CubePoseFromPositions(
      request.plant.GetPositions(request.context, request.cube_model))
                                          .rotation();
  VectorXd force = VectorXd::Zero(hand_velocity_count);
  OscExecutorResult result;
  result.pd_torque = pd;
  result.used_full_contact_force =
      request.use_full_contact_force &&
      request.has_c3_full_contact_force_plan;
  for (int finger = 0; finger < 4; ++finger) {
    result.normal_direction_world.at(finger) =
        R_WC * request.press_directions_in_cube.at(finger);
  }
  for (size_t active_index = 0;
       active_index < request.active_fingers.size(); ++active_index) {
    const int finger = request.active_fingers[active_index];
    const Vector3d c3_force_on_cube =
        request.c3_force_on_cube_world.at(finger);
    result.c3_force_on_cube_world.at(finger) = c3_force_on_cube;
    const double normal_force_target = SumNormalForce(
        request.lambda0_physical, request.normal_groups.at(active_index));
    result.normal_force_target.at(finger) = normal_force_target;
    // Keep the legacy normal-only path bit-for-bit equivalent when the
    // experimental full-force option is off.
    double normal_force = normal_force_target;
    normal_force = BlendJoinedNormalForce(
        request.force_crossfade_from.at(finger), normal_force,
        request.time - request.finger_joined_time.at(finger),
        request.force_ramp_time, request.lambda_torque_scale);
    // A negative scalar would turn a full contact-force command into a
    // tensile, outward push. The experimental branch refuses it; valid C3
    // normal-force solutions are already nonnegative.
    if (result.used_full_contact_force)
      normal_force = std::max(0.0, normal_force);
    result.normal_force_applied.at(finger) = normal_force;
    Vector3d force_command_on_cube =
        normal_force * result.normal_direction_world.at(finger);
    if (result.used_full_contact_force) {
      // Keep the same scalar normal-force safety ramp used by the legacy
      // executor, then apply every physical lambda_n/beta component with its
      // C3 world-frame direction. A nonpositive raw normal force has no
      // admissible friction force, so command no full contact force.
      const double scale =
          normal_force_target > 1e-9 ? normal_force / normal_force_target
                                     : 0.0;
      force_command_on_cube = scale * c3_force_on_cube;
      // This correction is constructed from the measured SAP contact
      // geometry and constrained before reaching the executor.  Keep it
      // separate from C3's normal-force crossfade: it is only present for an
      // established resolved contact, and blending it again would make the
      // measured outer loop's applied command ambiguous.
      const Vector3d feedback_force =
          request.wrench_feedback_force_on_cube_world.at(finger);
      force_command_on_cube += feedback_force;
      result.wrench_feedback_force_on_cube_world.at(finger) = feedback_force;
    }
    result.force_command_on_cube_world.at(finger) = force_command_on_cube;

    Eigen::MatrixXd J(3, request.plant.num_velocities());
    request.plant.CalcJacobianTranslationalVelocity(
        request.context, JacobianWrtVariable::kV,
        request.plant.get_body(request.tip_bodies.at(finger)).body_frame(),
        Vector3d::Zero(), request.plant.world_frame(), request.plant.world_frame(),
        &J);
    force += J.leftCols(hand_velocity_count).transpose() *
             force_command_on_cube;
  }
  result.force_torque = force;
  result.torque = gravity + pd + force;
  return result;
}

FingertipTorqueProjection ProjectHandTorqueToFingertipForces(
    const FingertipTorqueProjectionRequest& request) {
  FingertipTorqueProjection result;
  const int hand_velocity_count =
      request.plant.num_velocities(request.hand_model);
  for (int finger = 0; finger < 4; ++finger) {
    const int first_joint = request.finger_start.at(finger);
    if (first_joint < 0 || first_joint + 4 > hand_velocity_count ||
        first_joint + 4 > request.hand_torque.size()) {
      result.relative_torque_residual.at(finger) = 0.0;
      continue;
    }
    Eigen::MatrixXd J(3, request.plant.num_velocities());
    request.plant.CalcJacobianTranslationalVelocity(
        request.context, JacobianWrtVariable::kV,
        request.plant.get_body(request.tip_bodies.at(finger)).body_frame(),
        Vector3d::Zero(), request.plant.world_frame(),
        request.plant.world_frame(), &J);
    const Eigen::MatrixXd J_finger = J.block(0, first_joint, 3, 4);
    const Eigen::Vector4d tau_finger =
        request.hand_torque.segment<4>(first_joint);
    const Eigen::Vector3d force =
        J_finger.transpose().completeOrthogonalDecomposition().solve(tau_finger);
    result.force_world.at(finger) = force;
    const double torque_norm = tau_finger.norm();
    result.relative_torque_residual.at(finger) =
        torque_norm > 1e-12
            ? (J_finger.transpose() * force - tau_finger).norm() / torque_norm
            : 0.0;
  }
  return result;
}

Eigen::VectorXd ComputeTaskSpaceExecutorTorque(
    const TaskSpaceExecutorRequest& request) {
  const int hand_velocity_count = request.plant.num_velocities(request.hand_model);
  const VectorXd gravity = request.plant.GetVelocitiesFromArray(
      request.hand_model,
      -request.plant.CalcGravityGeneralizedForces(request.context));
  Eigen::VectorXd torque = request.add_gravity_compensation
                                ? gravity
                                : VectorXd::Zero(gravity.size());
  const RotationMatrix<double> R_WC = CubePoseFromPositions(
      request.plant.GetPositions(request.context, request.cube_model))
                                          .rotation();
  const std::array<Vector3d, 3> kPressDirectionsInCube{
      Vector3d(0, 1, 0), Vector3d(0, 1, 0), Vector3d(0, -1, 0)};
  const std::array<double, 3> force_floor{
      request.force_floor, request.force_floor, 2.0 * request.force_floor};

  for (int finger = 0; finger < 3; ++finger) {
    const double normal_force = std::max(
        std::max(SumNormalForce(request.lambda_delta0,
                                request.normal_groups.at(finger)),
                 0.0),
        force_floor.at(finger));
    Eigen::MatrixXd J(3, request.plant.num_velocities());
    request.plant.CalcJacobianTranslationalVelocity(
        request.context, JacobianWrtVariable::kV,
        request.plant.get_body(request.tip_bodies.at(finger)).body_frame(),
        Vector3d::Zero(), request.plant.world_frame(), request.plant.world_frame(),
        &J);
    const Eigen::MatrixXd hand_jacobian = J.leftCols(hand_velocity_count);
    const Vector3d p = request.plant.EvalBodyPoseInWorld(
        request.context, request.plant.get_body(request.tip_bodies.at(finger)))
                           .translation();
    const Vector3d v = hand_jacobian * request.hand_velocity;
    const Vector3d task_force =
        request.task_kp * (request.fingertip_position_desired.segment<3>(3 * finger) - p) -
        request.task_kd * v +
        normal_force * (R_WC * kPressDirectionsInCube.at(finger));
    torque += hand_jacobian.transpose() * task_force;
  }
  return torque + request.c3_input0_scaled;
}

}  // namespace dairlib::allegro_grasp_c3
