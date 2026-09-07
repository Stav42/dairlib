#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

#include <drake/multibody/tree/multibody_tree_indexes.h>

#include "allegro_grasp_c3_squeeze_control.h"

namespace drake {
namespace systems {
template <typename T>
class Context;
}
namespace multibody {
template <typename T>
class MultibodyPlant;
}
}  // namespace drake

namespace dairlib::allegro_grasp_c3 {

// Data needed to realize C3's first planned contact force as the OSC branch's
// Jacobian-transposed normal-force proxy.  It deliberately contains no gflags
// or C3 object: callers extract the physical lambda_0 first, which makes this
// computation a direct unit-test seam.
struct OscExecutorRequest {
  const drake::multibody::MultibodyPlant<double>& plant;
  const drake::systems::Context<double>& context;
  drake::multibody::ModelInstanceIndex hand_model;
  drake::multibody::ModelInstanceIndex cube_model;
  std::array<drake::multibody::BodyIndex, 4> tip_bodies;
  std::array<int, 4> finger_start;
  std::vector<int> active_fingers;
  std::vector<std::vector<int>> normal_groups;
  std::array<double, 4> force_crossfade_from;
  std::array<double, 4> finger_joined_time;
  std::array<Eigen::Vector3d, 4> press_directions_in_cube;
  Eigen::VectorXd q_desired;
  Eigen::VectorXd q_measured;
  Eigen::VectorXd v_measured;
  Eigen::VectorXd lambda0_physical;
  // Reconstructed force on the cube from the same C3 lambda0.  The
  // per-finger entries are zero for inactive fingers.
  std::array<Eigen::Vector3d, 4> c3_force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  // Low-rate measured-wrench feedback correction ON the cube.  It is already
  // contact- and force-rate-limited by the caller, and is zero when feedback
  // is disabled or has declared inadequate yaw authority.
  std::array<Eigen::Vector3d, 4> wrench_feedback_force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  bool has_c3_full_contact_force_plan{};
  bool use_full_contact_force{};
  double time{};
  double dt{};
  double kp{};
  double kd{};
  double desired_velocity_filter_tau{};
  double force_ramp_time{};
  double lambda_torque_scale{};
};

struct OscExecutorResult {
  Eigen::VectorXd torque;
  Eigen::VectorXd pd_torque;
  Eigen::VectorXd force_torque;
  bool used_full_contact_force{};
  std::array<double, 4> normal_force_target{};
  std::array<double, 4> normal_force_applied{};
  std::array<Eigen::Vector3d, 4> normal_direction_world{};
  // The first array is C3's unscaled full contact-force plan ON the cube.
  // The second is the force target actually mapped through Jᵀ after
  // lambda_torque_scale and any contact-join crossfade.
  std::array<Eigen::Vector3d, 4> c3_force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> force_command_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> wrench_feedback_force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
};

// Computes tau = tau_gravity + tau_PD + sum_i J_i^T f_i.  By default f_i is
// the normal-force proxy f_{n,i} n_i. In experimental full-force mode it is
// the complete C3 lambda_n + beta force reconstructed on the cube and scaled
// by the same normal-force safety ramp. This is intentionally not an
// operational-space inverse-dynamics controller.
OscExecutorResult ComputeOscExecutorTorque(
    const OscExecutorRequest& request,
    DesiredVelocityFilterState* desired_velocity_filter);

// A least-squares Cartesian-force proxy for one torque vector at each
// fingertip.  For every finger this solves J_i^T f_i ≈ tau_i, independently.
// It is a diagnostic projection of commanded torque, not the plant's resolved
// contact force (which also depends on contact dynamics and the environment).
struct FingertipTorqueProjectionRequest {
  const drake::multibody::MultibodyPlant<double>& plant;
  const drake::systems::Context<double>& context;
  drake::multibody::ModelInstanceIndex hand_model;
  std::array<drake::multibody::BodyIndex, 4> tip_bodies;
  std::array<int, 4> finger_start;
  Eigen::VectorXd hand_torque;
};

struct FingertipTorqueProjection {
  std::array<Eigen::Vector3d, 4> force_world{};
  std::array<double, 4> relative_torque_residual{};
};

FingertipTorqueProjection ProjectHandTorqueToFingertipForces(
    const FingertipTorqueProjectionRequest& request);

// Data for task_space after the caller has selected its position target and
// extracted the C3 dual-delta lambda.  Keeping the force projection and the
// torque realization here gives direct tests a single observable result.
struct TaskSpaceExecutorRequest {
  const drake::multibody::MultibodyPlant<double>& plant;
  const drake::systems::Context<double>& context;
  drake::multibody::ModelInstanceIndex hand_model;
  drake::multibody::ModelInstanceIndex cube_model;
  std::array<drake::multibody::BodyIndex, 3> tip_bodies;
  std::vector<std::vector<int>> normal_groups;
  Eigen::VectorXd fingertip_position_desired;
  Eigen::VectorXd hand_velocity;
  Eigen::VectorXd lambda_delta0;
  Eigen::VectorXd c3_input0_scaled;
  double task_kp{};
  double task_kd{};
  double force_floor{};
  bool add_gravity_compensation{};
};

Eigen::VectorXd ComputeTaskSpaceExecutorTorque(
    const TaskSpaceExecutorRequest& request);

}  // namespace dairlib::allegro_grasp_c3
