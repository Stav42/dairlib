#pragma once

#include <iostream>

#include <Eigen/Dense>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/inverse_kinematics/inverse_kinematics.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/solvers/solve.h>
#include <drake/systems/framework/context.h>

//   ┌───────────────┬───────────────────┬─────────────┐
//   │    Finger     │       Links       │  Tip body   │
//   ├───────────────┼───────────────────┼─────────────┤
//   │ Index         │ link_0 → link_3   │ link_3_tip  │
//   ├───────────────┼───────────────────┼─────────────┤
//   │ Middle        │ link_4 → link_7   │ link_7_tip  │
//   ├───────────────┼───────────────────┼─────────────┤
//   │ Ring (unused) │ link_8 → link_11  │ link_11_tip │
//   ├───────────────┼───────────────────┼─────────────┤
//   │ Thumb         │ link_12 → link_15 │ link_15_tip │
//   └───────────────┴───────────────────┴─────────────┘

namespace dairlib {

// Builds X_WC from the raw 7-element free-body position vector
// [qw, qx, qy, qz, x, y, z] that Drake's GetPositions returns.
inline drake::math::RigidTransform<double> CubePoseFromPositions(
    const Eigen::VectorXd& q) {
  Eigen::Quaterniond quat(q(0), q(1), q(2), q(3));
  return drake::math::RigidTransform<double>(
      drake::math::RotationMatrix<double>(quat), q.tail(3));
}

// Returns the six face-centre positions in world frame as an 18-vector
// [face1, face2, face3, face4, face5, face6], each a 3-vector.
// Face ordering: +X, -X, +Y, -Y, +Z, -Z.
inline Eigen::VectorXd GetFaceCentrePositions(
    const drake::math::RigidTransform<double>& X_WC, double cube_size) {
  const double h = cube_size / 2.0;
  Eigen::VectorXd centres(18);
  centres <<
      X_WC * Eigen::Vector3d( h, 0, 0),
      X_WC * Eigen::Vector3d(-h, 0, 0),
      X_WC * Eigen::Vector3d(0,  h, 0),
      X_WC * Eigen::Vector3d(0, -h, 0),
      X_WC * Eigen::Vector3d(0, 0,  h),
      X_WC * Eigen::Vector3d(0, 0, -h);
  return centres;
}

// Returns the three fingertip contact positions in world frame as a 9-vector
// [p_index, p_middle, p_thumb], each a 3-vector.
// a, b are x-offsets on face 4 (-Y); c is x-offset on face 3 (+Y).
inline Eigen::VectorXd GetGraspPositions(
    const drake::math::RigidTransform<double>& X_WC, double cube_size,
    double a, double b, double c) {
  const double h = cube_size / 2.0;
  Eigen::VectorXd centres(9);
  centres <<
      X_WC * Eigen::Vector3d(a, -h, 0),   // index:  face 4 (-Y)
      X_WC * Eigen::Vector3d(b, -h, 0),   // middle: face 4 (-Y)
      X_WC * Eigen::Vector3d(c, +h, 0);   // thumb:  face 3 (+Y)
  return centres;
}

inline Eigen::VectorXd GetIntermediatePosition(
  const drake::math::RigidTransform<double>& X_WC, double cube_size,
  double a, double b, double c) {
  const double h = cube_size / 2.0;
  Eigen::VectorXd centres(9);
  centres <<
      X_WC * Eigen::Vector3d(a, -h-0.04, 0),   // index:  face 4 (-Y)
      X_WC * Eigen::Vector3d(b, -h-0.04, 0),   // middle: face 4 (-Y)
      X_WC * Eigen::Vector3d(c, +h+0.04, 0);   // thumb:  face 3 (+Y)
  return centres;
}

// Given x-offsets a (index), b (middle), c (thumb) along their respective
// faces and the chosen index finger magnitude, returns
// [index_mag, middle_mag, thumb_mag, ratio].
// Derived from force + moment balance with normal-only forces on face 4 / face 3.
inline Eigen::Vector4d GetGraspMagnitudes(
    double a, double b, double c, double mag_index) {
  const double ratio = (a - c) / (c - b);
  const double middle_mag = ratio * mag_index;
  const double thumb_mag = mag_index + middle_mag;
  Eigen::Vector4d result;
  result << mag_index, middle_mag, thumb_mag, ratio;
  return result;
}

// Solves IK to place the three fingertips at the given world-frame contact
// positions. grasp_positions is a 9-vector [p_index, p_middle, p_thumb].
// tip_frame_offset is the point (expressed in each "*_tip" frame) that is
// actually constrained to the target — NOT necessarily the frame origin.
// The "*_tip" frames' origins do not coincide with the true fingertip
// collision surface (confirmed empirically via the /tip_frame/* Meshcat
// triads: the surface sits ~0.0115 m along the frame's own local +Z from
// the origin), so callers that want the SURFACE at a given world point
// should pass tip_frame_offset = Vector3d(0, 0, 0.0115), not the default
// origin. Returns the full plant position vector q (Allegro + cube DOF).
inline Eigen::VectorXd SolveGraspIK(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* plant_context,
    const Eigen::VectorXd& grasp_positions,
    const Eigen::Vector3d& tip_frame_offset = Eigen::Vector3d::Zero()) {
  const Eigen::Vector3d p_index  = grasp_positions.segment<3>(0);
  const Eigen::Vector3d p_middle = grasp_positions.segment<3>(3);
  const Eigen::Vector3d p_thumb  = grasp_positions.segment<3>(6);
  const Eigen::Vector3d tol = Eigen::Vector3d::Constant(0.001);

  drake::multibody::InverseKinematics ik(plant, plant_context);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_3_tip"), tip_frame_offset,
      plant.world_frame(), p_index - tol, p_index + tol);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_7_tip"), tip_frame_offset,
      plant.world_frame(), p_middle - tol, p_middle + tol);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_15_tip"), tip_frame_offset,
      plant.world_frame(), p_thumb - tol, p_thumb + tol);

  ik.get_mutable_prog()->SetInitialGuess(ik.q(), plant.GetPositions(*plant_context));

  auto result = drake::solvers::Solve(ik.prog());
  if (!result.is_success()) {
    std::cerr << "IK failed!" << std::endl;
  }

  return result.GetSolution(ik.q());
}

// Same as SolveGraspIK, plus a 4th constraint on the ring fingertip
// (link_11_tip — see the table above; unused by the normal 3-finger grasp).
// index/middle/thumb are explicitly pinned to grasp_positions here (not
// just left at the initial guess) so they are guaranteed to stay put, not
// merely likely to — the IK problem has no cost function, so an
// unconstrained body is free to move anywhere a feasible solution permits.
// ring_frame_offset is SEPARATE from tip_frame_offset (not shared with
// index/middle/thumb) because ring's own frame-origin-to-true-surface
// offset was found, empirically, to differ from theirs — see
// --ring_tip_surface_offset_{y,z} in allegro_grasp_c3_squeeze.cc.
inline Eigen::VectorXd SolveGraspIKWithRing(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* plant_context,
    const Eigen::VectorXd& grasp_positions,
    const Eigen::Vector3d& ring_position,
    const Eigen::Vector3d& tip_frame_offset = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& ring_frame_offset = Eigen::Vector3d::Zero()) {
  const Eigen::Vector3d p_index  = grasp_positions.segment<3>(0);
  const Eigen::Vector3d p_middle = grasp_positions.segment<3>(3);
  const Eigen::Vector3d p_thumb  = grasp_positions.segment<3>(6);
  const Eigen::Vector3d tol = Eigen::Vector3d::Constant(0.001);

  drake::multibody::InverseKinematics ik(plant, plant_context);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_3_tip"), tip_frame_offset,
      plant.world_frame(), p_index - tol, p_index + tol);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_7_tip"), tip_frame_offset,
      plant.world_frame(), p_middle - tol, p_middle + tol);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_15_tip"), tip_frame_offset,
      plant.world_frame(), p_thumb - tol, p_thumb + tol);

  ik.AddPositionConstraint(
      plant.GetFrameByName("link_11_tip"), ring_frame_offset,
      plant.world_frame(), ring_position - tol, ring_position + tol);

  ik.get_mutable_prog()->SetInitialGuess(ik.q(), plant.GetPositions(*plant_context));

  auto result = drake::solvers::Solve(ik.prog());
  if (!result.is_success()) {
    std::cerr << "IK failed!" << std::endl;
  }

  return result.GetSolution(ik.q());
}

}  // namespace dairlib
