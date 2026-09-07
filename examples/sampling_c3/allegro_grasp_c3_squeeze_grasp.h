#pragma once

#include <array>

#include <Eigen/Core>

#include <drake/common/trajectories/piecewise_polynomial.h>
#include <drake/math/rigid_transform.h>

#include "allegro_grasp_c3_squeeze_config.h"
#include "allegro_grasp_c3_squeeze_environment.h"

namespace dairlib::allegro_grasp_c3 {

// Immutable grasp geometry and IK products shared by reach, planning, and
// execution. Mutable contact footprints are intentionally kept separate from
// the command-line configuration because gait/recontact changes them.
class GraspSetup {
 public:
  GraspSetup(const SqueezeConfig& config, SimulationEnvironment* environment);

  void PreviewInitialPoses(bool wait_for_user);
  Eigen::VectorXd SolveContactIk(const drake::math::RigidTransform<double>& pose,
                                 const Eigen::VectorXd& seed,
                                 bool include_ring, bool* success);
  Eigen::VectorXd SolveLegIk(const drake::math::RigidTransform<double>& pose,
                            int moving_finger,
                            const Eigen::Vector3d& target_in_cube,
                            const Eigen::VectorXd& seed, bool* success);
  Eigen::Vector3d FingertipPosition(int finger) const;
  Eigen::VectorXd FingertipPositionsForHand(const Eigen::VectorXd& joints);

  static constexpr double kNominalCubeSize = 0.06;
  static constexpr std::array<int, 4> kFingerStarts{0, 4, 12, 8};

  int grasp_finger_count{};
  int hand_positions{};
  // Actual edge length of the selected cube model (m).
  double cube_size{};
  drake::math::RigidTransform<double> initial_cube_pose;
  Eigen::VectorXd initial_cube_positions;
  Eigen::Vector3d tip_surface_offset;
  Eigen::Vector3d ring_surface_offset;
  std::array<drake::multibody::BodyIndex, 4> tip_bodies;
  drake::multibody::BodyIndex cube_body;

  std::array<Eigen::Vector3d, 4> footprints;
  std::array<Eigen::Vector3d, 4> initial_footprints;
  Eigen::Vector3d relay_ring_hold;
  Eigen::Vector3d relay_ring_park;
  Eigen::Vector3d spider_ring_hold;
  Eigen::Vector3d spider_ring_park;
  std::array<Eigen::Vector3d, 3> relay_regrasp_targets;

  Eigen::VectorXd contact_positions;
  Eigen::VectorXd pregrasp_positions;
  Eigen::VectorXd release_middle_positions;
  Eigen::VectorXd ring_regrasp_positions;
  Eigen::VectorXd ring_regrasp_mid_positions;
  Eigen::VectorXd middle_regrasp_positions;
  Eigen::VectorXd middle_regrasp_mid_positions;
  Eigen::VectorXd index_regrasp_positions;
  Eigen::VectorXd index_regrasp_mid_positions;
  drake::trajectories::PiecewisePolynomial<double> reach_trajectory;
  drake::trajectories::PiecewisePolynomial<double> reach_velocity;

 private:
  const SqueezeConfig& config_;
  SimulationEnvironment* environment_{};
  std::unique_ptr<drake::systems::Context<double>> ik_context_;
};

}  // namespace dairlib::allegro_grasp_c3
