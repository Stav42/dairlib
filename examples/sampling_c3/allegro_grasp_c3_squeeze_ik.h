#pragma once

#include <array>
#include <string>

#include <Eigen/Core>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/model_instance.h>
#include <drake/systems/framework/context.h>

namespace dairlib::allegro_grasp_c3 {

// Owns the mutable scratch context used by static grasp IK.  Its methods take
// explicit targets and return hand-only joint vectors, making each solve
// independently exercisable in a unit test.
class GraspIkSolver {
 public:
  GraspIkSolver(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>* context,
      drake::multibody::ModelInstanceIndex hand_model,
      drake::multibody::ModelInstanceIndex cube_model,
      Eigen::VectorXd cube_positions,
      std::array<drake::multibody::BodyIndex, 4> tip_bodies,
      Eigen::Vector3d fingertip_surface_offset,
      Eigen::Vector3d ring_surface_offset);

  void ResetSeed();

  Eigen::VectorXd SolveThreeFinger(
      const std::string& label,
      const Eigen::VectorXd& fingertip_targets);

  Eigen::VectorXd SolveWithRing(
      const std::string& label,
      const Eigen::VectorXd& fingertip_targets,
      const Eigen::Vector3d& ring_target);

 private:
  const drake::multibody::MultibodyPlant<double>& plant_;
  drake::systems::Context<double>* context_;
  drake::multibody::ModelInstanceIndex hand_model_;
  drake::multibody::ModelInstanceIndex cube_model_;
  Eigen::VectorXd cube_positions_;
  std::array<drake::multibody::BodyIndex, 4> tip_bodies_;
  Eigen::Vector3d fingertip_surface_offset_;
  Eigen::Vector3d ring_surface_offset_;
};

}  // namespace dairlib::allegro_grasp_c3
