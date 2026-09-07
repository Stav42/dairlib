#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <drake/geometry/geometry_ids.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/model_instance.h>

#include "c3/core/lcs.h"
#include "c3/multibody/lcs_factory_options.h"

namespace dairlib::allegro_grasp_c3 {

// Owns the continuous plant and AutoDiff context used only for C3
// linearization. Simulation state is passed in explicitly; this class never
// reads or mutates the live simulator.
class LcsModel {
 public:
  explicit LcsModel(double cube_size_scale);
  ~LcsModel();

  LcsModel(const LcsModel&) = delete;
  LcsModel& operator=(const LcsModel&) = delete;

  int state_size() const;
  int input_size() const;
  int position_size() const;
  int hand_position_size() const;
  int hand_velocity_size() const;

  c3::LCS Linearize(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& input,
                    const std::vector<int>& active_fingers,
                    const c3::LCSFactoryOptions& options);

  // Diagnostic accessors. Control code should normally use Linearize().
  drake::multibody::MultibodyPlant<double>& plant();
  drake::systems::Context<double>& context();
  drake::multibody::ModelInstanceIndex cube_model() const;
  const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
  contact_pairs() const;
  // World-frame force directions corresponding one-for-one to the contact
  // variables in the most recent Linearize() call.  With the finger/cube
  // ordering used here, each direction is the force applied to the cube.
  const std::vector<Eigen::Vector3d>& contact_force_bases() const;
  // World-frame witness/contact points, one per active finger and in the
  // same order as contact_pairs(), from the most recent Linearize() call.
  // Each point is the midpoint of the two collider witness points.
  const std::vector<Eigen::Vector3d>& contact_points_world() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dairlib::allegro_grasp_c3
