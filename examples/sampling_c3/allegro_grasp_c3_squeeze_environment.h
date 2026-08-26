#pragma once

#include <memory>

#include <Eigen/Core>

#include <drake/geometry/meshcat.h>
#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/model_instance.h>
#include <drake/systems/framework/context.h>

#include "allegro_grasp_c3_squeeze_config.h"

namespace dairlib::allegro_grasp_c3 {

// Owns every object whose lifetime is required by the simulated plant:
// Diagram, Simulator, contexts, Meshcat, LCM, and fixed input ports.
class SimulationEnvironment {
 public:
  explicit SimulationEnvironment(const SqueezeConfig& config);
  ~SimulationEnvironment();

  SimulationEnvironment(const SimulationEnvironment&) = delete;
  SimulationEnvironment& operator=(const SimulationEnvironment&) = delete;

  drake::multibody::MultibodyPlant<double>& plant();
  const drake::multibody::MultibodyPlant<double>& plant() const;
  drake::systems::Context<double>& plant_context();
  const drake::systems::Context<double>& plant_context() const;
  drake::multibody::ModelInstanceIndex hand_model() const;
  drake::multibody::ModelInstanceIndex cube_model() const;
  std::shared_ptr<drake::geometry::Meshcat> meshcat() const;

  void SetInitialState(const Eigen::VectorXd& hand_positions,
                       const Eigen::VectorXd& cube_positions);
  void Initialize();
  void PublishContactTarget(const Eigen::VectorXd& hand_positions);
  void PublishState();
  void PublishIndexTracking(const Eigen::VectorXd& actual,
                            const Eigen::VectorXd* plan0,
                            const Eigen::VectorXd* plan1);
  void ApplyHandTorque(const Eigen::VectorXd& torque);
  void AdvanceTo(double time);
  void PinCube(const Eigen::VectorXd& cube_positions);
  void ForcedPublish();

  Eigen::VectorXd state() const;
  Eigen::VectorXd hand_positions() const;
  Eigen::VectorXd hand_velocities() const;
  Eigen::VectorXd cube_positions() const;
  const drake::multibody::ContactResults<double>& contacts() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dairlib::allegro_grasp_c3
