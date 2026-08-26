#pragma once

#include <memory>

#include <drake/geometry/meshcat.h>
#include <drake/lcm/drake_lcm.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/model_instance.h>
#include <drake/systems/framework/diagram_builder.h>

namespace drake {
namespace geometry {
template <typename T>
class SceneGraph;
}
}  // namespace drake

namespace dairlib {
namespace systems {
class C3StateSender;
}
namespace allegro_grasp_c3 {

struct SimulationSetupOptions {
  bool isolate_cube{true};
  bool show_ring_markers{false};
  bool show_relay_marker{false};
  bool publish_lcm{false};
  double lcm_publish_hz{100.0};
};

// Makes the simulated collision model match the fingertip-only LCS model.
void ConfigureSimulationCollisions(
    drake::multibody::MultibodyPlant<double>* plant,
    drake::geometry::SceneGraph<double>* scene_graph,
    drake::multibody::ModelInstanceIndex hand_model,
    drake::multibody::ModelInstanceIndex cube_model,
    bool isolate_cube);

std::shared_ptr<drake::geometry::Meshcat> AddGraspVisualization(
    drake::systems::DiagramBuilder<double>* builder,
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::geometry::SceneGraph<double>& scene_graph,
    bool show_ring_markers, bool show_relay_marker);

struct TelemetrySystems {
  std::unique_ptr<drake::lcm::DrakeLcm> lcm;
  systems::C3StateSender* state_sender{};
  systems::C3StateSender* torque_sender{};
  systems::C3StateSender* index_sender{};
};

TelemetrySystems AddGraspTelemetry(
    drake::systems::DiagramBuilder<double>* builder,
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::multibody::ModelInstanceIndex hand_model,
    bool publish, double publish_hz);

}  // namespace allegro_grasp_c3
}  // namespace dairlib
