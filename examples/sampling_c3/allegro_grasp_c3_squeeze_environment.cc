#include "allegro_grasp_c3_squeeze_environment.h"

#include <utility>

#include <drake/geometry/scene_graph.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/fixed_input_port_value.h>

#include "allegro_grasp_c3_squeeze_setup.h"
#include "allegro_hand_utils.h"
#include "common/find_resource.h"
#include "systems/senders/c3_state_sender.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::FixedInputPortValue;
using drake::systems::Simulator;
using Eigen::VectorXd;

}  // namespace

struct SimulationEnvironment::Impl {
  explicit Impl(const SqueezeConfig& config) {
    auto [plant_ref, scene_graph_ref] =
        AddMultibodyPlantSceneGraph(&builder, 0.001);
    plant = &plant_ref;
    scene_graph = &scene_graph_ref;
    hand = AddAllegroHandToPlant(plant, scene_graph);
    Parser parser(plant, scene_graph);
    cube = parser.AddModels(FindResourceOrThrow(
        "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];
    plant->set_discrete_contact_approximation(
        drake::multibody::DiscreteContactApproximation::kSap);
    plant->Finalize();

    ConfigureSimulationCollisions(plant, scene_graph, hand, cube,
                                  config.isolate_cube);
    meshcat = AddGraspVisualization(
        &builder, *plant, *scene_graph, config.release_middle,
        config.gait && config.gait_scheme == "relay");
    telemetry = AddGraspTelemetry(&builder, *plant, hand, config.lcm_publish,
                                  config.lcm_publish_hz);

    diagram = builder.Build();
    simulator = std::make_unique<Simulator<double>>(*diagram);
    simulator->set_target_realtime_rate(1.0);
    diagram_context = &simulator->get_mutable_context();
    plant_context =
        &diagram->GetMutableSubsystemContext(*plant, diagram_context);

    if (telemetry.state_sender != nullptr) {
      auto& sender_context = diagram->GetMutableSubsystemContext(
          *telemetry.state_sender, diagram_context);
      const int full_state_size = plant->num_positions() + plant->num_velocities();
      state_input = &telemetry.state_sender->get_input_port_target_state()
                         .FixValue(&sender_context,
                                   VectorXd::Zero(full_state_size));
      contact_target_input =
          &telemetry.state_sender->get_input_port_final_target_state().FixValue(
              &sender_context, VectorXd::Zero(full_state_size));
    }
    if (telemetry.torque_sender != nullptr) {
      auto& sender_context = diagram->GetMutableSubsystemContext(
          *telemetry.torque_sender, diagram_context);
      torque_input = &telemetry.torque_sender->get_input_port_target_state()
                          .FixValue(&sender_context,
                                    VectorXd::Zero(plant->num_positions(hand)));
    }
    if (telemetry.index_sender != nullptr) {
      auto& sender_context = diagram->GetMutableSubsystemContext(
          *telemetry.index_sender, diagram_context);
      index_input = &telemetry.index_sender->get_input_port_target_state()
                         .FixValue(&sender_context, VectorXd::Zero(12));
    }
    actuation_input = &plant->get_actuation_input_port(hand).FixValue(
        plant_context, VectorXd::Zero(plant->num_actuators(hand)));
  }

  DiagramBuilder<double> builder;
  MultibodyPlant<double>* plant{};
  drake::geometry::SceneGraph<double>* scene_graph{};
  ModelInstanceIndex hand;
  ModelInstanceIndex cube;
  std::shared_ptr<drake::geometry::Meshcat> meshcat;
  TelemetrySystems telemetry;
  std::unique_ptr<Diagram<double>> diagram;
  std::unique_ptr<Simulator<double>> simulator;
  drake::systems::Context<double>* diagram_context{};
  drake::systems::Context<double>* plant_context{};
  FixedInputPortValue* state_input{};
  FixedInputPortValue* contact_target_input{};
  FixedInputPortValue* torque_input{};
  FixedInputPortValue* index_input{};
  FixedInputPortValue* actuation_input{};
};

SimulationEnvironment::SimulationEnvironment(const SqueezeConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

SimulationEnvironment::~SimulationEnvironment() = default;

MultibodyPlant<double>& SimulationEnvironment::plant() { return *impl_->plant; }
const MultibodyPlant<double>& SimulationEnvironment::plant() const {
  return *impl_->plant;
}
drake::systems::Context<double>& SimulationEnvironment::plant_context() {
  return *impl_->plant_context;
}
const drake::systems::Context<double>& SimulationEnvironment::plant_context()
    const {
  return *impl_->plant_context;
}
ModelInstanceIndex SimulationEnvironment::hand_model() const {
  return impl_->hand;
}
ModelInstanceIndex SimulationEnvironment::cube_model() const {
  return impl_->cube;
}
std::shared_ptr<drake::geometry::Meshcat> SimulationEnvironment::meshcat()
    const {
  return impl_->meshcat;
}

void SimulationEnvironment::SetInitialState(const VectorXd& hand_positions,
                                            const VectorXd& cube_positions) {
  impl_->plant->SetPositions(impl_->plant_context, impl_->hand, hand_positions);
  impl_->plant->SetVelocities(impl_->plant_context, impl_->hand,
                              VectorXd::Zero(impl_->plant->num_velocities(impl_->hand)));
  PinCube(cube_positions);
}

void SimulationEnvironment::Initialize() { impl_->simulator->Initialize(); }

void SimulationEnvironment::PublishContactTarget(const VectorXd& hand_positions) {
  if (impl_->contact_target_input == nullptr) return;
  VectorXd target = VectorXd::Zero(impl_->plant->num_positions() +
                                   impl_->plant->num_velocities());
  target.head(hand_positions.size()) = hand_positions;
  impl_->contact_target_input->GetMutableVectorData<double>()->SetFromVector(target);
}

void SimulationEnvironment::PublishState() {
  if (impl_->state_input != nullptr) {
    impl_->state_input->GetMutableVectorData<double>()->SetFromVector(state());
  }
}

void SimulationEnvironment::PublishIndexTracking(
    const VectorXd& actual, const VectorXd* plan0, const VectorXd* plan1) {
  if (impl_->index_input == nullptr) return;
  VectorXd values(12);
  values.segment(0, 4) = actual;
  values.segment(4, 4) = plan0 == nullptr ? actual : *plan0;
  values.segment(8, 4) = plan1 == nullptr ? actual : *plan1;
  impl_->index_input->GetMutableVectorData<double>()->SetFromVector(values);
}

void SimulationEnvironment::ApplyHandTorque(const VectorXd& torque) {
  if (impl_->torque_input != nullptr) {
    impl_->torque_input->GetMutableVectorData<double>()->SetFromVector(torque);
  }
  impl_->actuation_input->GetMutableVectorData<double>()->SetFromVector(torque);
}

void SimulationEnvironment::AdvanceTo(double time) {
  impl_->simulator->AdvanceTo(time);
}

void SimulationEnvironment::PinCube(const VectorXd& cube_positions) {
  impl_->plant->SetPositions(impl_->plant_context, impl_->cube, cube_positions);
  impl_->plant->SetVelocities(impl_->plant_context, impl_->cube,
                              VectorXd::Zero(impl_->plant->num_velocities(impl_->cube)));
}

void SimulationEnvironment::ForcedPublish() {
  impl_->diagram->ForcedPublish(*impl_->diagram_context);
}

VectorXd SimulationEnvironment::state() const {
  return impl_->plant->GetPositionsAndVelocities(*impl_->plant_context);
}
VectorXd SimulationEnvironment::hand_positions() const {
  return impl_->plant->GetPositions(*impl_->plant_context, impl_->hand);
}
VectorXd SimulationEnvironment::hand_velocities() const {
  return impl_->plant->GetVelocities(*impl_->plant_context, impl_->hand);
}
VectorXd SimulationEnvironment::cube_positions() const {
  return impl_->plant->GetPositions(*impl_->plant_context, impl_->cube);
}
const drake::multibody::ContactResults<double>& SimulationEnvironment::contacts()
    const {
  return impl_->plant->get_contact_results_output_port().Eval<
      drake::multibody::ContactResults<double>>(*impl_->plant_context);
}

}  // namespace dairlib::allegro_grasp_c3
