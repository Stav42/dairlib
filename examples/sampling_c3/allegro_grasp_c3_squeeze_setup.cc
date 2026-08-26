#include "allegro_grasp_c3_squeeze_setup.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <drake/common/drake_assert.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/geometry/rgba.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/lcm/drake_lcm.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/systems/lcm/lcm_interface_system.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

#include "meshcat_utils.h"
#include "systems/senders/c3_state_sender.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::geometry::GeometryId;
using drake::geometry::GeometrySet;
using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::MeshcatVisualizerParams;
using drake::multibody::BodyIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using Eigen::Vector3d;

std::vector<GeometryId> CollisionGeometryForBodies(
    const MultibodyPlant<double>& plant, ModelInstanceIndex model,
    const std::vector<std::string>& body_names) {
  std::vector<GeometryId> result;
  for (const std::string& name : body_names) {
    const auto& geometries = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName(name, model));
    result.insert(result.end(), geometries.begin(), geometries.end());
  }
  return result;
}

}  // namespace

void ConfigureSimulationCollisions(
    MultibodyPlant<double>* plant, drake::geometry::SceneGraph<double>* scene_graph,
    ModelInstanceIndex hand_model, ModelInstanceIndex cube_model,
    bool isolate_cube) {
  auto manager = scene_graph->collision_filter_manager();
  const auto thumb = CollisionGeometryForBodies(
      *plant, hand_model,
      {"link_12", "link_13", "link_14", "link_15", "link_15_tip"});
  const auto palm = plant->GetCollisionGeometriesForBody(
      plant->GetBodyByName("palm_link", hand_model));
  manager.Apply(drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
      GeometrySet(thumb), GeometrySet(palm)));

  const std::array<std::vector<GeometryId>, 3> fingers{
      CollisionGeometryForBodies(
          *plant, hand_model,
          {"link_0", "link_1", "link_2", "link_3", "link_3_tip"}),
      CollisionGeometryForBodies(
          *plant, hand_model,
          {"link_4", "link_5", "link_6", "link_7", "link_7_tip"}),
      CollisionGeometryForBodies(
          *plant, hand_model,
          {"link_8", "link_9", "link_10", "link_11", "link_11_tip"})};
  for (int first = 0; first < 3; ++first) {
    for (int second = first + 1; second < 3; ++second) {
      manager.Apply(
          drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
              GeometrySet(fingers[first]), GeometrySet(fingers[second])));
    }
  }
  std::cout << "[setup] finger-finger collision filter APPLIED: "
               "index/middle/ring excluded from colliding with each other.\n";

  if (!isolate_cube) {
    std::cout << "[setup] cube-isolation filter SKIPPED "
                 "(--isolate_cube=false): full-hand collision active.\n";
    return;
  }
  const std::array<std::string, 4> tip_names{
      "link_3_tip", "link_7_tip", "link_15_tip", "link_11_tip"};
  std::vector<GeometryId> non_tip_geometries;
  for (BodyIndex index : plant->GetBodyIndices(hand_model)) {
    const auto& body = plant->get_body(index);
    if (std::find(tip_names.begin(), tip_names.end(), body.name()) !=
        tip_names.end()) {
      continue;
    }
    const auto& geometries = plant->GetCollisionGeometriesForBody(body);
    non_tip_geometries.insert(non_tip_geometries.end(), geometries.begin(),
                              geometries.end());
  }
  const auto cube_geometries = plant->GetCollisionGeometriesForBody(
      plant->get_body(plant->GetBodyIndices(cube_model)[0]));
  manager.Apply(drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
      GeometrySet(cube_geometries), GeometrySet(non_tip_geometries)));
  std::cout << "[setup] cube-isolation filter APPLIED: cube collides only "
               "with the 4 fingertips ("
            << non_tip_geometries.size() << " non-tip geoms excluded).\n";
}

std::shared_ptr<Meshcat> AddGraspVisualization(
    drake::systems::DiagramBuilder<double>* builder,
    const MultibodyPlant<double>& plant,
    drake::geometry::SceneGraph<double>& scene_graph,
    bool show_ring_markers, bool show_relay_marker) {
  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams parameters;
  parameters.publish_period = 1.0 / 60.0;
  MeshcatVisualizer<double>::AddToBuilder(builder, scene_graph, meshcat,
                                          std::move(parameters));
  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      builder, plant, meshcat);
  meshcat->SetCameraPose(Vector3d(0.7, 0.7, 0.8), Vector3d(0, 0, 0.55));
  AddFrameTriad(meshcat.get(), "/cube_frame", 0.005, 0.1);
  for (const char* path : {"/tip_frame/index", "/tip_frame/middle",
                           "/tip_frame/thumb"}) {
    AddFrameTriad(meshcat.get(), path, 0.0015, 0.02);
  }
  if (show_ring_markers) {
    meshcat->SetObject("/tip_frame/ring_surface",
                       drake::geometry::Box(0.004, 0.004, 0.004),
                       drake::geometry::Rgba(1, 1, 0, 1));
  }
  const drake::geometry::Rgba red(1, 0, 0, 1);
  for (const char* path : {"/grasp/index", "/grasp/middle", "/grasp/thumb"}) {
    meshcat->SetObject(path, drake::geometry::Sphere(0.006), red);
  }
  if (show_ring_markers) {
    meshcat->SetObject("/grasp/ring", drake::geometry::Sphere(0.006), red);
  }
  if (show_relay_marker) {
    meshcat->SetObject("/gait/ring_touch_target",
                       drake::geometry::Sphere(0.006),
                       drake::geometry::Rgba(0, 1, 1, 1));
  }
  return meshcat;
}

TelemetrySystems AddGraspTelemetry(
    drake::systems::DiagramBuilder<double>* builder,
    const MultibodyPlant<double>& plant, ModelInstanceIndex hand_model,
    bool publish, double publish_hz) {
  TelemetrySystems result;
  if (!publish) return result;

  result.lcm = std::make_unique<drake::lcm::DrakeLcm>();
  auto* interface = builder->AddSystem<drake::systems::lcm::LcmInterfaceSystem>(
      result.lcm.get());
  const int full_state_size = plant.num_positions() + plant.num_velocities();
  std::vector<std::string> state_names;
  for (int i = 0; i < plant.num_positions(hand_model); ++i)
    state_names.push_back("hand_q" + std::to_string(i));
  state_names.insert(state_names.end(), {"cube_qw", "cube_qx", "cube_qy",
                                          "cube_qz", "cube_x", "cube_y",
                                          "cube_z"});
  for (int i = 0; i < plant.num_velocities(hand_model); ++i)
    state_names.push_back("hand_v" + std::to_string(i));
  state_names.insert(state_names.end(), {"cube_wx", "cube_wy", "cube_wz",
                                          "cube_vx", "cube_vy", "cube_vz"});
  DRAKE_DEMAND(static_cast<int>(state_names.size()) == full_state_size);
  result.state_sender = builder->AddSystem<systems::C3StateSender>(
      full_state_size, state_names);
  auto* state_publisher = builder->AddSystem(
      drake::systems::lcm::LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
          "GRASP_STATE", interface, 1.0 / publish_hz));
  builder->Connect(result.state_sender->get_output_port_target_c3_state(),
                   state_publisher->get_input_port());
  auto* target_publisher = builder->AddSystem(
      drake::systems::lcm::LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
          "GRASP_Q_CONTACT", interface, 1.0 / publish_hz));
  builder->Connect(result.state_sender->get_output_port_final_target_c3_state(),
                   target_publisher->get_input_port());

  std::vector<std::string> torque_names;
  for (int i = 0; i < plant.num_positions(hand_model); ++i)
    torque_names.push_back("tau" + std::to_string(i));
  result.torque_sender = builder->AddSystem<systems::C3StateSender>(
      plant.num_positions(hand_model), torque_names);
  result.torque_sender->set_name("tau_sender");
  auto* torque_publisher = builder->AddSystem(
      drake::systems::lcm::LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
          "GRASP_TAU", interface, 1.0 / publish_hz));
  builder->Connect(result.torque_sender->get_output_port_target_c3_state(),
                   torque_publisher->get_input_port());

  const std::vector<std::string> index_names{
      "act_q0", "act_q1", "act_q2", "act_q3", "plan0_q0", "plan0_q1",
      "plan0_q2", "plan0_q3", "plan1_q0", "plan1_q1", "plan1_q2",
      "plan1_q3"};
  result.index_sender =
      builder->AddSystem<systems::C3StateSender>(12, index_names);
  result.index_sender->set_name("idx_sender");
  auto* index_publisher = builder->AddSystem(
      drake::systems::lcm::LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
          "GRASP_IDX_TRACK", interface, 1.0 / publish_hz));
  builder->Connect(result.index_sender->get_output_port_target_c3_state(),
                   index_publisher->get_input_port());
  return result;
}

}  // namespace dairlib::allegro_grasp_c3
