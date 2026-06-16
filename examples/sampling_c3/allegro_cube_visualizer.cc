#include <limits>
#include <memory>

#include <gflags/gflags.h>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/geometry/scene_graph.h>
#include <drake/multibody/parsing/parser.h>
#include "drake/multibody/plant/multibody_plant.h"
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/primitives/constant_vector_source.h>

#include "allegro_hand_utils.h"
#include "common/find_resource.h"

namespace dairlib {

using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::MeshcatVisualizerParams;
using drake::geometry::SceneGraph;
using drake::math::RigidTransform;
using drake::math::RollPitchYaw;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::ConstantVectorSource;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;

DEFINE_double(simulation_time, std::numeric_limits<double>::infinity(),
              "How long to simulate (seconds). Default: run forever.");

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Build the diagram
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);

  // Add Allegro hand (welded to world, palm up)
  ModelInstanceIndex allegro_index = AddAllegroHandToPlant(&plant, &scene_graph);

  // Add numbered cube
  Parser parser(&plant, &scene_graph);
  ModelInstanceIndex cube_index = parser.AddModels(FindResourceOrThrow(
      "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];

  // Finalize plant
  plant.Finalize();

  // Add actuators to all Allegro joints (16 joints total)
  // We'll apply zero torque to keep fingers in their initial position
  const int num_allegro_joints = 16;
  Eigen::VectorXd zero_torques = Eigen::VectorXd::Zero(num_allegro_joints);
  auto* torque_source = builder.AddSystem<ConstantVectorSource<double>>(zero_torques);
  builder.Connect(torque_source->get_output_port(),
                  plant.get_actuation_input_port(allegro_index));

  // Add Meshcat visualization
  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams params;
  params.publish_period = 1.0 / 60.0;
  auto& visualizer = MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));

  // Set camera to isometric view
  meshcat->SetCameraPose(Eigen::Vector3d(0.8, 0.8, 0.8),
                         Eigen::Vector3d(0, 0, 0.5));

  // Build diagram and create simulator
  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);

  simulator.set_target_realtime_rate(1.0);
  simulator.set_publish_every_time_step(false);
  simulator.set_publish_at_initialization(false);

  // Set initial state
  auto& plant_context = diagram->GetMutableSubsystemContext(
      plant, &simulator.get_mutable_context());

  // Set Allegro hand to "open" pose (all fingers extended)
  // Joint names from Allegro URDF: joint_0.0 through joint_15.0
  // Open hand: all joints at 0 (fully extended)
  Eigen::VectorXd q_allegro = Eigen::VectorXd::Zero(num_allegro_joints);
  plant.SetPositions(&plant_context, allegro_index, q_allegro);

  // Position the cube on the palm
  // Palm is at z=0.5 (from weld transform), cube is 6cm tall
  // Place cube center at z=0.53 (just above palm surface)
  // Drake free-body order: [qw, qx, qy, qz, x, y, z]
  Eigen::VectorXd q_cube(7);
  q_cube << 1, 0, 0, 0, 0, 0, 0.53;
  plant.SetPositions(&plant_context, cube_index, q_cube);

  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::DoMain(argc, argv);
}
