#include <limits>
#include <memory>

#include <gflags/gflags.h>

#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/coulomb_friction.h>
#include "drake/multibody/plant/multibody_plant.h"
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/primitives/constant_vector_source.h>

#include "allegro_hand_utils.h"
#include "common/find_resource.h"
#include "cube_kinematics.h"
#include "meshcat_utils.h"
#include "pd_joint_controller.h"

#include <iostream>

namespace dairlib {

using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::MeshcatVisualizerParams;
using drake::math::RigidTransform;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::ConstantVectorSource;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;

DEFINE_double(simulation_time, std::numeric_limits<double>::infinity(),
              "How long to simulate (seconds). Default: run forever.");
DEFINE_double(kp, 50.0, "PD proportional gain (Nm/rad).");
DEFINE_double(kd, 5.0, "PD derivative gain (Nm*s/rad).");
DEFINE_double(settle_time, 2.0,
              "Seconds to let the cube fall and settle before solving IK.");

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);

  ModelInstanceIndex allegro_index =
      AddAllegroHandToPlant(&plant, &scene_graph);

  Parser parser(&plant, &scene_graph);
  ModelInstanceIndex cube_index = parser.AddModels(FindResourceOrThrow(
      "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];

  plant.RegisterCollisionGeometry(
      plant.world_body(), drake::math::RigidTransform<double>::Identity(),
      drake::geometry::HalfSpace(), "ground_collision",
      drake::multibody::CoulombFriction<double>(0.8, 0.5));
  plant.RegisterVisualGeometry(
      plant.world_body(), drake::math::RigidTransform<double>::Identity(),
      drake::geometry::HalfSpace(), "ground_visual",
      Eigen::Vector4d(0.5, 0.5, 0.5, 0.5));

  plant.Finalize();

  const int num_allegro_joints = plant.num_positions(allegro_index);

  // Placeholder zeros — updated after IK solves.
  auto* q_des_source = builder.AddSystem<ConstantVectorSource<double>>(
      Eigen::VectorXd::Zero(num_allegro_joints));

  auto* controller = builder.AddSystem<PdJointController>(
      num_allegro_joints, FLAGS_kp, FLAGS_kd);

  builder.Connect(plant.get_state_output_port(allegro_index),
                  controller->GetInputPort("state"));
  builder.Connect(q_des_source->get_output_port(),
                  controller->GetInputPort("q_desired"));
  builder.Connect(controller->get_output_port(0),
                  plant.get_actuation_input_port(allegro_index));

  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams params;
  params.publish_period = 1.0 / 60.0;
  MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));
  meshcat->SetCameraPose(Eigen::Vector3d(0.8, 0.8, 0.8),
                         Eigen::Vector3d(0, 0, 0.5));
  meshcat->AddButton("Respawn Cube", "KeyR");

  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  simulator.set_publish_every_time_step(false);
  simulator.set_publish_at_initialization(false);

  auto& plant_context = diagram->GetMutableSubsystemContext(
      plant, &simulator.get_mutable_context());

  plant.SetPositions(&plant_context, allegro_index,
                     Eigen::VectorXd::Zero(num_allegro_joints));

  auto get_cube_pose = [&]() {
    return plant.GetPositions(plant_context, cube_index);
  };

  auto spawn_cube = [&]() {
    Eigen::VectorXd q_cube(7);
    q_cube << 1, 0, 0, 0, 0, 0.02, 0.8;
    plant.SetPositions(&plant_context, cube_index, q_cube);
    plant.SetVelocities(&plant_context, cube_index,
                        Eigen::VectorXd::Zero(6));
  };

  AddFrameTriad(meshcat.get(), "/cube_frame", 0.005, 0.1);

  const Eigen::Vector4d grasp_mags =
      GetGraspMagnitudes(-0.02, 0.02, 0.0, 1.0);

  spawn_cube();
  simulator.Initialize();

  const double cube_size = 0.06;
  const double poll_step = 1.0 / 30.0;
  int last_clicks = 0;

  // Phase 1: cube falls and settles; fingers stay at zero (q_desired = 0).
  for (double t = poll_step; t < FLAGS_settle_time; t += poll_step) {
    simulator.AdvanceTo(t);
    RigidTransform<double> X_WC = CubePoseFromPositions(get_cube_pose());
    meshcat->SetTransform("/cube_frame", X_WC);
    int clicks = meshcat->GetButtonClicks("Respawn Cube");
    if (clicks > last_clicks) {
      last_clicks = clicks;
      spawn_cube();
    }
  }

  // Save settled cube state before IK perturbs the plant context.
  Eigen::VectorXd q_cube_settled = get_cube_pose();
  Eigen::VectorXd v_cube_settled =
      plant.GetVelocities(plant_context, cube_index);

  // Solve IK from the settled cube pose to find finger joint targets.
  Eigen::VectorXd grasp_positions = GetGraspPositions(
      CubePoseFromPositions(q_cube_settled), cube_size, -0.02, 0.02, 0.0);
  Eigen::VectorXd q_ik = SolveGraspIK(plant, &plant_context, grasp_positions);

  // Extract just the Allegro joints from the full IK solution.
  plant.SetPositions(&plant_context, q_ik);
  Eigen::VectorXd q_allegro = plant.GetPositions(plant_context, allegro_index);

  // Restore cube to its settled state and fingers to zero so PD can drive them.
  plant.SetPositions(&plant_context, cube_index, q_cube_settled);
  plant.SetVelocities(&plant_context, cube_index, v_cube_settled);
  plant.SetPositions(&plant_context, allegro_index,
                     Eigen::VectorXd::Zero(num_allegro_joints));
  plant.SetVelocities(&plant_context, allegro_index,
                      Eigen::VectorXd::Zero(num_allegro_joints));

  // Update q_desired now that IK has solved.
  auto& source_context = diagram->GetMutableSubsystemContext(
      *q_des_source, &simulator.get_mutable_context());
  q_des_source->get_mutable_source_value(&source_context)
      .SetFromVector(q_allegro);

  // Phase 2: PD drives fingers toward contact while cube stays settled.
  for (double t = FLAGS_settle_time + poll_step;
       t < FLAGS_simulation_time; t += poll_step) {
    simulator.AdvanceTo(t);
    RigidTransform<double> X_WC = CubePoseFromPositions(get_cube_pose());
    meshcat->SetTransform("/cube_frame", X_WC);
    int clicks = meshcat->GetButtonClicks("Respawn Cube");
    if (clicks > last_clicks) {
      last_clicks = clicks;
      spawn_cube();
    }
  }

  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::DoMain(argc, argv);
}
