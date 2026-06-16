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
#include <drake/systems/framework/leaf_system.h>

#include "allegro_hand_utils.h"
#include "common/find_resource.h"

namespace dairlib {

using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::MeshcatVisualizerParams;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::LeafSystem;
using drake::systems::Simulator;

DEFINE_double(simulation_time, std::numeric_limits<double>::infinity(),
              "How long to simulate (seconds). Default: run forever.");
DEFINE_double(kp, 50.0, "PD proportional gain (Nm/rad).");
DEFINE_double(kd, 5.0, "PD derivative gain (Nm*s/rad).");

// Holds all joints at q_desired via PD feedback on the per-instance state port.
class PdJointController : public LeafSystem<double> {
 public:
  PdJointController(int num_joints, Eigen::VectorXd q_desired,
                    double kp, double kd)
      : num_joints_(num_joints),
        q_desired_(std::move(q_desired)),
        kp_(kp),
        kd_(kd) {
    state_port_ =
        DeclareVectorInputPort("state", 2 * num_joints).get_index();
    DeclareVectorOutputPort("torques", num_joints,
                            &PdJointController::CalcTorques);
  }

 private:
  void CalcTorques(const Context<double>& context,
                   BasicVector<double>* output) const {
    const Eigen::VectorXd state = get_input_port(state_port_).Eval(context);
    const Eigen::VectorXd q = state.head(num_joints_);
    const Eigen::VectorXd v = state.tail(num_joints_);
    output->SetFromVector(kp_ * (q_desired_ - q) - kd_ * v);
  }

  const int num_joints_;
  const Eigen::VectorXd q_desired_;
  const double kp_;
  const double kd_;
  int state_port_;
};

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);

  ModelInstanceIndex allegro_index = AddAllegroHandToPlant(&plant, &scene_graph);

  Parser parser(&plant, &scene_graph);
  ModelInstanceIndex cube_index = parser.AddModels(FindResourceOrThrow(
      "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];

  plant.Finalize();

  // PD controller: hold all 16 joints at q=0 (fingers fully extended).
  const int num_allegro_joints = plant.num_positions(allegro_index);
  auto* controller = builder.AddSystem<PdJointController>(
      num_allegro_joints, Eigen::VectorXd::Zero(num_allegro_joints),
      FLAGS_kp, FLAGS_kd);

  // plant.get_state_output_port(allegro_index) outputs [q(16), v(16)].
  builder.Connect(plant.get_state_output_port(allegro_index),
                  controller->get_input_port(0));
  builder.Connect(controller->get_output_port(0),
                  plant.get_actuation_input_port(allegro_index));

  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams params;
  params.publish_period = 1.0 / 60.0;
  MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));

  meshcat->SetCameraPose(Eigen::Vector3d(0.8, 0.8, 0.8),
                         Eigen::Vector3d(0, 0, 0.5));

  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  simulator.set_publish_every_time_step(false);
  simulator.set_publish_at_initialization(false);

  auto& plant_context = diagram->GetMutableSubsystemContext(
      plant, &simulator.get_mutable_context());

  // Allegro: all joints at 0 (fully extended, palm up).
  plant.SetPositions(&plant_context, allegro_index,
                     Eigen::VectorXd::Zero(num_allegro_joints));

  // Cube: identity rotation, centered 3cm above the palm surface.
  // Drake free-body order: [qw, qx, qy, qz, x, y, z]
  Eigen::VectorXd q_cube(7);
  q_cube << 1, 0, 0, 0, 0, 0, 0.8;
  plant.SetPositions(&plant_context, cube_index, q_cube);

  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::DoMain(argc, argv);
}
