#include <algorithm>
#include <limits>
#include <memory>

#include <gflags/gflags.h>

#include <drake/geometry/collision_filter_declaration.h>
#include <drake/geometry/geometry_set.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
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
#include <drake/common/trajectories/piecewise_polynomial.h>

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
using drake::trajectories::PiecewisePolynomial;

DEFINE_double(simulation_time, std::numeric_limits<double>::infinity(),
              "How long to simulate (seconds). Default: run forever.");
DEFINE_double(kp, 15.0, "PD proportional gain (Nm/rad).");
DEFINE_double(kd, 2.0, "PD derivative gain (Nm*s/rad).");
DEFINE_double(settle_time, 2.0,
              "Seconds to let the cube fall and settle before solving IK.");

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);
  // plant.set_discrete_contact_approximation(
  //     drake::multibody::DiscreteContactApproximation::kSap);

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

  // Small thin platform 0.2 m above the palm (palm at z=0.5 → top at z=0.7).
  // Cube centre rests at z = 0.7 + 0.03 = 0.73 m.
  const drake::math::RigidTransform<double> X_WPlatform(
      Eigen::Vector3d(0.0, 0.0, 0.545));  // box centre (10 mm thick)
  const drake::geometry::GeometryId platform_collision_id =
      plant.RegisterCollisionGeometry(
          plant.world_body(), X_WPlatform,
          drake::geometry::Box(0.08, 0.08, 0.01), "platform_collision",
          drake::multibody::CoulombFriction<double>(0.8, 0.5));
  plant.RegisterVisualGeometry(
      plant.world_body(), X_WPlatform,
      drake::geometry::Box(0.08, 0.08, 0.01), "platform_visual",
      Eigen::Vector4d(0.55, 0.35, 0.15, 1.0));

  plant.Finalize();

  // Collect all Allegro collision geometries and exclude them from the platform,
  // so the platform only interacts with the cube.
  std::vector<drake::geometry::GeometryId> allegro_collision_geoms;
  for (const auto body_idx : plant.GetBodyIndices(allegro_index)) {
    const auto& geoms =
        plant.GetCollisionGeometriesForBody(plant.get_body(body_idx));
    allegro_collision_geoms.insert(
        allegro_collision_geoms.end(), geoms.begin(), geoms.end());
  }
  scene_graph.collision_filter_manager().Apply(
      drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
          drake::geometry::GeometrySet({platform_collision_id}),
          drake::geometry::GeometrySet(allegro_collision_geoms)));

  // Exclude all thumb links ↔ palm collisions.
  // Only link_15_tip was excluded before; joints 12 and 13 were stuck at their
  // limits because link_12/link_13 were still colliding with the palm.
  {
    std::vector<drake::geometry::GeometryId> thumb_geoms;
    for (const char* name :
         {"link_12", "link_13", "link_14", "link_15", "link_15_tip"}) {
      const auto& g = plant.GetCollisionGeometriesForBody(
          plant.GetBodyByName(name, allegro_index));
      thumb_geoms.insert(thumb_geoms.end(), g.begin(), g.end());
    }
    scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet(thumb_geoms),
            drake::geometry::GeometrySet(
                plant.GetCollisionGeometriesForBody(
                    plant.GetBodyByName("palm_link", allegro_index)))));
  }

  const int num_allegro_joints = plant.num_positions(allegro_index);

  // Both sources start at zero; updated each poll step during Phase 2.
  auto* q_des_source = builder.AddSystem<ConstantVectorSource<double>>(
      Eigen::VectorXd::Zero(num_allegro_joints));
  auto* qdot_des_source = builder.AddSystem<ConstantVectorSource<double>>(
      Eigen::VectorXd::Zero(num_allegro_joints));

  auto* controller = builder.AddSystem<PdJointController>(
      num_allegro_joints, FLAGS_kp, FLAGS_kd);

  builder.Connect(plant.get_state_output_port(allegro_index),
                  controller->GetInputPort("state"));
  builder.Connect(q_des_source->get_output_port(),
                  controller->GetInputPort("q_desired"));
  builder.Connect(qdot_des_source->get_output_port(),
                  controller->GetInputPort("qdot_desired"));
  builder.Connect(controller->get_output_port(0),
                  plant.get_actuation_input_port(allegro_index));

  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams params;
  params.publish_period = 1.0 / 60.0;
  MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));

  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat);
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
    q_cube << 1, 0, 0, 0, 0, 0, 0.8;  // drop from above platform centre
    plant.SetPositions(&plant_context, cube_index, q_cube);
    plant.SetVelocities(&plant_context, cube_index,
                        Eigen::VectorXd::Zero(6));
  };

  AddFrameTriad(meshcat.get(), "/cube_frame", 0.005, 0.1);

  const double cube_size = 0.06;
  const double grasp_a = -0.02, grasp_b = 0.02, grasp_c = 0.0;

  // Add red spheres at the three contact points; positions updated each loop.
  const drake::geometry::Rgba red(1.0, 0.0, 0.0, 1.0);
  meshcat->SetObject("/grasp/index",  drake::geometry::Sphere(0.008), red);
  meshcat->SetObject("/grasp/middle", drake::geometry::Sphere(0.008), red);
  meshcat->SetObject("/grasp/thumb",  drake::geometry::Sphere(0.008), red);

  auto update_grasp_spheres = [&](const RigidTransform<double>& X_WC) {
    Eigen::VectorXd gp =
        GetGraspPositions(X_WC, cube_size, grasp_a, grasp_b, grasp_c);
    meshcat->SetTransform("/grasp/index",
                          RigidTransform<double>(gp.segment<3>(0)));
    meshcat->SetTransform("/grasp/middle",
                          RigidTransform<double>(gp.segment<3>(3)));
    meshcat->SetTransform("/grasp/thumb",
                          RigidTransform<double>(gp.segment<3>(6)));
  };

  spawn_cube();
  simulator.Initialize();

  const double poll_step = 1.0 / 30.0;
  int last_clicks = 0;

  // Phase 1: cube falls and settles; fingers stay at zero (q_desired = 0).
  for (double t = poll_step; t < FLAGS_settle_time; t += poll_step) {
    simulator.AdvanceTo(t);
    RigidTransform<double> X_WC = CubePoseFromPositions(get_cube_pose());
    meshcat->SetTransform("/cube_frame", X_WC);
    update_grasp_spheres(X_WC);
    int clicks = meshcat->GetButtonClicks("Respawn Cube");
    if (clicks > last_clicks) {
      last_clicks = clicks;
      spawn_cube();
    }
  }

  // Save settled cube state before IK perturbs the plant context.
  const Eigen::VectorXd q_cube_settled = get_cube_pose();
  const Eigen::VectorXd v_cube_settled =
      plant.GetVelocities(plant_context, cube_index);
  const RigidTransform<double> X_WC_settled =
      CubePoseFromPositions(q_cube_settled);

  // Fingertip targets in world frame.
  // Contact targets are placed 5 mm inside each cube face so the PD maintains
  // a small constant inward force at the surface → stable contact equilibrium.
  const double penetration = 0.005;
  const Eigen::VectorXd pregrasp_positions =
      GetIntermediatePosition(X_WC_settled, cube_size, grasp_a, grasp_b, grasp_c);
  const Eigen::VectorXd contact_positions =
      GetGraspPositions(X_WC_settled, cube_size - 2 * penetration,
                        grasp_a, grasp_b, grasp_c);

  // Seeds thumb joints 12-15 before each IK call so IPOPT doesn't start from
  // the fully-retracted zero pose.
  auto set_thumb_seed = [&]() {
    Eigen::VectorXd q_seed = Eigen::VectorXd::Zero(num_allegro_joints);
    q_seed[12] = 1.0;
    q_seed[13] = 0.5;
    q_seed[14] = 0.5;
    q_seed[15] = 0.3;
    plant.SetPositions(&plant_context, allegro_index, q_seed);
  };

  // IK for pre-grasp (fingertips just outside each cube face).
  set_thumb_seed();
  plant.SetPositions(&plant_context,
                     SolveGraspIK(plant, &plant_context, pregrasp_positions));
  const Eigen::VectorXd q_pregrasp =
      plant.GetPositions(plant_context, allegro_index);

  // IK for contact (fingertips at cube faces). Reset cube first in case it
  // drifted during the pre-grasp IK solve.
  plant.SetPositions(&plant_context, cube_index, q_cube_settled);
  set_thumb_seed();
  plant.SetPositions(&plant_context,
                     SolveGraspIK(plant, &plant_context, contact_positions));
  const Eigen::VectorXd q_contact =
      plant.GetPositions(plant_context, allegro_index);

  // Diagnostic: check thumb tip FK at the contact IK solution.
  {
    const Eigen::Vector3d p_thumb_target = contact_positions.segment<3>(6);
    const Eigen::Vector3d p_thumb_fk =
        plant.EvalBodyPoseInWorld(
            plant_context,
            plant.GetBodyByName("link_15_tip", allegro_index)).translation();
    std::cout << "thumb target (world): " << p_thumb_target.transpose() << "\n";
    std::cout << "thumb FK @ q_contact: " << p_thumb_fk.transpose() << "\n";
    std::cout << "thumb FK error:       "
              << (p_thumb_fk - p_thumb_target).norm() << " m\n";
  }

  // Restore settled state; PD will drive fingers from open (zero) along spline.
  plant.SetPositions(&plant_context, cube_index, q_cube_settled);
  plant.SetVelocities(&plant_context, cube_index, v_cube_settled);
  plant.SetPositions(&plant_context, allegro_index,
                     Eigen::VectorXd::Zero(num_allegro_joints));
  plant.SetVelocities(&plant_context, allegro_index,
                      Eigen::VectorXd::Zero(num_allegro_joints));

  // Cubic spline: open (t=0) → pre-grasp (t=T1) → contact (t=T2).
  // CubicShapePreserving with zero_end_point_derivatives gives smooth
  // acceleration from rest and deceleration to rest at contact.
  const double T_pregrasp = 2.0;
  const double T_contact  = 4.0;

  std::vector<Eigen::MatrixXd> traj_samples(3,
      Eigen::MatrixXd(num_allegro_joints, 1));
  traj_samples[0] = Eigen::VectorXd::Zero(num_allegro_joints);
  traj_samples[1] = q_pregrasp;
  traj_samples[2] = q_contact;

  const auto traj = PiecewisePolynomial<double>::CubicShapePreserving(
      {0.0, T_pregrasp, T_contact}, traj_samples,
      /*zero_end_point_derivatives=*/true);

  // Derivative of the spline gives reference velocity for feedforward damping.
  // At t=0 and t=T_contact the derivative is zero (zero_end_point_derivatives).
  const auto traj_dot = traj.derivative(1);

  std::cout << "q_pregrasp: " << q_pregrasp.transpose() << "\n";
  std::cout << "q_contact:  " << q_contact.transpose() << "\n";

  auto& source_context = diagram->GetMutableSubsystemContext(
      *q_des_source, &simulator.get_mutable_context());
  auto& qdot_source_context = diagram->GetMutableSubsystemContext(
      *qdot_des_source, &simulator.get_mutable_context());

  double next_print = FLAGS_settle_time + 1.0;

  // Phase 2: PD tracks the spline, then holds q_contact once the trajectory ends.
  for (double t = FLAGS_settle_time + poll_step;
       t < FLAGS_simulation_time; t += poll_step) {
    simulator.AdvanceTo(t);

    const double local_t =
        std::clamp(t - FLAGS_settle_time, 0.0, traj.end_time());
    const Eigen::VectorXd q_ref = traj.value(local_t).col(0);
    const Eigen::VectorXd qdot_ref = traj_dot.value(local_t).col(0);
    q_des_source->get_mutable_source_value(&source_context)
        .SetFromVector(q_ref);
    qdot_des_source->get_mutable_source_value(&qdot_source_context)
        .SetFromVector(qdot_ref);

    RigidTransform<double> X_WC = CubePoseFromPositions(get_cube_pose());
    meshcat->SetTransform("/cube_frame", X_WC);
    update_grasp_spheres(X_WC);

    if (t >= next_print) {
      const Eigen::VectorXd q_actual =
          plant.GetPositions(plant_context, allegro_index);
      std::cout << "t=" << t << "\n"
                << "  q_ref:    " << q_ref.transpose() << "\n"
                << "  q_actual: " << q_actual.transpose() << "\n"
                << "  error:    " << (q_ref - q_actual).transpose() << "\n";
      next_print += 1.0;
    }

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
