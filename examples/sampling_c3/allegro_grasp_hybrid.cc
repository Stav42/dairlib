// Hybrid two-phase grasp controller (see grasp_trajopt_formulation.html §9).
//
//   Phase 1 (reach):   a synchronized cubic spline drives all three fingertips
//                      to the cube surface; tracked by joint-space PD + gravity
//                      compensation. The cube is held in place by an external
//                      support wrench so an early-arriving finger cannot shove
//                      it. We switch when all three fingers report contact.
//   Phase 2 (squeeze): force control tau = sum_i J_i^T f_i (+ gravity comp,
//                      + light joint damping). The squeeze magnitude alpha is a
//                      direct command (--alpha_m). After the grip is on, the
//                      support wrench is ramped to zero so ONLY friction holds
//                      the cube against gravity -- the friction-vs-mg test.
//
// The control logic lives in the simulation loop. Torque (Allegro actuation) and
// the cube support wrench (applied_spatial_force) are applied by fixing those
// input ports and mutating them each control step.

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <gflags/gflags.h>

#include <drake/common/trajectories/piecewise_polynomial.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/geometry/rgba.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/math/spatial_algebra.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/plant/externally_applied_spatial_force.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

#include "allegro_hand_utils.h"
#include "common/find_resource.h"
#include "cube_kinematics.h"
#include "meshcat_utils.h"

namespace dairlib {

using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::MeshcatVisualizerParams;
using drake::math::RigidTransform;
using drake::math::RotationMatrix;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::BodyIndex;
using drake::multibody::ContactResults;
using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::JacobianWrtVariable;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::SpatialForce;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;
using drake::trajectories::PiecewisePolynomial;
using Eigen::Vector3d;
using Eigen::VectorXd;

DEFINE_double(alpha_m, 1.0, "Index/middle squeeze force target (N). Thumb = 2x.");
DEFINE_double(kp, 10.0,
              "PD proportional gain (Nm/rad). Used for the Phase-1 reach AND "
              "the Phase-2 posture hold.");
DEFINE_double(kd, 1.5,
              "PD derivative gain (Nm*s/rad). Used for the Phase-1 reach AND "
              "the Phase-2 posture hold.");
DEFINE_double(tau_max, 10.0,
              "Per-joint torque saturation (Nm). Safety net against numerical "
              "blow-up. The visualizer uses no clamp; 10 Nm gives plenty of "
              "headroom for gravity compensation + PD.");
DEFINE_double(contact_force_thresh, 0.02,
              "Per-finger contact force to count as touching (N).");
DEFINE_double(contact_enable_t, 1.0,
              "Ignore contact before this time (s). Prevents the freeze from "
              "firing at t=0 when the pre-grasp pose is already near the cube.");
DEFINE_double(settle_time, 1.5,
              "Seconds to hold q_pregrasp with the cube parked far away. "
              "Eliminates the initial contact-force kick from sphere overlap.");
DEFINE_double(t_pregrasp, 2.5, "Spline time to reach the pre-grasp pose (s).");
DEFINE_double(t_contact, 5.0, "Spline time to reach the contact pose (s).");
DEFINE_double(squeeze_ramp, 0.5, "Seconds to ramp the squeeze from 0 to alpha.");
DEFINE_double(support_hold, 1.5,
              "Seconds to keep holding the cube after the grip starts.");
DEFINE_double(release_dur, 1.0,
              "Seconds over which the cube support is released.");
DEFINE_double(sim_time, std::numeric_limits<double>::infinity(),
              "Total sim time (s).");

namespace {
// Seed the thumb joints so IK does not start fully retracted.
void SetThumbSeed(const MultibodyPlant<double>& plant,
                  ModelInstanceIndex allegro_index,
                  drake::systems::Context<double>* ctx) {
  VectorXd q = VectorXd::Zero(plant.num_positions(allegro_index));
  q[12] = 1.0;
  q[13] = 0.5;
  q[14] = 0.5;
  q[15] = 0.3;
  plant.SetPositions(ctx, allegro_index, q);
}
}  // namespace

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // ---- Plant: Allegro hand + free cube. No platform; the cube is held by an
  // external support wrench during the reach, then released for the friction
  // test. ----
  DiagramBuilder<double> builder;

  // Plant is the physics engine and scene_graph is the geometry enginer. It handles the collisions. 
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.001);
  ModelInstanceIndex allegro_index = AddAllegroHandToPlant(&plant, &scene_graph);
  Parser parser(&plant, &scene_graph);
  ModelInstanceIndex cube_index = parser.AddModels(FindResourceOrThrow(
      "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];
  // SAP is more robust to stiff/impacting contact than the default TAMSI.
  plant.set_discrete_contact_approximation(
      drake::multibody::DiscreteContactApproximation::kSap);
  plant.Finalize();

  // Exclude thumb links from colliding with the palm.
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
            drake::geometry::GeometrySet(plant.GetCollisionGeometriesForBody(
                plant.GetBodyByName("palm_link", allegro_index)))));
  }

  // ---- Visualization ----
  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams vparams;
  vparams.publish_period = 1.0 / 60.0;
  MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat,
                                          std::move(vparams));
  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat);
  meshcat->SetCameraPose(Vector3d(0.7, 0.7, 0.8), Vector3d(0, 0, 0.55));
  AddFrameTriad(meshcat.get(), "/cube_frame", 0.005, 0.1);

  // Visual-only red markers at the three grasp points (no physics interaction).
  const drake::geometry::Rgba kRed(1.0, 0.0, 0.0, 1.0);
  meshcat->SetObject("/grasp/index", drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/middle", drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/thumb", drake::geometry::Sphere(0.006), kRed);

  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  auto& ctx = simulator.get_mutable_context();
  auto& plant_ctx = diagram->GetMutableSubsystemContext(plant, &ctx);

  // ---- Geometry / bodies for contact detection and Jacobians ----
  const std::array<std::string, 3> tip_names{"link_3_tip", "link_7_tip",
                                             "link_15_tip"};  // index, mid, thumb
  std::array<BodyIndex, 3> tip_bodies;
  for (int i = 0; i < 3; ++i)
    tip_bodies[i] = plant.GetBodyByName(tip_names[i], allegro_index).index();
  const BodyIndex cube_body = plant.GetBodyIndices(cube_index)[0];

  const double cube_size = 0.06;
  const double a = -0.02, b = 0.02, c = 0.0;  // face offsets
  const RigidTransform<double> X_WC0(RotationMatrix<double>(),
                                     Vector3d(0.0, 0.0, 0.58));
  VectorXd q_cube0(7);
  q_cube0 << 1, 0, 0, 0, X_WC0.translation();

  // ---- IK: pre-grasp and contact joint configurations ----
  auto solve_ik = [&](const char* label, const VectorXd& targets) {
    SetThumbSeed(plant, allegro_index, &plant_ctx);
    plant.SetPositions(&plant_ctx, cube_index, q_cube0);
    plant.SetPositions(&plant_ctx,
                       SolveGraspIK(plant, &plant_ctx, targets));
    const VectorXd q = plant.GetPositions(plant_ctx, allegro_index);
    double err = 0;
    for (int i = 0; i < 3; ++i) {
      const Vector3d p =
          plant.EvalBodyPoseInWorld(plant_ctx, plant.get_body(tip_bodies[i]))
              .translation();
      err += (p - targets.template segment<3>(3 * i)).norm();
    }
    std::cout << "IK " << label << " total FK error = " << err << " m\n";
    return q;
  };
  const int n_hand = plant.num_positions(allegro_index);

  // Contact pose: fingertip frames 5 mm inside each face.
  const VectorXd q_contact =
      solve_ik("contact", GetGraspPositions(X_WC0, cube_size - 0.01, a, b, c));

  // Pre-grasp pose: fingertips 1 cm outside each face — fingers are pointing
  // toward the cube but not touching. We seed from q_contact (which the solver
  // just found) so the pre-grasp IK starts from a nearby configuration and finds
  // the correct approach direction rather than a wrapped-around local minimum.
  plant.SetPositions(&plant_ctx, allegro_index, q_contact);
  const VectorXd q_pregrasp =
      solve_ik("pregrasp", GetGraspPositions(X_WC0, cube_size + 0.02, a, b, c));

  // ---- Reach spline: pre-grasp -> contact ----
  // Two waypoints only. The sim starts at q_pregrasp (fingers 1 cm outside the
  // cube faces) and drives to q_contact over t_contact seconds. The motion is
  // small in joint space so the interpolated Cartesian path is well-behaved.
  std::vector<Eigen::MatrixXd> samples{q_pregrasp, q_contact};
  const auto traj = PiecewisePolynomial<double>::CubicShapePreserving(
      {0.0, FLAGS_t_contact}, samples, true);
  const auto traj_dot = traj.derivative(1);

  // ---- Squeeze targets (force closure: thumb opposes index+middle) ----
  const double alpha_I = FLAGS_alpha_m, alpha_M = FLAGS_alpha_m;
  const double alpha_T = alpha_I + alpha_M;
  const std::array<double, 3> alpha{alpha_I, alpha_M, alpha_T};
  // Inward face normals in the cube frame: index/middle press +Y, thumb -Y.
  const std::array<Vector3d, 3> n_cube{Vector3d(0, 1, 0), Vector3d(0, 1, 0),
                                       Vector3d(0, -1, 0)};

  // Actuation port must be fixed before ForcedPublish.
  auto& act_fixed = plant.get_actuation_input_port(allegro_index)
                        .FixValue(&plant_ctx, VectorXd::Zero(n_hand));

  // Helper: publish current state to Meshcat with updated markers.
  auto preview = [&](const char* label) {
    const VectorXd gp = GetGraspPositions(X_WC0, cube_size, a, b, c);
    meshcat->SetTransform("/grasp/index",
                          RigidTransform<double>(Vector3d(gp.segment<3>(0))));
    meshcat->SetTransform("/grasp/middle",
                          RigidTransform<double>(Vector3d(gp.segment<3>(3))));
    meshcat->SetTransform("/grasp/thumb",
                          RigidTransform<double>(Vector3d(gp.segment<3>(6))));
    meshcat->SetTransform("/cube_frame", X_WC0);
    diagram->ForcedPublish(ctx);
    std::cout << "\n=== STATIC PREVIEW: " << label << " ===\n"
              << "Check Meshcat: fingers on the right cube faces?\n"
              << "Press Enter to continue...\n";
    std::cin.get();
  };

  // Show q_pregrasp: fingers should be 1 cm outside the cube faces.
  plant.SetPositions(&plant_ctx, allegro_index, q_pregrasp);
  plant.SetPositions(&plant_ctx, cube_index, q_cube0);
  plant.SetVelocities(&plant_ctx, allegro_index, VectorXd::Zero(n_hand));
  plant.SetVelocities(&plant_ctx, cube_index, VectorXd::Zero(6));
  preview("q_pregrasp (1 cm outside cube faces -- sim starts here)");

  // Show q_contact: fingers should be touching the cube faces.
  plant.SetPositions(&plant_ctx, allegro_index, q_contact);
  preview("q_contact (on cube faces -- sim ends here)");

  // ---- Initial state: start at q_pregrasp, not zero ----
  // The joint-space path from zero to q_contact is ill-conditioned -- the
  // fingers thrash through bad intermediate Cartesian poses. Starting at
  // q_pregrasp (fingers already aimed at the cube, 1 cm away) keeps the
  // entire reach motion small and well-behaved.
  plant.SetPositions(&plant_ctx, allegro_index, q_pregrasp);
  plant.SetVelocities(&plant_ctx, allegro_index, VectorXd::Zero(n_hand));
  plant.SetPositions(&plant_ctx, cube_index, q_cube0);
  plant.SetVelocities(&plant_ctx, cube_index, VectorXd::Zero(6));

  simulator.Initialize();

  // ---- Control loop ----
  enum Phase { kReach, kSqueeze };
  Phase phase = kReach;
  double squeeze_start = 0.0;
  bool cube_pinned = true;  // hold the cube perfectly fixed during reach + grip
  const double control_dt = 0.001;  // match the plant step (1 kHz control)
  double next_print = 0.25;  // throttle the fingertip-error print

  // Per-finger reach state: once a finger touches, freeze it (hold its joints)
  // instead of driving it deeper, while the others keep reaching.
  std::array<bool, 3> arrived{false, false, false};
  std::array<VectorXd, 3> q_freeze{VectorXd::Zero(4), VectorXd::Zero(4),
                                   VectorXd::Zero(4)};
  const std::array<int, 3> finger_start{0, 4, 12};  // index, middle, thumb

  std::cout << "alpha = [" << alpha_I << ", " << alpha_M << ", " << alpha_T
            << "] N\n";

  // Print gravity compensation torques at the initial pose so we can verify
  // none exceed tau_max (which would cause clamping and the fingers to fall).
  {
    const VectorXd tau_g0 = plant.GetVelocitiesFromArray(
        allegro_index,
        -plant.CalcGravityGeneralizedForces(plant_ctx));
    std::cout << "Gravity comp at q_pregrasp (Nm):\n  "
              << tau_g0.transpose() << "\n";
    std::cout << "Max |tau_g| = " << tau_g0.cwiseAbs().maxCoeff()
              << " Nm  (tau_max = " << FLAGS_tau_max << " Nm)\n";
  }

  for (double t = control_dt; t < FLAGS_sim_time; t += control_dt) {
    const VectorXd v_hand = plant.GetVelocities(plant_ctx, allegro_index);
    const RigidTransform<double> X_WC =
        CubePoseFromPositions(plant.GetPositions(plant_ctx, cube_index));

    // Gravity compensation for the hand (-tau_gravity), extracted to hand DOFs.
    const VectorXd tau_g_comp_full = -plant.CalcGravityGeneralizedForces(plant_ctx);
    const VectorXd tau_g_hand =
        plant.GetVelocitiesFromArray(allegro_index, tau_g_comp_full);

    // Per-finger contact detection from the plant's contact results.
    const auto& contacts =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            plant_ctx);
    std::array<bool, 3> touching{false, false, false};
    for (int k = 0; k < contacts.num_point_pair_contacts(); ++k) {
      const auto& info = contacts.point_pair_contact_info(k);
      const double fn = info.contact_force().norm();
      if (fn < FLAGS_contact_force_thresh) continue;
      for (int i = 0; i < 3; ++i) {
        const bool pair = (info.bodyA_index() == tip_bodies[i] &&
                           info.bodyB_index() == cube_body) ||
                          (info.bodyB_index() == tip_bodies[i] &&
                           info.bodyA_index() == cube_body);
        if (pair) touching[i] = true;
      }
    }

    VectorXd tau_hand(n_hand);
    if (phase == kReach) {
      const VectorXd q_hand = plant.GetPositions(plant_ctx, allegro_index);
      // Latch arrival and freeze the joints of any finger that just touched.
      for (int i = 0; i < 3; ++i) {
        if (!arrived[i] && touching[i] && t > FLAGS_contact_enable_t) {
          arrived[i] = true;
          q_freeze[i] = q_hand.segment(finger_start[i], 4);
          std::cout << "[t=" << t << "] finger " << i << " in contact\n";
        }
      }
      // Target = spline, except arrived fingers hold their frozen pose.
      const double tl = std::clamp(t, 0.0, traj.end_time());
      VectorXd q_tgt = traj.value(tl).col(0);
      VectorXd qd_tgt = traj_dot.value(tl).col(0);
      for (int i = 0; i < 3; ++i) {
        if (arrived[i]) {
          q_tgt.segment(finger_start[i], 4) = q_freeze[i];
          qd_tgt.segment(finger_start[i], 4).setZero();
        }
      }
      tau_hand = FLAGS_kp * (q_tgt - q_hand) + FLAGS_kd * (qd_tgt - v_hand) +
                 tau_g_hand;
      if (arrived[0] && arrived[1] && arrived[2]) {
        phase = kSqueeze;
        squeeze_start = t;
        std::cout << "[t=" << t << "] all fingers in contact -> SQUEEZE\n";
        // ---- Geometry diagnostic: verify squeeze directions are inward ----
        // For each finger, print its world-frame tip position, the cube face
        // centre it should be on, and the squeeze force direction.
        // Inward means the force points from the fingertip TOWARD the cube centre.
        const Vector3d cube_centre = X_WC.translation();
        std::cout << "  cube centre (world): " << cube_centre.transpose() << "\n";
        const char* fnames[3] = {"index", "middle", "thumb"};
        for (int i = 0; i < 3; ++i) {
          const Vector3d p_tip =
              plant.EvalBodyPoseInWorld(plant_ctx,
                                       plant.get_body(tip_bodies[i]))
                  .translation();
          const Vector3d force_dir = X_WC.rotation() * n_cube[i];
          const Vector3d to_centre = (cube_centre - p_tip).normalized();
          const double alignment = force_dir.dot(to_centre);  // +1 = inward
          std::cout << "  " << fnames[i]
                    << ": tip=" << p_tip.transpose()
                    << "  force_dir=" << force_dir.transpose()
                    << "  alignment=" << alignment
                    << (alignment > 0 ? " (INWARD OK)" : " (WRONG: outward!)")
                    << "\n";
        }
      }
    } else {  // kSqueeze
      const VectorXd q_hand = plant.GetPositions(plant_ctx, allegro_index);
      const double ramp =
          std::clamp((t - squeeze_start) / FLAGS_squeeze_ramp, 0.0, 1.0);

      // (1) Posture hold: keep each grasp finger at the pose it latched on
      // contact (q_freeze), so the fingertips stay pinned on the cube faces.
      // q_freeze was captured at first contact (near-zero penetration), so this
      // term mostly regulates position and adds little normal force on its own.
      VectorXd q_hold = q_hand;  // unlatched DOFs (ring) hold at current -> no error
      for (int i = 0; i < 3; ++i)
        q_hold.segment(finger_start[i], 4) = q_freeze[i];
      // Posture hold reuses the nominal reach gains (kp, kd).
      const VectorXd tau_hold =
          FLAGS_kp * (q_hold - q_hand) - FLAGS_kd * v_hand;

      // (2) Squeeze: command the grip force alpha directly via J^T f. This is
      // the dial swept against gravity once the cube is released. It rides on
      // top of the posture hold, so alpha sets the actual normal force.
      VectorXd tau_sq_full = VectorXd::Zero(plant.num_velocities());
      for (int i = 0; i < 3; ++i) {
        const Vector3d f_i = ramp * alpha[i] * (X_WC.rotation() * n_cube[i]);
        Eigen::MatrixXd Jv(3, plant.num_velocities());
        plant.CalcJacobianTranslationalVelocity(
            plant_ctx, JacobianWrtVariable::kV,
            plant.GetFrameByName(tip_names[i], allegro_index),
            Vector3d::Zero(), plant.world_frame(), plant.world_frame(), &Jv);
        tau_sq_full += Jv.transpose() * f_i;
      }
      tau_hand = tau_hold +
                 plant.GetVelocitiesFromArray(allegro_index, tau_sq_full) +
                 tau_g_hand;
    }
    // Saturate per joint: a hard safety net against the light-joint blow-up.
    // tau_hand = tau_hand.cwiseMax(-FLAGS_tau_max).cwiseMin(FLAGS_tau_max);
    act_fixed.GetMutableVectorData<double>()->SetFromVector(tau_hand);

    simulator.AdvanceTo(t);

    // Kinematic pin: hold the cube perfectly fixed during the reach and the
    // initial grip, then release it so friction ALONE must hold it against
    // gravity. (Teleport it back to its reference each step -- effectively a weld,
    // with none of the instability of a stiff explicit support spring.)
    if (phase == kSqueeze && (t - squeeze_start) > FLAGS_support_hold) {
      if (cube_pinned) std::cout << "[t=" << t << "] cube released\n";
      cube_pinned = false;
    }
    if (cube_pinned) {
      plant.SetPositions(&plant_ctx, cube_index, q_cube0);
      plant.SetVelocities(&plant_ctx, cube_index, VectorXd::Zero(6));
    }
    const RigidTransform<double> X_WC_now =
        CubePoseFromPositions(plant.GetPositions(plant_ctx, cube_index));
    meshcat->SetTransform("/cube_frame", X_WC_now);

    // Move the red grasp markers to the (current) grasp points on the cube faces.
    const VectorXd gp = GetGraspPositions(X_WC_now, cube_size, a, b, c);
    meshcat->SetTransform("/grasp/index",
                          RigidTransform<double>(Vector3d(gp.segment<3>(0))));
    meshcat->SetTransform("/grasp/middle",
                          RigidTransform<double>(Vector3d(gp.segment<3>(3))));
    meshcat->SetTransform("/grasp/thumb",
                          RigidTransform<double>(Vector3d(gp.segment<3>(6))));

    // Print the fingertip position error (distance from each tip to its grasp
    // point), throttled to a few times per second.
    if (t >= next_print) {
      next_print += 0.25;
      std::cout << "[t=" << t << "] tip pos err [index, middle, thumb] = ";
      for (int i = 0; i < 3; ++i) {
        const Vector3d p_tip =
            plant.EvalBodyPoseInWorld(plant_ctx, plant.get_body(tip_bodies[i]))
                .translation();
        std::cout << (p_tip - Vector3d(gp.segment<3>(3 * i))).norm() << " ";
      }
      std::cout << "m\n";
      std::cout << "[t=" << t << "] tau_hand (Nm) = " << tau_hand.transpose()
                << "\n";
    }
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
