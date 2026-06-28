// Two-phase grasp controller — §12 of grasp_trajopt_formulation.html.
//
//   Phase 1 (kReach): PD + cubic spline drives the three fingertips to the
//                     cube faces. The cube is kinematically pinned. Logic is
//                     identical to allegro_grasp_hybrid.cc.
//
//   Handoff: when all three fingers report contact, the LCS is linearized at
//            the contact configuration, and the cube pin is held briefly
//            (0.5 s) before release. The grasping fingers are PD-held at
//            q_contact (5 mm inside the faces) so they keep a real preload.
//
//   Phase 2 (kC3): C3/ADMM computes joint torques every control step. The Q
//                  cost anchors the cube at its world-frame reference; gravity
//                  in the LCS drift forces C3 to find enough contact force to
//                  hold the cube up. The squeeze force is not commanded — it
//                  emerges from "cube must not fall."
//
// Two plants are required: a discrete sim plant (SAP, connected to the
// Drake simulator and Meshcat) and a continuous lcs plant (used only for
// LCS linearization via LCSFactory). Their state vectors have the same layout
// so state read from the sim plant can be passed directly to C3::Solve.

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
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/solvers/osqp_solver.h>
#include <drake/solvers/solver_options.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/system.h>

#include "c3/core/c3.h"
#include "c3/core/c3_options.h"
#include "c3/core/c3_plus.h"
#include "c3/core/lcs.h"
#include "c3/multibody/lcs_factory.h"
#include "c3/multibody/lcs_factory_options.h"

#include "allegro_hand_utils.h"
#include "common/find_resource.h"
#include "cube_kinematics.h"
#include "meshcat_utils.h"

namespace dairlib {

using drake::AutoDiffXd;
using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::MeshcatVisualizerParams;
using drake::math::RigidTransform;
using drake::math::RotationMatrix;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::BodyIndex;
using drake::multibody::ContactResults;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::solvers::OsqpSolver;
using drake::solvers::SolverOptions;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;
using drake::systems::System;
using drake::trajectories::PiecewisePolynomial;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

using c3::C3;
using c3::C3Options;
using c3::C3Plus;
using c3::LCS;
using c3::LCSFactoryOptions;
using c3::multibody::GetContactModelMap;
using c3::multibody::LCSFactory;

// ── Reach-phase flags ────────────────────────────────────────────────────────
DEFINE_double(kp, 10.0, "PD proportional gain (Nm/rad) for the reach phase.");
DEFINE_double(kd, 1.5,
              "PD derivative gain (Nm·s/rad) for the reach phase.");
DEFINE_double(tau_max, 100.0,
              "Per-joint torque box bound (Nm). Applied as INPUT constraints "
              "inside the C3 QP to keep it bounded (prevents DualInfeasible).");
DEFINE_double(contact_force_thresh, 0.02,
              "Per-finger contact force threshold to count as touching (N).");
DEFINE_double(contact_enable_t, 1.0,
              "Ignore contact before this time (s). Guards against false "
              "triggers when the pre-grasp pose is near the cube at t=0.");
DEFINE_double(t_contact, 5.0,
              "Spline end time: seconds to reach the contact pose.");

// ── C3-squeeze-phase flags ───────────────────────────────────────────────────
DEFINE_double(k_hold, 10.0,
              "Q cost weight on hand-joint deviations from q_contact. Keeps "
              "fingertips on the cube faces inside the ADMM solve.");
DEFINE_double(w_cube, 1000.0,
              "Q cost weight on cube pose and velocity (world-frame anchor). "
              "This is the load-bearing term: C3 must keep the cube here "
              "against gravity, which forces Σμλ_n ≥ mg.");
DEFINE_double(w_vel, 0.1,
              "Q cost weight on hand joint velocities (light damping).");
DEFINE_double(w_R, 0.01, "R cost weight on joint torques.");
DEFINE_double(w_G, 1.0, "ADMM augmented-Lagrangian G weight.");
DEFINE_double(w_U, 1.0, "ADMM augmented-Lagrangian U weight.");
DEFINE_double(w_lambda, 100.0,
              "Force-reference cost on the three normal contact forces "
              "λ_n[3,4,5]. Regularises the squeeze nullspace (§11.5) and gives "
              "the solve a physical grip target. Set 0 to disable.");
DEFINE_double(alpha_m, 1.0,
              "Per-finger normal grip-force target (N) for the λ_n reference. "
              "Thumb target = 2·alpha_m (force closure vs index+middle).");
DEFINE_int32(N, 5, "C3 prediction horizon (knot points).");
DEFINE_double(c3_dt, 0.04, "LCS timestep (s) for C3 linearization.");
DEFINE_double(mu, 0.5, "Friction coefficient used in the LCS.");
DEFINE_int32(num_friction_directions, 2,
             "Friction directions per contact (Stewart-Trinkle).");
DEFINE_string(contact_model, "stewart_and_trinkle",
              "LCS contact model: 'stewart_and_trinkle' (explicit λ_n) or "
              "'anitescu' (convex; a contact's normal force is the sum of its "
              "cone forces). n_lambda, n_z, the force reference, and the λ_n "
              "diagnostic all adapt automatically to this choice.");
DEFINE_double(input_scale, 0.0,
              "Torque-input rescaling s_u (0 = auto = |A|/|B|). Shrinks B "
              "to condition the QP without changing contact forces λ.");
DEFINE_bool(relinearize, true,
            "Re-linearize the LCS at the current state each control step. "
            "More accurate for a drifting contact configuration; heavier.");
DEFINE_double(osqp_eps, 1e-3,
              "OSQP convergence tolerance (eps_abs = eps_rel).");

// ── General ──────────────────────────────────────────────────────────────────
DEFINE_double(sim_time, std::numeric_limits<double>::infinity(),
              "Total simulation time (s).");

namespace {
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

// Per-contact λ-index groups whose SUM is that contact's normal force. This is
// the one piece of model-specific bookkeeping; the force reference and the λ_n
// diagnostic are both built from it, so switching contact models needs nothing
// else.
//   stewart_and_trinkle: λ = [γ(nc) | λ_n(nc) | β(...)]  → group i = { nc + i }
//   anitescu:            λ = per-contact cone blocks      → group i = block i
//        (normal force = Σ of the contact's cone forces, because the Anitescu
//         basis J_c = Eᵀ Jₙ + μ Jₜ puts Jₙ in every row of the contact's block.)
std::vector<std::vector<int>> NormalForceGroups(const std::string& model,
                                                int n_contacts,
                                                int n_friction_directions) {
  std::vector<std::vector<int>> groups(n_contacts);
  if (model == "anitescu") {
    const int block = 2 * n_friction_directions;  // λ entries per contact
    for (int i = 0; i < n_contacts; ++i)
      for (int j = 0; j < block; ++j) groups[i].push_back(i * block + j);
  } else {  // stewart_and_trinkle
    for (int i = 0; i < n_contacts; ++i) groups[i].push_back(n_contacts + i);
  }
  return groups;
}
}  // namespace

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_contact_model != "stewart_and_trinkle" &&
      FLAGS_contact_model != "anitescu") {
    throw std::runtime_error(
        "contact_model must be 'stewart_and_trinkle' or 'anitescu'.");
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 1. Sim plant — discrete (1 ms), SAP, connected to the Drake simulator
  //    and Meshcat. This is the plant that runs the physics.
  // ══════════════════════════════════════════════════════════════════════════
  DiagramBuilder<double> sim_builder;
  auto [sim_plant, sim_scene_graph] =
      AddMultibodyPlantSceneGraph(&sim_builder, 0.001);

  ModelInstanceIndex sim_allegro =
      AddAllegroHandToPlant(&sim_plant, &sim_scene_graph);
  ModelInstanceIndex sim_cube;
  {
    Parser parser(&sim_plant, &sim_scene_graph);
    sim_cube = parser.AddModels(FindResourceOrThrow(
        "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];
  }

  sim_plant.set_discrete_contact_approximation(
      drake::multibody::DiscreteContactApproximation::kSap);
  sim_plant.Finalize();

  // Filter thumb links vs palm to prevent SAP impulse blow-up at the knuckle.
  {
    std::vector<GeometryId> thumb_geoms;
    for (const char* name :
         {"link_12", "link_13", "link_14", "link_15", "link_15_tip"}) {
      const auto& g = sim_plant.GetCollisionGeometriesForBody(
          sim_plant.GetBodyByName(name, sim_allegro));
      thumb_geoms.insert(thumb_geoms.end(), g.begin(), g.end());
    }
    sim_scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet(thumb_geoms),
            drake::geometry::GeometrySet(
                sim_plant.GetCollisionGeometriesForBody(
                    sim_plant.GetBodyByName("palm_link", sim_allegro)))));
  }

  // Visualization.
  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizerParams vparams;
  vparams.publish_period = 1.0 / 60.0;
  MeshcatVisualizer<double>::AddToBuilder(&sim_builder, sim_scene_graph,
                                          meshcat, std::move(vparams));
  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &sim_builder, sim_plant, meshcat);
  meshcat->SetCameraPose(Vector3d(0.7, 0.7, 0.8), Vector3d(0, 0, 0.55));
  AddFrameTriad(meshcat.get(), "/cube_frame", 0.005, 0.1);

  const drake::geometry::Rgba kRed(1.0, 0.0, 0.0, 1.0);
  meshcat->SetObject("/grasp/index",  drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/middle", drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/thumb",  drake::geometry::Sphere(0.006), kRed);

  auto sim_diagram = sim_builder.Build();
  Simulator<double> simulator(*sim_diagram);
  simulator.set_target_realtime_rate(1.0);
  auto& sim_ctx = simulator.get_mutable_context();
  auto& plant_ctx =
      sim_diagram->GetMutableSubsystemContext(sim_plant, &sim_ctx);

  // ══════════════════════════════════════════════════════════════════════════
  // 2. LCS plant — continuous (time_step=0), no visualizer.
  //    The factory needs a continuous plant so it can differentiate through
  //    the dynamics. Same bodies in the same order → same state layout as the
  //    sim plant, so state from the sim can be passed directly to C3::Solve.
  // ══════════════════════════════════════════════════════════════════════════
  DiagramBuilder<double> lcs_builder;
  auto [lcs_plant, lcs_scene_graph] =
      AddMultibodyPlantSceneGraph(&lcs_builder, 0.0);

  ModelInstanceIndex lcs_allegro =
      AddAllegroHandToPlant(&lcs_plant, &lcs_scene_graph);
  ModelInstanceIndex lcs_cube;
  {
    Parser parser(&lcs_plant, &lcs_scene_graph);
    lcs_cube = parser.AddModels(FindResourceOrThrow(
        "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];
  }
  lcs_plant.Finalize();

  std::unique_ptr<MultibodyPlant<AutoDiffXd>> lcs_plant_ad =
      System<double>::ToAutoDiffXd(lcs_plant);

  auto lcs_diagram = lcs_builder.Build();
  auto lcs_diagram_ctx = lcs_diagram->CreateDefaultContext();
  auto& lcs_ctx =
      lcs_diagram->GetMutableSubsystemContext(lcs_plant, lcs_diagram_ctx.get());
  auto lcs_ctx_ad = lcs_plant_ad->CreateDefaultContext();

  // ══════════════════════════════════════════════════════════════════════════
  // 3. Contact geometry pairs for the LCS factory.
  //    Ordering: [index, middle, thumb] → λ_n at indices [3, 4, 5] for
  //    Stewart-Trinkle with 2 friction directions per contact.
  // ══════════════════════════════════════════════════════════════════════════
  const GeometryId lcs_cube_geom =
      lcs_plant.GetCollisionGeometriesForBody(
          lcs_plant.get_body(lcs_plant.GetBodyIndices(lcs_cube)[0]))[0];
  const GeometryId lcs_index_geom =
      lcs_plant.GetCollisionGeometriesForBody(
          lcs_plant.GetBodyByName("link_3_tip",  lcs_allegro))[0];
  const GeometryId lcs_middle_geom =
      lcs_plant.GetCollisionGeometriesForBody(
          lcs_plant.GetBodyByName("link_7_tip",  lcs_allegro))[0];
  const GeometryId lcs_thumb_geom =
      lcs_plant.GetCollisionGeometriesForBody(
          lcs_plant.GetBodyByName("link_15_tip", lcs_allegro))[0];

  const std::vector<SortedPair<GeometryId>> contact_pairs{
      SortedPair<GeometryId>(lcs_index_geom,  lcs_cube_geom),
      SortedPair<GeometryId>(lcs_middle_geom, lcs_cube_geom),
      SortedPair<GeometryId>(lcs_thumb_geom,  lcs_cube_geom)};

  // ══════════════════════════════════════════════════════════════════════════
  // 4. IK: q_contact (fingertips 5 mm inside faces) and q_pregrasp (1 cm
  //    outside). Both solved on the sim plant context. The sim starts at
  //    q_pregrasp so the entire reach is a small, well-conditioned motion.
  // ══════════════════════════════════════════════════════════════════════════
  const double cube_size = 0.06;
  const double a = -0.02, b = 0.02, c_off = 0.0;
  const RigidTransform<double> X_WC0(RotationMatrix<double>(),
                                     Vector3d(0.0, 0.0, 0.58));
  VectorXd q_cube0(7);
  q_cube0 << 1, 0, 0, 0, X_WC0.translation();

  const std::array<std::string, 3> tip_names{"link_3_tip", "link_7_tip",
                                             "link_15_tip"};
  std::array<BodyIndex, 3> tip_bodies;
  for (int i = 0; i < 3; ++i)
    tip_bodies[i] =
        sim_plant.GetBodyByName(tip_names[i], sim_allegro).index();
  const BodyIndex cube_body = sim_plant.GetBodyIndices(sim_cube)[0];

  const int n_hand = sim_plant.num_positions(sim_allegro);

  auto solve_ik = [&](const char* label, const VectorXd& targets) {
    SetThumbSeed(sim_plant, sim_allegro, &plant_ctx);
    sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
    sim_plant.SetPositions(
        &plant_ctx, SolveGraspIK(sim_plant, &plant_ctx, targets));
    const VectorXd q = sim_plant.GetPositions(plant_ctx, sim_allegro);
    double err = 0;
    for (int i = 0; i < 3; ++i) {
      const Vector3d p =
          sim_plant
              .EvalBodyPoseInWorld(plant_ctx,
                                   sim_plant.get_body(tip_bodies[i]))
              .translation();
      err += (p - targets.template segment<3>(3 * i)).norm();
    }
    std::cout << "IK " << label << " total FK error = " << err << " m\n";
    return q;
  };

  const VectorXd q_contact = solve_ik(
      "contact", GetGraspPositions(X_WC0, cube_size - 0.01, a, b, c_off));

  // Seed the pregrasp IK from q_contact so the solver finds the same
  // approach direction rather than a wrapped-around local minimum.
  sim_plant.SetPositions(&plant_ctx, sim_allegro, q_contact);
  const VectorXd q_pregrasp = solve_ik(
      "pregrasp", GetGraspPositions(X_WC0, cube_size + 0.02, a, b, c_off));

  // ══════════════════════════════════════════════════════════════════════════
  // 5. Reach spline: q_pregrasp → q_contact over t_contact seconds.
  // ══════════════════════════════════════════════════════════════════════════
  std::vector<Eigen::MatrixXd> spline_pts{q_pregrasp, q_contact};
  const auto traj = PiecewisePolynomial<double>::CubicShapePreserving(
      {0.0, FLAGS_t_contact}, spline_pts, true);
  const auto traj_dot = traj.derivative(1);

  // ══════════════════════════════════════════════════════════════════════════
  // 6. Actuation port + static Meshcat preview.
  // ══════════════════════════════════════════════════════════════════════════
  auto& act_fixed = sim_plant.get_actuation_input_port(sim_allegro)
                        .FixValue(&plant_ctx, VectorXd::Zero(n_hand));

  auto update_markers = [&](const RigidTransform<double>& X_WC) {
    const VectorXd gp = GetGraspPositions(X_WC, cube_size, a, b, c_off);
    meshcat->SetTransform("/grasp/index",
                          RigidTransform<double>(Vector3d(gp.segment<3>(0))));
    meshcat->SetTransform("/grasp/middle",
                          RigidTransform<double>(Vector3d(gp.segment<3>(3))));
    meshcat->SetTransform("/grasp/thumb",
                          RigidTransform<double>(Vector3d(gp.segment<3>(6))));
    meshcat->SetTransform("/cube_frame", X_WC);
  };

  auto preview = [&](const char* label) {
    update_markers(X_WC0);
    sim_diagram->ForcedPublish(sim_ctx);
    std::cout << "\n=== STATIC PREVIEW: " << label << " ===\n"
              << "Press Enter to continue...\n";
    std::cin.get();
  };

  sim_plant.SetPositions(&plant_ctx, sim_allegro, q_pregrasp);
  sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
  sim_plant.SetVelocities(&plant_ctx, sim_allegro, VectorXd::Zero(n_hand));
  sim_plant.SetVelocities(&plant_ctx, sim_cube, VectorXd::Zero(6));
  preview("q_pregrasp  (1 cm outside cube faces — sim starts here)");

  sim_plant.SetPositions(&plant_ctx, sim_allegro, q_contact);
  preview("q_contact  (on cube faces — C3 target configuration)");

  // ══════════════════════════════════════════════════════════════════════════
  // 7. LCS factory options and C3 dimension bookkeeping.
  //    These are fixed for the whole run; the LCS itself is built at handoff.
  // ══════════════════════════════════════════════════════════════════════════
  LCSFactoryOptions lcs_opts;
  lcs_opts.contact_model = FLAGS_contact_model;
  lcs_opts.N = FLAGS_N;
  lcs_opts.dt = FLAGS_c3_dt;
  lcs_opts.num_contacts = 3;
  lcs_opts.num_friction_directions = FLAGS_num_friction_directions;
  lcs_opts.mu = FLAGS_mu;

  const int n_x = lcs_plant.num_positions() + lcs_plant.num_velocities();
  const int n_u = lcs_plant.num_actuators();
  const int n_lambda = LCSFactory::GetNumContactVariables(
      GetContactModelMap().at(lcs_opts.contact_model), 3,
      FLAGS_num_friction_directions);
  // C3+ augments z with an explicit η variable (n_lambda extra), so its
  // z-size is n_x + n_u + 2·n_lambda (plain C3/C3QP would be n_x + n_u + n_lambda).
  const int n_z = n_x + n_u + 2 * n_lambda;

  // Model-specific bookkeeping, derived once from the contact-model flag and
  // reused by the force reference and the λ_n diagnostic below.
  const int n_contacts = lcs_opts.num_contacts.value();
  const std::vector<std::vector<int>> normal_groups = NormalForceGroups(
      FLAGS_contact_model, n_contacts, FLAGS_num_friction_directions);

  std::cout << "C3 dimensions: n_x=" << n_x << "  n_u=" << n_u
            << "  n_lambda=" << n_lambda << "  n_z=" << n_z << "\n";

  // State layout (same for both plants — same DOF count and ordering):
  //   positions [0 .. n_hand_q-1]       : hand joints (16)
  //   positions [n_hand_q .. n_pos-1]   : cube quaternion + xyz (7)
  //   velocities [n_pos .. n_pos+n_hand_v-1]     : hand vel (16)
  //   velocities [n_pos+n_hand_v .. n_x-1]       : cube vel (6)
  const int n_hand_q = lcs_plant.num_positions(lcs_allegro);   // 16
  const int n_hand_v = lcs_plant.num_velocities(lcs_allegro);  // 16
  const int n_pos    = lcs_plant.num_positions();               // 23

  // Q cost matrix (fixed for the whole C3 phase):
  //   k_hold   on hand joint positions  → keeps fingers on cube faces
  //   w_cube   on cube q and v          → world-frame cube anchor (key term)
  //   w_vel    on hand joint velocities → light damping
  MatrixXd Q_knot = MatrixXd::Zero(n_x, n_x);
  for (int i = 0; i < n_hand_q; ++i)          Q_knot(i, i) = FLAGS_k_hold;
  for (int i = n_hand_q; i < n_pos; ++i)      Q_knot(i, i) = FLAGS_w_cube;
  for (int i = n_pos; i < n_pos + n_hand_v; ++i) Q_knot(i, i) = FLAGS_w_vel;
  for (int i = n_pos + n_hand_v; i < n_x; ++i)   Q_knot(i, i) = FLAGS_w_cube;

  // ══════════════════════════════════════════════════════════════════════════
  // 8. Initial sim state and simulator initialization.
  // ══════════════════════════════════════════════════════════════════════════
  sim_plant.SetPositions(&plant_ctx, sim_allegro, q_pregrasp);
  sim_plant.SetVelocities(&plant_ctx, sim_allegro, VectorXd::Zero(n_hand));
  sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
  sim_plant.SetVelocities(&plant_ctx, sim_cube, VectorXd::Zero(6));
  simulator.Initialize();

  // ══════════════════════════════════════════════════════════════════════════
  // 9. Control loop.
  // ══════════════════════════════════════════════════════════════════════════
  enum Phase { kReach, kC3 };
  Phase phase = kReach;
  bool cube_pinned = true;
  const double control_dt = 0.001;
  double next_print = 0.25;

  std::array<bool, 3>    arrived{false, false, false};
  const std::array<int, 3> finger_start{0, 4, 12};  // index, middle, thumb

  // C3 objects allocated at handoff.
  std::unique_ptr<C3> c3;
  double s_u = 1.0;
  double handoff_t = -1.0;  // set at handoff, cube released after 0.5 s

  for (double t = control_dt; t < FLAGS_sim_time; t += control_dt) {
    const VectorXd v_hand = sim_plant.GetVelocities(plant_ctx, sim_allegro);

    // Contact detection from the sim plant's contact results.
    const auto& contacts =
        sim_plant.get_contact_results_output_port()
            .Eval<ContactResults<double>>(plant_ctx);
    std::array<bool, 3> touching{false, false, false};
    for (int k = 0; k < contacts.num_point_pair_contacts(); ++k) {
      const auto& info = contacts.point_pair_contact_info(k);
      if (info.contact_force().norm() < FLAGS_contact_force_thresh) continue;
      for (int i = 0; i < 3; ++i) {
        if ((info.bodyA_index() == tip_bodies[i] &&
             info.bodyB_index() == cube_body) ||
            (info.bodyB_index() == tip_bodies[i] &&
             info.bodyA_index() == cube_body))
          touching[i] = true;
      }
    }

    VectorXd tau_hand(n_hand);

    // ── Phase 1: reach ──────────────────────────────────────────────────────
    if (phase == kReach) {
      const VectorXd q_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);

      // Latch any finger that just made contact (after the guard time).
      for (int i = 0; i < 3; ++i) {
        if (!arrived[i] && touching[i] && t > FLAGS_contact_enable_t) {
          arrived[i] = true;
          std::cout << "[t=" << t << "] finger " << i << " arrived\n";
        }
      }

      const double tl = std::clamp(t, 0.0, traj.end_time());
      VectorXd q_tgt   = traj.value(tl).col(0);
      VectorXd qd_tgt  = traj_dot.value(tl).col(0);
      for (int i = 0; i < 3; ++i) {
        if (arrived[i]) {
          // Hold the grasping finger at q_contact (5 mm inside the face) so the
          // position error keeps a real inward preload, not just a light touch.
          q_tgt.segment(finger_start[i], 4) =
              q_contact.segment(finger_start[i], 4);
          qd_tgt.segment(finger_start[i], 4).setZero();
        }
      }

      const VectorXd tau_g_hand = sim_plant.GetVelocitiesFromArray(
          sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));

      tau_hand = FLAGS_kp * (q_tgt - q_hand) +
                 FLAGS_kd * (qd_tgt - v_hand) + tau_g_hand;

      // ── Handoff: all three fingers in contact ───────────────────────────
      if (arrived[0] && arrived[1] && arrived[2]) {
        std::cout << "[t=" << t
                  << "] all fingers in contact → C3 handoff\n";

        // Read full state from the sim plant at this instant.
        const VectorXd x_contact =
            sim_plant.GetPositionsAndVelocities(plant_ctx);

        // Sync the lcs plant context and build the LCS linearized here.
        lcs_plant.SetPositionsAndVelocities(&lcs_ctx, x_contact);

        LCS lcs_init = LCSFactory::LinearizePlantToLCS(
            lcs_plant, lcs_ctx, *lcs_plant_ad, *lcs_ctx_ad,
            contact_pairs, lcs_opts, x_contact, VectorXd::Zero(n_u));

        // Contact-gap diagnostics: each entry should be ≤ 0 (touching).
        const VectorXd eta0 =
            lcs_init.E()[0] * x_contact + lcs_init.c()[0];
        if (FLAGS_contact_model == "stewart_and_trinkle") {
          // Normal gaps φ live in the λ_n block [n_contacts, 2·n_contacts).
          std::cout << "  contact gaps φ [index, middle, thumb] = "
                    << eta0.segment(n_contacts, n_contacts).transpose()
                    << "  (<=0 means active)\n";
        } else {
          std::cout << "  (Anitescu: η is the cone-velocity constraint, not a "
                       "per-finger signed distance; gaps not shown)\n";
        }
        std::cout << "  LCS norms: |A|=" << lcs_init.A()[0].norm()
                  << "  |B|=" << lcs_init.B()[0].norm()
                  << "  |D|=" << lcs_init.D()[0].norm() << "\n";

        // Input scaling: s_u shrinks B so |B_scaled| ≈ |A|. This is a
        // change of variables on u only; λ is unchanged. R is scaled by
        // s_u² so the penalty on the real torque is preserved.
        s_u = FLAGS_input_scale > 0.0
                  ? FLAGS_input_scale
                  : lcs_init.A()[0].norm() / lcs_init.B()[0].norm();
        {
          std::vector<MatrixXd> B_sc = lcs_init.B();
          for (auto& Bk : B_sc) Bk *= s_u;
          lcs_init.set_B(B_sc);
        }
        std::cout << "  s_u=" << s_u
                  << "  |B| now=" << lcs_init.B()[0].norm() << "\n";

        // Desired state: q_contact for the three grasping fingers, current
        // ring-finger q, cube at the world-frame reference q_cube0.
        VectorXd q_des_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);
        for (int i = 0; i < 3; ++i)
          q_des_hand.segment(finger_start[i], 4) =
              q_contact.segment(finger_start[i], 4);

        VectorXd x_des = VectorXd::Zero(n_x);
        x_des.head(n_hand_q) = q_des_hand;
        x_des.segment(n_hand_q, 7) = q_cube0;
        // Hand and cube velocities desired = 0 (zero-initialized above).

        const std::vector<VectorXd> x_desired(FLAGS_N + 1, x_des);

        // Cost matrices.
        const std::vector<MatrixXd> Q_vec(FLAGS_N + 1, Q_knot);
        const std::vector<MatrixXd> R_vec(
            FLAGS_N,
            (s_u * s_u) * FLAGS_w_R * MatrixXd::Identity(n_u, n_u));
        const std::vector<MatrixXd> G_vec(
            FLAGS_N, FLAGS_w_G * MatrixXd::Identity(n_z, n_z));
        const std::vector<MatrixXd> U_vec(
            FLAGS_N, FLAGS_w_U * MatrixXd::Identity(n_z, n_z));

        // ADMM options (tuned to match the working allegro_grasp_c3.cc).
        C3Options c3_opts;
        c3_opts.admm_iter  = 3;
        c3_opts.rho_scale  = 3;
        c3_opts.warm_start = false;
        c3_opts.scale_lcs  = true;
        c3_opts.gamma      = 1.0;

        c3 = std::make_unique<C3Plus>(
            lcs_init, C3::CostMatrices(Q_vec, R_vec, G_vec, U_vec),
            x_desired, c3_opts);

        // Force reference on each contact's NORMAL force — seeds a real grip
        // and regularises the squeeze nullspace. A contact's normal force is
        // the sum over normal_groups[i] (a single λ_n entry for Stewart-
        // Trinkle, the whole cone block for Anitescu), so the per-contact cost
        // w_lambda·(Σ_{j∈g} λ_j − αᵢ)² is built from the rank-1 block 1·1ᵀ over
        // the group with λ_des spreading αᵢ evenly across it. For Stewart-
        // Trinkle (singleton group) this collapses to the old diagonal cost on
        // λ_n. Targets: index=middle=alpha_m, thumb=2·alpha_m (force closure).
        if (FLAGS_w_lambda > 0.0) {
          const std::array<double, 3> alpha{FLAGS_alpha_m, FLAGS_alpha_m,
                                            2.0 * FLAGS_alpha_m};
          std::vector<MatrixXd> W_lam(FLAGS_N,
                                      MatrixXd::Zero(n_lambda, n_lambda));
          std::vector<VectorXd> lambda_des(FLAGS_N,
                                           VectorXd::Zero(n_lambda));
          for (int k = 0; k < FLAGS_N; ++k) {
            for (int i = 0; i < 3; ++i) {
              const std::vector<int>& g = normal_groups[i];
              for (int a : g)
                for (int b : g) W_lam[k](a, b) = FLAGS_w_lambda;
              for (int a : g)
                lambda_des[k](a) = alpha[i] / static_cast<double>(g.size());
            }
          }
          c3->SetForceTrackingWeight(W_lam);
          c3->UpdateForceTarget(lambda_des);
        }

        // Box constraints on the torque, one INPUT bound per hand joint.
        // C3 optimises a scaled input ũ with τ = s_u·ũ (B was scaled by s_u),
        // so a physical bound |τ_i| ≤ tau_max becomes |ũ_i| ≤ tau_max / s_u.
        // Boxing the input makes the QP's feasible set bounded, which removes
        // the DualInfeasible (unbounded-primal) failure at pin release.
        {
          const double u_bound = FLAGS_tau_max / s_u;
          for (int i = 0; i < n_u; ++i) {
            Eigen::RowVectorXd Ai = Eigen::RowVectorXd::Zero(n_u);
            Ai(i) = 1.0;
            c3->AddLinearConstraint(Ai, -u_bound, u_bound,
                                    c3::ConstraintVariable::INPUT);
          }
        }

        // OSQP options (same tuning as allegro_grasp_c3.cc).
        SolverOptions osqp_opts;
        const auto oid = OsqpSolver::id();
        osqp_opts.SetOption(oid, "max_iter",            4000);
        osqp_opts.SetOption(oid, "verbose",             0);
        osqp_opts.SetOption(oid, "warm_starting",       1);
        osqp_opts.SetOption(oid, "polishing",           1);
        osqp_opts.SetOption(oid, "polish_refine_iter",  3);
        osqp_opts.SetOption(oid, "scaled_termination",  1);
        osqp_opts.SetOption(oid, "check_termination",   25);
        osqp_opts.SetOption(oid, "scaling",             15);
        osqp_opts.SetOption(oid, "adaptive_rho",        1);
        osqp_opts.SetOption(oid, "rho",                 1e-4);
        osqp_opts.SetOption(oid, "sigma",               1e-6);
        osqp_opts.SetOption(oid, "alpha",               1.6);
        osqp_opts.SetOption(oid, "eps_abs",             FLAGS_osqp_eps);
        osqp_opts.SetOption(oid, "eps_rel",             FLAGS_osqp_eps);
        c3->SetSolverOptions(osqp_opts);

        phase = kC3;
        handoff_t = t;  // cube released after 0.5 s warm-up (see pin block)
      }

    } else {
      // ── Phase 2: C3 ─────────────────────────────────────────────────────
      const VectorXd x_current =
          sim_plant.GetPositionsAndVelocities(plant_ctx);

      if (FLAGS_relinearize) {
        lcs_plant.SetPositionsAndVelocities(&lcs_ctx, x_current);
        LCS lcs_new = LCSFactory::LinearizePlantToLCS(
            lcs_plant, lcs_ctx, *lcs_plant_ad, *lcs_ctx_ad,
            contact_pairs, lcs_opts, x_current, VectorXd::Zero(n_u));
        std::vector<MatrixXd> B_sc = lcs_new.B();
        for (auto& Bk : B_sc) Bk *= s_u;
        lcs_new.set_B(B_sc);
        c3->UpdateLCS(lcs_new);
      }

      c3->Solve(x_current);

      if (cube_pinned) {
        // Warm-up period: cube is still pinned, so C3's cost gradient on the
        // cube is near zero and it finds near-zero torques as "optimal". This
        // causes fingers to drift away from the cube. Instead, use PD to hold
        // fingers at q_contact (5 mm inside the faces) while C3 runs in the
        // background, so they keep a real inward preload on the cube.
        const VectorXd q_hand =
            sim_plant.GetPositions(plant_ctx, sim_allegro);
        const VectorXd tau_g = sim_plant.GetVelocitiesFromArray(
            sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));
        VectorXd q_tgt_c3 = q_hand;
        for (int i = 0; i < 3; ++i)
          q_tgt_c3.segment(finger_start[i], 4) =
              q_contact.segment(finger_start[i], 4);
        tau_hand = FLAGS_kp * (q_tgt_c3 - q_hand) +
                   FLAGS_kd * (-v_hand) + tau_g;
      } else {
        // Pin released: use C3 output.
        tau_hand = s_u * c3->GetInputSolution()[0];
        if (tau_hand.norm() < 1e-6) {
          tau_hand = sim_plant.GetVelocitiesFromArray(
              sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));
        }
      }
    }

    // Clamp only in kC3: C3 output can be arbitrarily large; the reach phase
    // needs unclamped torques (the hybrid showed 30-80 Nm peaks during reach
    // that are necessary to overcome the initial SAP transient and track the
    // spline). Clamping at 10 Nm during reach prevents index/middle from
    // ever reaching the cube.
    // if (phase == kC3) {
    //   tau_hand = tau_hand.cwiseMin(FLAGS_tau_max).cwiseMax(-FLAGS_tau_max);
    // }
    act_fixed.GetMutableVectorData<double>()->SetFromVector(tau_hand);

    simulator.AdvanceTo(t);

    // Kinematic pin: hold cube fixed during reach and for 0.5 s after handoff
    // to give C3 time to warm up before the cube is physically free.
    if (cube_pinned && phase == kC3 && t >= handoff_t + 0.5) {
      cube_pinned = false;
      std::cout << "[t=" << t << "] cube pin released\n";
    }
    if (cube_pinned) {
      sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
      sim_plant.SetVelocities(&plant_ctx, sim_cube, VectorXd::Zero(6));
    }

    // Update Meshcat cube frame and grasp markers.
    const RigidTransform<double> X_WC_now = CubePoseFromPositions(
        sim_plant.GetPositions(plant_ctx, sim_cube));
    update_markers(X_WC_now);

    // Throttled per-step diagnostics.
    if (t >= next_print) {
      next_print += 0.25;
      const VectorXd gp =
          GetGraspPositions(X_WC_now, cube_size, a, b, c_off);
      std::cout << "[t=" << t << "] tip err [idx,mid,thu] = ";
      for (int i = 0; i < 3; ++i) {
        const Vector3d p =
            sim_plant
                .EvalBodyPoseInWorld(plant_ctx,
                                     sim_plant.get_body(tip_bodies[i]))
                .translation();
        std::cout << (p - Vector3d(gp.segment<3>(3 * i))).norm() << " ";
      }
      std::cout << "m";
      if (phase == kC3 && c3) {
        // Per-contact normal force = Σ over that contact's group (works for
        // both Stewart-Trinkle and Anitescu).
        const std::vector<VectorXd>& lam = c3->GetForceSolution();
        std::cout << "  λ_n [idx,mid,thu] = ";
        for (int i = 0; i < 3; ++i) {
          double fn = 0.0;
          for (int idx : normal_groups[i]) fn += lam[0](idx);
          std::cout << fn << " ";
        }
        std::cout << "N";
        const Vector3d cube_p = X_WC_now.translation();
        std::cout << "  cube z=" << cube_p.z();
      }
      std::cout << "\n";
    }
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
