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
#include <chrono>
#include <cmath>
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
#include <drake/solvers/mathematical_program.h>
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
DEFINE_bool(isolate_cube, true,
            "Filter the cube so ONLY the three fingertips can collide with it "
            "(model/physics consistency for C3). WARNING: the reach phase relies "
            "on full-hand collision to seat the fingers — set false to A/B test "
            "if fingers stop reaching the cube.");
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
DEFINE_double(cube_bob_amp, 0.0,
              "Amplitude (m) of the up/down sine the cube z-reference follows "
              "after pin release. 0 = static hold. A moving reference gives C3 "
              "a nonzero tracking error to chase (movement-vs-holding test).");
DEFINE_double(cube_bob_period, 3.0,
              "Period (s) of the cube z-reference sine.");
DEFINE_bool(realize_forces, true,
            "Realize C3's plan with a low-level layer: τ = gravity comp + PD "
            "tracking C3's planned next state + Jᵀ·(C3's planned contact "
            "forces). This is the franka-pipeline architecture (C3 plans, a "
            "low-level controller executes). false = legacy: apply C3's raw "
            "input solution s_u·u[0] directly (known to output ~0 torque: in "
            "the ADMM QP λ is free, so the cube cost is absorbed by λ and "
            "torque is never 'needed').");
DEFINE_double(grip_scale, 1.0,
              "Multiplier on the Jᵀλ grip torque. Diagnostic knob: if the LCS "
              "λ turns out to be impulse-scaled (dt·F) rather than force, "
              "compensate here (e.g. 25 for dt=0.04) without a rebuild.");
DEFINE_double(kp_track, 30.0,
              "PD gain for tracking C3's planned next state in the "
              "realization layer. Deliberately softer than the reach kp: the "
              "plan comes from an ill-conditioned LCS (|A|~600) and can jump; "
              "stiff tracking of a wild plan saturated the torque (321 Nm) "
              "and destroyed the grasp.");
DEFINE_double(kd_track, 2.0,
              "PD derivative gain for plan tracking in the realization layer.");
DEFINE_double(plan_dev_max, 0.1,
              "Clamp on the per-joint position deviation (rad) between C3's "
              "planned next state and the current state before the tracking "
              "PD sees it. Bounds the damage from a garbage plan step.");
DEFINE_double(plan_vel_dev_max, 2.0,
              "Clamp on the per-joint velocity deviation (rad/s) for plan "
              "tracking, same purpose as plan_dev_max.");

// ── OSC (inverse-dynamics QP) realization layer ──────────────────────────────
// The franka pipeline splits control: C3 PLANS (low rate), an Operational-Space
// Controller EXECUTES (high rate) by solving a whole-body inverse-dynamics QP
// that tracks task positions AND contact forces subject to friction cones and
// torque limits. This is the same idea over the 16-DOF Allegro hand: 3 fingertip
// position objectives + 3 fingertip force objectives (targets = C3's planned λ).
DEFINE_bool(use_osc, true,
            "Realize C3's plan with an inverse-dynamics QP (OSC) instead of the "
            "hand-rolled Jᵀλ layer. Tracks fingertip positions + C3's planned "
            "contact forces under friction-cone and torque constraints.");
DEFINE_bool(plan_debug, true,
            "Print C3 plan-quality diagnostics each solve. Factor 1: planned "
            "cube-z trajectory + per-knot state-tracking cost (does C3 THINK it "
            "can hold?). Factor 2a: ADMM consensus residual ‖z−δ‖ per knot (is "
            "the plan physically feasible?). Factor 2b: one-step tracking error "
            "(did we ACHIEVE last solve's predicted next state?).");
DEFINE_double(osc_w_pos, 100.0,
              "OSC weight on fingertip position (task-accel) tracking.");
DEFINE_double(osc_w_force, 1.0,
              "OSC weight on fingertip contact-force tracking.");
DEFINE_double(osc_kp, 100.0, "OSC task-space position gain (fingertips).");
DEFINE_double(osc_kd, 20.0, "OSC task-space damping gain (fingertips).");
DEFINE_double(osc_w_reg, 1e-4, "OSC regularization on joint accelerations.");
DEFINE_double(osc_w_reg_tau, 1e-4, "OSC regularization on joint torques.");
DEFINE_double(force_floor, 0.0,
              "Lower bound (N) on each fingertip's commanded normal force in "
              "the OSC: f_des = max(C3 λ_proj, floor). Thumb floor = 2×. 0 = "
              "realize C3's force as-is (which collapses to 0 at release). Set "
              "≈alpha_m to keep the grip alive across the contact-gate hover "
              "band — the force-floor idea, applied inside the OSC.");
DEFINE_double(phi_offset, 0.003,
              "Gap inflation ε (m): subtract ε from the signed-distance rows "
              "of the LCS constant term c, so the model treats 'within ε of "
              "the face' as touching. The physical gap hovers at φ≈0±1mm "
              "(SAP equilibrium), and strict complementarity gates λ to 0 the "
              "moment φ reads positive — an absorbing zero-grip equilibrium. "
              "The offset keeps the contact active (λ>0 representable) inside "
              "the ε margin, the standard contact-implicit relaxation. 0 = "
              "off.");
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
DEFINE_int32(c3_period_steps, 40,
             "Solve C3 once every this many control steps (control_dt=1ms). "
             "Between solves the previous input solution is reused. 1 = the old "
             "1 kHz behaviour; 40 ≈ one solve per c3_dt. This is the main "
             "compute knob: it does NOT change the plan, only how often it is "
             "recomputed.");
DEFINE_int32(relin_period_steps, 40,
             "Re-linearize the LCS once every this many control steps when "
             "--relinearize is set. Best kept equal to (or a multiple of) "
             "c3_period_steps so each fresh LCS feeds a solve.");
DEFINE_double(osqp_eps, 1e-3,
              "OSQP convergence tolerance (eps_abs = eps_rel).");
DEFINE_int32(admm_iter, 3,
             "ADMM iterations per C3 solve. 3 is very few: if C3 reports a "
             "grip force (λ_n≈target) but commands ~0 torque and the cube "
             "drops at pin release, it likely has not reconciled (u, λ). Try "
             "10-30.");
DEFINE_bool(warm_start, false,
            "Warm-start each C3 solve from the previous solution. Helps "
            "convergence when solving repeatedly at a slowly drifting state.");

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

  // Isolate the cube: filter it against EVERY hand geometry except the three
  // grasping fingertips, so only link_3_tip / link_7_tip / link_15_tip can ever
  // touch it. This makes the physics world match the LCS model world by
  // construction — the LCS only knows those three contact pairs, so any other
  // link touching the cube would inject a force C3 never modelled and corrupt
  // its plan (model mismatch). See grasp_questions_answered.html §Q2.
  //
  // CAVEAT: the reach phase relies on full-hand collision to seat the fingers,
  // so this can stop index/middle from ever reaching the cube. Gated behind
  // --isolate_cube for A/B testing.
  if (FLAGS_isolate_cube) {
    const std::array<std::string, 3> tip_names{"link_3_tip", "link_7_tip",
                                               "link_15_tip"};
    std::vector<GeometryId> non_tip_geoms;
    for (BodyIndex bi : sim_plant.GetBodyIndices(sim_allegro)) {
      const auto& body = sim_plant.get_body(bi);
      if (std::find(tip_names.begin(), tip_names.end(), body.name()) !=
          tip_names.end())
        continue;  // keep the three fingertips able to hit the cube
      const auto& g = sim_plant.GetCollisionGeometriesForBody(body);
      non_tip_geoms.insert(non_tip_geoms.end(), g.begin(), g.end());
    }

    const auto& cube_geoms = sim_plant.GetCollisionGeometriesForBody(
        sim_plant.get_body(sim_plant.GetBodyIndices(sim_cube)[0]));

    sim_scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet(cube_geoms),
            drake::geometry::GeometrySet(non_tip_geoms)));
    std::cout << "[setup] cube-isolation filter APPLIED: cube collides only "
                 "with the 3 fingertips (" << non_tip_geoms.size()
              << " non-tip geoms excluded).\n";
  } else {
    std::cout << "[setup] cube-isolation filter SKIPPED (--isolate_cube=false): "
                 "full-hand collision with the cube is active.\n";
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

  // Base C3 state target (built at handoff) and the cube's nominal z. The C3
  // phase overwrites the cube z-reference with a sine (see --cube_bob_amp).
  VectorXd x_des_base;
  const double cube_z0 = X_WC0.translation().z();
  double last_zref = cube_z0;  // most recent commanded cube-z target (logging)

  // Gap inflation (see --phi_offset): shift the signed-distance rows of c so
  // the model sees φ−ε. Row layout is model-specific (lcs_factory.cc):
  //   stewart_and_trinkle: φ enters c UNSCALED in the λ_n block [nc, 2nc)
  //   anitescu:            φ enters every cone row scaled by 1/dt
  // Applied AFTER the gap diagnostics are read, so the printed φ stays the
  // physical (un-inflated) value.
  auto inflate_gaps = [&](LCS* lcs) {
    if (FLAGS_phi_offset <= 0.0) return;
    std::vector<VectorXd> c_off = lcs->c();
    for (auto& ck : c_off) {
      if (FLAGS_contact_model == "stewart_and_trinkle") {
        ck.segment(n_contacts, n_contacts).array() -= FLAGS_phi_offset;
      } else {  // anitescu: all rows are cone rows
        ck.array() -= FLAGS_phi_offset / FLAGS_c3_dt;
      }
    }
    lcs->set_c(c_off);
  };

  // ── OSC: inverse-dynamics QP over the 16-DOF hand ──────────────────────────
  // Decision vars: joint accel v̇ (16), joint torque τ (16), fingertip contact
  // forces f (3×3, world frame). Minimize fingertip position-tracking + force-
  // tracking + regularization, subject to the hand equations of motion, a
  // friction pyramid per contact, and the torque box. Returns the 16 joint
  // torques. fn_des = the per-fingertip desired NORMAL force (from C3's plan,
  // optionally floored). This is the "executor" layer; C3 is the "planner".
  const double kInf = std::numeric_limits<double>::infinity();
  auto osc_torque = [&](const std::array<double, 3>& fn_des) -> VectorXd {
    const int nv = lcs_plant.num_velocities();  // 22 (== sim_plant)
    const int nvh = n_hand_v;                   // 16

    MatrixXd M(nv, nv);
    sim_plant.CalcMassMatrix(plant_ctx, &M);
    VectorXd Cv(nv);
    sim_plant.CalcBiasTerm(plant_ctx, &Cv);
    const VectorXd tau_g = sim_plant.CalcGravityGeneralizedForces(plant_ctx);
    const VectorXd v_hand = sim_plant.GetVelocities(plant_ctx, sim_allegro);

    // Hand block: the hand (welded base) and the free cube share no bodies, so
    // the mass matrix is block-diagonal and the hand occupies velocity indices
    // [0, nvh) (state-layout comment, §7).
    const MatrixXd Mh = M.topLeftCorner(nvh, nvh);
    const VectorXd Ch = Cv.head(nvh);
    const VectorXd tgh = tau_g.head(nvh);

    // Force press directions use the CURRENT cube rotation (physically, the
    // finger presses the real, possibly-tilted face). But the fingertip POSITION
    // targets use the DESIRED cube pose X_WC0 (0.58, upright): this is what makes
    // the OSC regulate the cube TO its reference — the fingers are pulled toward
    // where they'd sit if the cube were at X_WC0, dragging it back up and
    // untilting it. Targeting the live pose instead just follows the cube
    // wherever it sags (the residual 7 cm droop / 23° tilt seen before).
    const RigidTransform<double> X_WC =
        CubePoseFromPositions(sim_plant.GetPositions(plant_ctx, sim_cube));
    const RotationMatrix<double> R_WC = X_WC.rotation();
    const VectorXd p_des_all =
        GetGraspPositions(X_WC0, cube_size - 0.01, a, b, c_off);
    const std::array<Vector3d, 3> press_C{Vector3d(0, 1, 0), Vector3d(0, 1, 0),
                                          Vector3d(0, -1, 0)};

    drake::solvers::MathematicalProgram prog;
    auto vd = prog.NewContinuousVariables(nvh, "vd");
    auto tau = prog.NewContinuousVariables(nvh, "tau");
    auto f = prog.NewContinuousVariables(9, "f");

    // Equations of motion. f is the force the robot exerts ON the cube (inward,
    // along n̂ᵢ). The reaction ON the robot is −f, contributing −Jhᵀf, so:
    //   M·v̇ + C = τ_g + τ − Jhᵀf   ⇒   Mh·v̇ − τ + Σ Jhᵢᵀ fᵢ = τ_g − C.
    MatrixXd Aeq = MatrixXd::Zero(nvh, 2 * nvh + 9);
    Aeq.leftCols(nvh) = Mh;
    Aeq.middleCols(nvh, nvh) = -MatrixXd::Identity(nvh, nvh);
    const VectorXd beq = tgh - Ch;

    for (int i = 0; i < 3; ++i) {
      MatrixXd J(3, nv);
      sim_plant.CalcJacobianTranslationalVelocity(
          plant_ctx, drake::multibody::JacobianWrtVariable::kV,
          sim_plant.get_body(tip_bodies[i]).body_frame(), Vector3d::Zero(),
          sim_plant.world_frame(), sim_plant.world_frame(), &J);
      const MatrixXd Jh = J.leftCols(nvh);
      Aeq.middleCols(2 * nvh + 3 * i, 3) = Jh.transpose();  // +Jhᵀ (see above)

      // Fingertip position tracking → desired task acceleration a_des.
      const Vector3d p_i =
          sim_plant
              .EvalBodyPoseInWorld(plant_ctx, sim_plant.get_body(tip_bodies[i]))
              .translation();
      const Vector3d pdot = Jh * v_hand;
      const Vector3d Jdotv = sim_plant.CalcBiasTranslationalAcceleration(
          plant_ctx, drake::multibody::JacobianWrtVariable::kV,
          sim_plant.get_body(tip_bodies[i]).body_frame(), Vector3d::Zero(),
          sim_plant.world_frame(), sim_plant.world_frame());
      const Vector3d p_des = p_des_all.segment<3>(3 * i);
      const Vector3d a_des =
          FLAGS_osc_kp * (p_des - p_i) + FLAGS_osc_kd * (-pdot);
      // ‖Jh·v̇ + J̇v − a_des‖²  (drive fingertip accel to a_des)
      prog.Add2NormSquaredCost(std::sqrt(FLAGS_osc_w_pos) * Jh,
                               std::sqrt(FLAGS_osc_w_pos) * (a_des - Jdotv), vd);

      // Contact-force tracking: fᵢ → fn_des[i]·n̂ᵢ (inward normal, world frame).
      const Vector3d n_i = R_WC * press_C[i];
      const Vector3d f_des = fn_des[i] * n_i;
      prog.Add2NormSquaredCost(std::sqrt(FLAGS_osc_w_force) *
                                   Eigen::Matrix3d::Identity(),
                               std::sqrt(FLAGS_osc_w_force) * f_des,
                               f.segment(3 * i, 3));

      // Friction pyramid: n̂ᵀf ≥ 0 and |tⱼᵀf| ≤ (μ/√2)·n̂ᵀf.
      Vector3d t1 = n_i.cross(Vector3d::UnitX());
      if (t1.norm() < 1e-6) t1 = n_i.cross(Vector3d::UnitY());
      t1.normalize();
      const Vector3d t2 = n_i.cross(t1).normalized();
      const double mc = FLAGS_mu / std::sqrt(2.0);
      auto fi = f.segment(3 * i, 3);
      prog.AddLinearConstraint(n_i.transpose(), 0.0, kInf, fi);
      prog.AddLinearConstraint((t1 - mc * n_i).transpose(), -kInf, 0.0, fi);
      prog.AddLinearConstraint((-t1 - mc * n_i).transpose(), -kInf, 0.0, fi);
      prog.AddLinearConstraint((t2 - mc * n_i).transpose(), -kInf, 0.0, fi);
      prog.AddLinearConstraint((-t2 - mc * n_i).transpose(), -kInf, 0.0, fi);
    }

    drake::solvers::VectorXDecisionVariable z(2 * nvh + 9);
    z << vd, tau, f;
    prog.AddLinearEqualityConstraint(Aeq, beq, z);

    prog.AddBoundingBoxConstraint(-FLAGS_tau_max, FLAGS_tau_max, tau);
    prog.Add2NormSquaredCost(
        std::sqrt(FLAGS_osc_w_reg) * MatrixXd::Identity(nvh, nvh),
        VectorXd::Zero(nvh), vd);
    prog.Add2NormSquaredCost(
        std::sqrt(FLAGS_osc_w_reg_tau) * MatrixXd::Identity(nvh, nvh),
        VectorXd::Zero(nvh), tau);

    OsqpSolver osc_solver;
    const auto result = osc_solver.Solve(prog);
    if (!result.is_success()) return tgh;  // fall back to gravity comp
    return result.GetSolution(tau);
  };

  // Profiling + C3-cadence bookkeeping (kC3 phase only). c3_iter counts control
  // steps spent in kC3 and drives the decimation; the *_ms_* accumulators are
  // reported in the throttled diagnostics so you can see where the time goes.
  long c3_iter = 0;
  long relin_calls = 0, solve_calls = 0;
  double relin_ms_sum = 0.0, relin_ms_max = 0.0;
  double solve_ms_sum = 0.0, solve_ms_max = 0.0;
  VectorXd last_gaps;      // most recent Stewart-Trinkle contact gaps φ
  bool have_gaps = false;

  // Factor 2b: the previous solve's predicted next state (GetStateSolution[1]).
  // Solves are c3_period_steps apart = c3_dt (one LCS step), so last solve's x₁
  // predicted "now" — comparing it to the actual current state is the one-step
  // tracking error.
  VectorXd x_plan_prev;
  bool have_prev_plan = false;

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

        inflate_gaps(&lcs_init);
        if (FLAGS_phi_offset > 0.0)
          std::cout << "  gap inflation ON: model sees φ−ε, ε="
                    << FLAGS_phi_offset << " m\n";

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

        x_des_base = x_des;  // captured for the moving-reference update below
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
        c3_opts.admm_iter  = FLAGS_admm_iter;
        c3_opts.rho_scale  = 3;
        c3_opts.warm_start = FLAGS_warm_start;
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

      // Decimate the two expensive calls off the 1 kHz control loop. c3_iter
      // starts at 0, so the very first kC3 step always relinearizes and solves;
      // thereafter each fires on its own period. Between solves the cached input
      // solution (c3->GetInputSolution) is reused unchanged.
      const bool do_relin =
          FLAGS_relinearize &&
          (c3_iter % std::max(1, FLAGS_relin_period_steps) == 0);
      const bool do_solve =
          (c3_iter % std::max(1, FLAGS_c3_period_steps) == 0);
      ++c3_iter;

      if (do_relin) {
        const auto t0 = std::chrono::steady_clock::now();
        lcs_plant.SetPositionsAndVelocities(&lcs_ctx, x_current);
        LCS lcs_new = LCSFactory::LinearizePlantToLCS(
            lcs_plant, lcs_ctx, *lcs_plant_ad, *lcs_ctx_ad,
            contact_pairs, lcs_opts, x_current, VectorXd::Zero(n_u));
        std::vector<MatrixXd> B_sc = lcs_new.B();
        for (auto& Bk : B_sc) Bk *= s_u;
        lcs_new.set_B(B_sc);

        // Cache the contact gaps φ for the diagnostics (Stewart-Trinkle only:
        // φ lives in the λ_n block [n_contacts, 2·n_contacts) of η). ≤0 = the
        // finger is seated; rising above 0 means it has lifted off the cube.
        if (FLAGS_contact_model == "stewart_and_trinkle") {
          const VectorXd eta0 = lcs_new.E()[0] * x_current + lcs_new.c()[0];
          last_gaps = eta0.segment(n_contacts, n_contacts);
          have_gaps = true;
        }

        inflate_gaps(&lcs_new);
        c3->UpdateLCS(lcs_new);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        relin_ms_sum += ms;
        relin_ms_max = std::max(relin_ms_max, ms);
        ++relin_calls;
      }

      // Moving cube reference: overwrite the cube z-target (and its velocity)
      // with a sine once the pin is released, giving C3 a nonzero tracking
      // error to chase. State layout: cube z-position at index n_hand_q+6, cube
      // linear-z velocity at n_pos+n_hand_v+5. The whole horizon k=0..N is
      // filled with the look-ahead so C3 tracks a moving target, not a lagged
      // step. amp=0 or still pinned → base (static) target, unchanged.
      if (FLAGS_cube_bob_amp != 0.0 && !cube_pinned) {
        const double t_ref = std::max(0.0, t - (handoff_t + 0.5));
        const double w = 2.0 * M_PI / FLAGS_cube_bob_period;
        std::vector<VectorXd> x_des_traj(FLAGS_N + 1, x_des_base);
        for (int k = 0; k <= FLAGS_N; ++k) {
          const double tk = t_ref + k * FLAGS_c3_dt;
          x_des_traj[k](n_hand_q + 6) =
              cube_z0 + FLAGS_cube_bob_amp * std::sin(w * tk);
          x_des_traj[k](n_pos + n_hand_v + 5) =
              FLAGS_cube_bob_amp * w * std::cos(w * tk);
        }
        last_zref = x_des_traj[0](n_hand_q + 6);
        c3->UpdateTarget(x_des_traj);
      }

      if (do_solve) {
        const auto t0 = std::chrono::steady_clock::now();
        c3->Solve(x_current);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        solve_ms_sum += ms;
        solve_ms_max = std::max(solve_ms_max, ms);
        ++solve_calls;

        // Simplified diagnostic: printed at each C3 solve instant (not on the
        // 0.25s wall clock). φ is the physical gap (cached from the last
        // relinearization, un-inflated). "C3 u" is the raw input solution
        // (GetInputSolution, scaled to physical Nm) — NOT what is applied by
        // the realization layer, just what C3 itself solved for. "λ_n" is the
        // raw per-contact normal force (GetForceSolution) — can be negative;
        // this is C3's unprojected output, before the ≥0 projection used by
        // the realization layer's Jᵀλ term.
        const VectorXd u0 = s_u * c3->GetInputSolution()[0];
        const std::vector<VectorXd>& lam = c3->GetForceSolution();
        const Eigen::RowVector3d phi_row =
            have_gaps ? Eigen::RowVector3d(last_gaps.transpose())
                      : Eigen::RowVector3d::Zero();
        std::cout << "[t=" << t << "] φ[idx,mid,thu]=" << phi_row
                  << "  C3 u |max|=" << u0.cwiseAbs().maxCoeff()
                  << "  C3 λ_n[idx,mid,thu]=";
        for (int i = 0; i < 3; ++i) {
          double fn = 0.0;
          for (int idx : normal_groups[i]) fn += lam[0](idx);
          std::cout << fn << " ";
        }
        std::cout << "\n";

        if (FLAGS_plan_debug) {
          const std::vector<VectorXd> xplan = c3->GetStateSolution();
          const std::vector<VectorXd> xdes  = c3->GetDesiredState();
          const std::vector<VectorXd> zfull = c3->GetFullSolution();
          const std::vector<VectorXd> delta = c3->GetDualDeltaSolution();

          // ── Factor 1: does the PLAN keep the cube up, and at what cost? ──
          // planned cube-z per knot; if these stay ≈0.58 the plan believes it
          // holds. Per-knot state cost J_k = (x_k−x*_k)ᵀQ(x_k−x*_k): low ⇒ C3
          // thinks it is on target (fail is downstream); high ⇒ C3 knows it
          // cannot hold (Factor 1).
          std::cout << "   [F1] plan cube z=";
          for (const auto& xk : xplan) std::cout << xk(n_hand_q + 6) << " ";
          std::cout << " cost=";
          double Jtot = 0.0;
          for (size_t k = 0; k < xplan.size() && k < xdes.size(); ++k) {
            const VectorXd dx = xplan[k] - xdes[k];
            const double Jk = dx.dot(Q_knot * dx);
            Jtot += Jk;
            std::cout << Jk << " ";
          }
          std::cout << " Jtot=" << Jtot << "\n";

          // ── Factor 2a: ADMM consensus residual ‖z_k − δ_k‖ per knot. ──
          // z = cost/dynamics-optimal copy (may violate complementarity);
          // δ = complementarity-feasible projected copy. At convergence they
          // agree; a large gap ⇒ no single trajectory is both feasible AND
          // what C3 wants ⇒ the committed plan is infeasible/ambiguous.
          std::cout << "   [F2a] |z-δ|=";
          for (size_t k = 0; k < zfull.size() && k < delta.size(); ++k)
            std::cout << (zfull[k] - delta[k]).norm() << " ";
          std::cout << "\n";

          // ── Factor 2b: did we ACHIEVE last solve's one-step prediction? ──
          if (have_prev_plan && x_plan_prev.size() == x_current.size()) {
            const VectorXd terr = x_current - x_plan_prev;
            std::cout << "   [F2b] track err: cube_pos="
                      << terr.segment(n_hand_q + 4, 3).norm()
                      << " cube_quat=" << terr.segment(n_hand_q, 4).norm()
                      << " hand_q=" << terr.head(n_hand_q).norm() << "\n";
          }
          if (xplan.size() > 1) {
            x_plan_prev = xplan[1];
            have_prev_plan = true;
          }
        }
      }

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
      } else if (FLAGS_use_osc) {
        // Pin released — OSC executor. Desired per-fingertip normal force =
        // C3's projected λ, optionally floored so the grip survives the
        // contact-gate hover band. Thumb floor = 2× (force closure).
        const VectorXd lam_proj =
            c3->GetDualDeltaSolution()[0].segment(n_x, n_lambda);
        const std::array<double, 3> floor{FLAGS_force_floor, FLAGS_force_floor,
                                          2.0 * FLAGS_force_floor};
        std::array<double, 3> fn_des{};
        for (int i = 0; i < 3; ++i) {
          double fn = 0.0;
          for (int idx : normal_groups[i]) fn += lam_proj(idx);
          fn_des[i] = std::max(std::max(fn, 0.0), floor[i]);
        }
        tau_hand = osc_torque(fn_des);
      } else if (FLAGS_realize_forces) {
        // Pin released — realization layer (grasp_theory_critique.html §5c).
        // C3 is the PLANNER here; its raw input solution is not applied. In
        // the ADMM QP λ is a free variable, so the cube-tracking cost is
        // absorbed directly by λ and the optimal u is ~0 (verified: |τ| ≈
        // 0.2 Nm at pin release for every tuning tried). The grip lives in
        // C3's planned FORCES, so realize those instead:
        //   τ = τ_g                      (hand holds its own weight)
        //     + Kp(q₁ - q) + Kd(v₁ - v)  (track C3's planned next state)
        //     + Σᵢ Jᵢᵀ fᵢ                (realize C3's planned normal forces)
        const VectorXd tau_g = sim_plant.GetVelocitiesFromArray(
            sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));
        const VectorXd q_hand =
            sim_plant.GetPositions(plant_ctx, sim_allegro);

        // C3's planned state one LCS step ahead (hand joints only). The plan
        // comes from an ill-conditioned linearization (|A|~600), so its next
        // state can jump far from the current one; clamp the deviation the
        // PD sees so a garbage plan step cannot command a violent motion.
        const std::vector<VectorXd>& x_plan = c3->GetStateSolution();
        const VectorXd q1 = x_plan[1].head(n_hand_q);
        const VectorXd v1 = x_plan[1].segment(n_pos, n_hand_v);
        const VectorXd dq =
            (q1 - q_hand)
                .cwiseMax(-FLAGS_plan_dev_max)
                .cwiseMin(FLAGS_plan_dev_max);
        const VectorXd dv = (v1 - v_hand)
                                .cwiseMax(-FLAGS_plan_vel_dev_max)
                                .cwiseMin(FLAGS_plan_vel_dev_max);

        // Projected (feasible) contact forces from the ADMM copy variable:
        // z = [x; λ; u] → λ = delta.segment(n_x, n_lambda). Unlike the QP's
        // λ (GetForceSolution), the projected copy respects λ ≥ 0.
        const VectorXd lam_proj =
            c3->GetDualDeltaSolution()[0].segment(n_x, n_lambda);

        // Inward press directions in the cube frame (theory html §1):
        // index/middle press the −Y face along +ŷ, thumb presses +Y along −ŷ.
        const RotationMatrix<double> R_WC =
            CubePoseFromPositions(sim_plant.GetPositions(plant_ctx, sim_cube))
                .rotation();
        const std::array<Vector3d, 3> press_C{Vector3d(0, 1, 0),
                                              Vector3d(0, 1, 0),
                                              Vector3d(0, -1, 0)};

        VectorXd tau_grip = VectorXd::Zero(n_hand);
        for (int i = 0; i < 3; ++i) {
          double fn = 0.0;
          for (int idx : normal_groups[i]) fn += lam_proj(idx);
          fn = std::max(fn, 0.0);
          const Vector3d f_W = fn * (R_WC * press_C[i]);
          Eigen::MatrixXd J(3, sim_plant.num_velocities());
          sim_plant.CalcJacobianTranslationalVelocity(
              plant_ctx, drake::multibody::JacobianWrtVariable::kV,
              sim_plant.get_body(tip_bodies[i]).body_frame(),
              Vector3d::Zero(), sim_plant.world_frame(),
              sim_plant.world_frame(), &J);
          // Hand velocities are the first n_hand_v entries of v, so the hand
          // columns of J are its left block (state-layout comment, §7).
          tau_grip += J.leftCols(n_hand_v).transpose() * f_W;
        }

        tau_hand = tau_g + FLAGS_kp_track * dq + FLAGS_kd_track * dv +
                   FLAGS_grip_scale * tau_grip;

      } else {
        // Legacy path: apply C3's raw input solution directly. Kept for A/B
        // only — structurally outputs ~0 torque at a grasp equilibrium.
        tau_hand = s_u * c3->GetInputSolution()[0];
        if (tau_hand.norm() < 1e-6) {
          tau_hand = sim_plant.GetVelocitiesFromArray(
              sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));
        }
      }
    }

    // Clamp only in kC3: the realization/C3 output must never exceed the
    // physical torque budget (an unclamped 321 Nm spike at pin release wrecked
    // the grasp). The reach phase stays unclamped — it needs 30-80 Nm peaks to
    // overcome the initial SAP transient and track the spline.
    if (phase == kC3) {
      tau_hand = tau_hand.cwiseMin(FLAGS_tau_max).cwiseMax(-FLAGS_tau_max);
    }
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
      if (phase == kReach) {
        // Joint-space tracking error of each finger vs its seat target
        // q_contact. LARGE here (with small tip err impossible) → the PD is
        // too weak to hold the finger against gravity/dynamics (raise kp/kd).
        // SMALL here but tip err large → a kinematic/IK problem instead.
        const VectorXd q_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);
        std::cout << "  qErr[idx,mid,thu]=";
        for (int i = 0; i < 3; ++i)
          std::cout << (q_hand.segment(finger_start[i], 4) -
                        q_contact.segment(finger_start[i], 4))
                           .norm()
                    << " ";
        std::cout << "rad  arrived=" << arrived[0] << arrived[1] << arrived[2];
      }
      if (phase == kC3 && c3) {
        // Minimal outcome check — the per-solve line above (φ, C3's raw u/λ)
        // carries the diagnostic detail now. This just tracks the cube.
        const Vector3d cube_p = X_WC_now.translation();
        const double ang_err = X_WC_now.rotation().ToAngleAxis().angle();
        std::cout << "  cube z=" << cube_p.z() << " ang=" << ang_err;
      }
      std::cout << "\n";
    }
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
