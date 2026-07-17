// Two-phase grasp controller — §12 of grasp_trajopt_formulation.html.
//
//   Phase 1 (kReach): PD + cubic spline drives the three fingertips to the
//                     cube faces. The cube is kinematically pinned. Logic is
//                     identical to allegro_grasp_hybrid.cc.
//
//   Handoff: when all three fingers report contact, the LCS is linearized at
//            the contact configuration, and the cube pin is held briefly
//            (0.5 s) before release. The grasping fingers are PD-held at
//            q_contact (3 mm inside the faces) so they keep a real preload.
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <drake/common/trajectories/piecewise_polynomial.h>
#include <drake/geometry/meshcat.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/geometry/rgba.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/shape_specification.h>
#include <drake/lcm/drake_lcm.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/plant/coulomb_friction.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/solvers/osqp_solver.h>
#include <drake/solvers/solver_options.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/fixed_input_port_value.h>
#include <drake/systems/framework/system.h>
#include <drake/systems/lcm/lcm_interface_system.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

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
#include "systems/senders/c3_state_sender.h"

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
using drake::systems::lcm::LcmInterfaceSystem;
using drake::systems::lcm::LcmPublisherSystem;
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
DEFINE_double(kp, 200.0, "PD proportional gain (Nm/rad) for the reach phase.");
DEFINE_double(kd, 10.0,
              "PD derivative gain (Nm·s/rad) for the reach phase.");
DEFINE_double(tau_max, 1000.0,
              "Per-joint torque box bound (Nm). Applied as INPUT constraints "
              "inside the C3 QP to keep it bounded (prevents DualInfeasible).");
DEFINE_double(contact_force_thresh, 0.02,
              "Per-finger contact force threshold to count as touching (N).");
DEFINE_double(penetration_index_middle, -0.0035,
              "q_contact target: how far the TRUE fingertip surface (see "
              "--tip_surface_offset_z) sits inside the -Y cube face for the "
              "index and middle fingers (m).");
DEFINE_double(penetration_thumb, -0.008,
              "q_contact target: how far the TRUE fingertip surface sits "
              "inside the +Y cube face for the thumb (m). 0 = flush with "
              "the nominal face, no penetration.");
DEFINE_double(tip_surface_offset_z, 0.0115,
              "Offset (m) along each *_tip body frame's own local +Z from "
              "the frame ORIGIN to the true fingertip collision SURFACE — "
              "confirmed empirically via the /tip_frame/* Meshcat triads "
              "(--tip_frame_viz_offset_z sweep). Used for TWO things: (1) "
              "the /tip_frame/* triad visualization offset, and (2) as the "
              "tip_frame_offset passed to SolveGraspIK when solving "
              "q_contact/q_pregrasp, so the IK constrains the real surface "
              "point (not the frame origin) to sit at the intended margin "
              "from the cube face.");
DEFINE_bool(contact_force_log, false,
            "Print the sim plant's resolved contact force (ContactResults, "
            "one PointPairContactInfo per colliding geometry pair), "
            "throttled to --contact_force_log_hz and only when all 3 "
            "fingertip-cube pairs are simultaneously in contact. Suppresses "
            "the C3 PLAN / --plan_debug SOLVER DIAG tables, the relin "
            "|A|/|B|/|D|/|v_hand| line, and the tau(unclamped) saturation "
            "line while active, so this is the ONLY per-step output.");
DEFINE_double(contact_force_log_hz, 10.0,
              "Print rate (Hz) for --contact_force_log.");
DEFINE_bool(lcm_publish, true,
            "Publish live LCM state telemetry (full sim plant [q;v], "
            "channel GRASP_STATE, message dairlib::lcmt_c3_state) for "
            "real-time plotting via signal-scope.");
DEFINE_double(lcm_publish_hz, 100.0,
              "Publish rate (Hz) for --lcm_publish.");
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
DEFINE_double(handoff_settle_time, 0.15,
              "Seconds to wait after the LAST finger's arrived[] latches "
              "before triggering the C3 handoff. Both the LCS gap φ and the "
              "instantaneous contact-force detector proved too noisy at the "
              "1ms scale to gate on directly (φ swings tens of mm on a "
              "static finger; contact force flickers across its threshold "
              "near quasi-static equilibrium) — a fixed dwell time lets the "
              "PD hold settle into a stable preload instead.");

// ── C3-squeeze-phase flags ───────────────────────────────────────────────────
DEFINE_double(k_hold, 10.0,
              "Q cost weight on hand-joint deviations from q_contact. Keeps "
              "fingertips on the cube faces inside the ADMM solve.");
DEFINE_double(w_cube, 10000.0,
              "Q cost weight on cube POSE only (world-frame anchor). This is "
              "the load-bearing term: C3 must keep the cube here against "
              "gravity, which forces Σμλ_n ≥ mg. Was 1000 (shared with cube "
              "velocity); now 10× higher and pose-only, since the cost "
              "breakdown showed the velocity term (below) dominating and "
              "drowning out the position anchor.");
DEFINE_double(w_cube_vel, 10.0,
              "Q cost weight on cube VELOCITY (split out from w_cube). Was "
              "folded into w_cube at 1000; now 100× lower (10) so it no "
              "longer dominates the objective over the pose anchor.");
DEFINE_double(w_vel, 0.1,
              "Q cost weight on hand joint velocities (light damping).");
DEFINE_double(w_R, 0.01, "R cost weight on joint torques.");
DEFINE_double(w_G, 1.0, "ADMM augmented-Lagrangian G weight.");
DEFINE_double(w_U, 1.0, "ADMM augmented-Lagrangian U weight.");
DEFINE_double(w_lambda, 0.0,
              "Force-reference cost on the three normal contact forces "
              "λ_n[3,4,5]. Regularises the squeeze nullspace (§11.5) and gives "
              "the solve a physical grip target. 0 = disabled (default).");
DEFINE_double(alpha_m, 1.0,
              "Per-finger normal grip-force target (N) for the λ_n reference. "
              "Thumb target = 2·alpha_m (force closure vs index+middle).");
DEFINE_double(cube_bob_amp, 0.0,
              "Amplitude (m) of the up/down sine the cube z-reference follows "
              "after pin release. 0 = static hold. A moving reference gives C3 "
              "a nonzero tracking error to chase (movement-vs-holding test).");
DEFINE_double(cube_bob_period, 3.0,
              "Period (s) of the cube z-reference sine.");
DEFINE_double(push_distance, 0.04,
              "PUSH TEST: distance (m) to push the cube in +Y (toward where "
              "the thumb was) by index+middle. The cube's Y target is set to "
              "its start Y + this. Cube z is kinematically fixed at 0.58 "
              "(the 'table'); only x/y/rotation are free.");
// ── Low-level realization layer ──────────────────────────────────────────────
// C3 PLANS (low rate); a task-space PD + Jacobian-transpose grip EXECUTES (high
// rate, every control tick): τ = τ_g + Σ Jᵢᵀ[Kp(p_des−p) − Kd·ṗ + fₙ·n̂], with
// C3's raw input solution added as feedforward. p_des is a fingertip position
// target (fixed grasp points, or C3's planned config — see --fk_target); fₙ is
// the desired per-fingertip normal force (C3's projected λ, optionally floored
// — see --force_floor).
DEFINE_bool(plan_debug, false,
            "Replace the per-solve C3 PLAN printout (contact gap / normal "
            "force / cube-z tables) with solver-quality diagnostics: "
            "(1) LINEARIZATION ACCURACY — the previous solve's knot-ahead "
            "state prediction vs. the actual measured state now (is the LCS "
            "a good model of reality?). (2) DYNAMICS RESIDUAL — per knot, "
            "‖x_{k+1} − (A x_k + B u_k + D λ_k + d)‖ for the SOLVED "
            "trajectory (should be ~0; large values mean the ADMM/QP solve "
            "didn't actually satisfy its own equality constraint). "
            "(3) COMPLEMENTARITY RESIDUAL — per knot, feasibility of "
            "λ≥0 / η≥0 and the gap max|λ⊙η| (should be ~0 for a clean LCP "
            "solution; large values mean a poorly-resolved contact mode).");
DEFINE_double(task_kp, 70.0, "Task-space position gain (fingertips).");
DEFINE_double(task_kd, 0.0, "Task-space damping gain (fingertips).");
DEFINE_double(force_floor, 0.0,
              "Lower bound (N) on each fingertip's commanded normal force: "
              "f_des = max(C3 λ_proj, floor). Thumb floor = 2×. 0 = realize "
              "C3's force as-is (which collapses to 0 at release). Set "
              "≈alpha_m to keep the grip alive across the contact-gate hover "
              "band — the force-floor idea.");
DEFINE_bool(fk_target, false,
            "Fingertip POSITION target source. DEFAULT false = fixed grasp "
            "points on the upright reference cube X_WC0 (the geometric hold — "
            "the setup that holds the cube). true = forward-kinematics of C3's "
            "planned next hand config q1 (GetStateSolution()[1]): track where "
            "the plan says the fingertips go next. NOTE: C3's plan is currently "
            "passive (u≈0 ⇒ q1≈now), so 'true' targets ≈ the current fingertip "
            "positions and tends to let the cube sag — the honest test of "
            "executing C3's plan.");
DEFINE_int32(N, 5, "C3 prediction horizon (knot points).");
DEFINE_double(c3_dt, 0.05, "LCS timestep (s) for C3 linearization.");
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
DEFINE_int32(admm_iter, 20,
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

  // PUSH TEST: a real frictional table (half-space) the cube rests on.
  // Surface at z = 0.58 - cube_size/2 = 0.55 (cube center 0.58, half 0.03),
  // anchored to the world body. Same table is added to the LCS plant below
  // so C3 models the cube-table contact and friction. Friction matches the
  // cube SDF (μ_static 0.8, μ_dynamic 0.5).
  const double kTableZ = 0.55;
  const drake::multibody::CoulombFriction<double> kTableFriction(0.8, 0.5);
  const RigidTransform<double> X_WTable =
      drake::geometry::HalfSpace::MakePose(Vector3d::UnitZ(),
                                           Vector3d(0, 0, kTableZ));
  const GeometryId sim_table_geom = sim_plant.RegisterCollisionGeometry(
      sim_plant.world_body(), X_WTable, drake::geometry::HalfSpace(),
      "table_collision", kTableFriction);
  sim_plant.RegisterVisualGeometry(
      sim_plant.world_body(), X_WTable, drake::geometry::HalfSpace(),
      "table_visual", Eigen::Vector4d(0.5, 0.5, 0.5, 0.5));

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
    // PUSH TEST: only index (link_3_tip) and middle (link_7_tip) may touch
    // the cube. The thumb tip is deliberately NOT in this list, so the
    // isolation filter below also excludes the thumb from the cube — the
    // thumb is out of consideration entirely.
    const std::array<std::string, 2> tip_names{"link_3_tip", "link_7_tip"};
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
                 "with index+middle tips (" << non_tip_geoms.size()
              << " geoms excluded, incl. thumb).\n";
  } else {
    std::cout << "[setup] cube-isolation filter SKIPPED (--isolate_cube=false): "
                 "full-hand collision with the cube is active.\n";
  }

  // PUSH TEST: filter the table against the whole hand so only the cube ever
  // touches the table (fingers must not collide with the ground plane).
  {
    std::vector<GeometryId> all_hand_geoms;
    for (BodyIndex bi : sim_plant.GetBodyIndices(sim_allegro)) {
      const auto& g = sim_plant.GetCollisionGeometriesForBody(
          sim_plant.get_body(bi));
      all_hand_geoms.insert(all_hand_geoms.end(), g.begin(), g.end());
    }
    sim_scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet({sim_table_geom}),
            drake::geometry::GeometrySet(all_hand_geoms)));
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

  // Live triads at the "*_tip" BODY FRAME ORIGIN that SolveGraspIK actually
  // constrains (cube_kinematics.h) — i.e. exactly the point q_contact was
  // solved to place 3mm inside the nominal cube face. Small scale (fingertip-
  // sized) vs. the cube_frame triad above. If this frame visibly sits deeper
  // than the fingertip's rendered geometry, that confirms the frame origin
  // does not coincide with the true collision surface.
  AddFrameTriad(meshcat.get(), "/tip_frame/index",  0.0015, 0.02);
  AddFrameTriad(meshcat.get(), "/tip_frame/middle", 0.0015, 0.02);
  AddFrameTriad(meshcat.get(), "/tip_frame/thumb",  0.0015, 0.02);

  const drake::geometry::Rgba kRed(1.0, 0.0, 0.0, 1.0);
  meshcat->SetObject("/grasp/index",  drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/middle", drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/thumb",  drake::geometry::Sphere(0.006), kRed);

  // ══════════════════════════════════════════════════════════════════════════
  // Live LCM state telemetry for real-time plotting (signalscope). Publishes
  // the full sim-plant state [q;v] (hand + cube) on channel GRASP_STATE as
  // dairlib::lcmt_c3_state, at --lcm_publish_hz. Values are pushed into
  // state_sender's input port every control step below (see state_input
  // FixedInputPortValue); the periodic LcmPublisherSystem below picks up
  // whatever is currently set whenever its own schedule fires during
  // simulator.AdvanceTo(). Reuses dairlib's existing C3StateSender/
  // lcmt_c3_state — not C3-specific in meaning here, just a convenient
  // named-float-vector-over-LCM message that already exists.
  systems::C3StateSender* state_sender = nullptr;
  systems::C3StateSender* tau_sender = nullptr;
  drake::lcm::DrakeLcm drake_lcm;
  if (FLAGS_lcm_publish) {
    // Layout matches this file's own state-vector convention (see the
    // n_hand_q/n_pos/n_hand_v comments elsewhere): 16 hand joint positions,
    // then cube [qw,qx,qy,qz,x,y,z], then 16 hand joint velocities, then
    // cube [wx,wy,wz,vx,vy,vz].
    std::vector<std::string> state_names;
    for (int i = 0; i < sim_plant.num_positions(sim_allegro); ++i)
      state_names.push_back("hand_q" + std::to_string(i));
    state_names.insert(state_names.end(),
                       {"cube_qw", "cube_qx", "cube_qy", "cube_qz", "cube_x",
                        "cube_y", "cube_z"});
    for (int i = 0; i < sim_plant.num_velocities(sim_allegro); ++i)
      state_names.push_back("hand_v" + std::to_string(i));
    state_names.insert(state_names.end(), {"cube_wx", "cube_wy", "cube_wz",
                                           "cube_vx", "cube_vy", "cube_vz"});

    const int n_x_full = sim_plant.num_positions() + sim_plant.num_velocities();
    DRAKE_DEMAND(static_cast<int>(state_names.size()) == n_x_full);

    auto* lcm_iface =
        sim_builder.AddSystem<LcmInterfaceSystem>(&drake_lcm);
    state_sender = sim_builder.AddSystem<systems::C3StateSender>(
        n_x_full, state_names);
    auto* state_pub = sim_builder.AddSystem(
        LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
            "GRASP_STATE", lcm_iface, 1.0 / FLAGS_lcm_publish_hz));
    sim_builder.Connect(state_sender->get_output_port_target_c3_state(),
                        state_pub->get_input_port());

    // Second channel, reusing state_sender's otherwise-unused
    // final_target_state input/output pair: q_contact (the IK-solved hand
    // config the reach phase holds each finger at), same state_names/size
    // convention as GRASP_STATE, so plotting tools can overlay actual vs.
    // target by name. Set ONCE after q_contact is computed below (see
    // q_contact_input) — it's a static target, not a per-step signal.
    auto* q_contact_pub = sim_builder.AddSystem(
        LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
            "GRASP_Q_CONTACT", lcm_iface, 1.0 / FLAGS_lcm_publish_hz));
    sim_builder.Connect(state_sender->get_output_port_final_target_c3_state(),
                        q_contact_pub->get_input_port());

    // Third channel: applied hand torque (tau_hand), one entry per hand
    // joint — a separate, smaller sender (no cube slots needed). Updated
    // every control step below (see tau_input), same as state_input.
    std::vector<std::string> tau_names;
    for (int i = 0; i < sim_plant.num_positions(sim_allegro); ++i)
      tau_names.push_back("tau" + std::to_string(i));
    tau_sender = sim_builder.AddSystem<systems::C3StateSender>(
        sim_plant.num_positions(sim_allegro), tau_names);
    // C3StateSender's constructor hardcodes set_name("c3_state_sender") —
    // override it so this second instance doesn't collide with
    // state_sender's name (Drake requires unique subsystem names).
    tau_sender->set_name("tau_sender");
    auto* tau_pub = sim_builder.AddSystem(
        LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
            "GRASP_TAU", lcm_iface, 1.0 / FLAGS_lcm_publish_hz));
    sim_builder.Connect(tau_sender->get_output_port_target_c3_state(),
                        tau_pub->get_input_port());
  }

  auto sim_diagram = sim_builder.Build();
  Simulator<double> simulator(*sim_diagram);
  simulator.set_target_realtime_rate(1.0);
  auto& sim_ctx = simulator.get_mutable_context();
  auto& plant_ctx =
      sim_diagram->GetMutableSubsystemContext(sim_plant, &sim_ctx);

  // Mutable handle to state_sender's input, updated every control step in
  // the main loop below (see "LCM telemetry" comment there). Null when
  // --lcm_publish=false.
  drake::systems::FixedInputPortValue* state_input = nullptr;
  // q_contact_input: set ONCE below, right after q_contact is computed (a
  // static IK-solved target, not a per-step signal) — see the "LIVE LCM
  // state telemetry" wiring above for what channel this feeds.
  drake::systems::FixedInputPortValue* q_contact_input = nullptr;
  if (state_sender != nullptr) {
    auto& state_sender_ctx =
        sim_diagram->GetMutableSubsystemContext(*state_sender, &sim_ctx);
    const int n_x_full =
        sim_plant.num_positions() + sim_plant.num_velocities();
    state_input = &state_sender->get_input_port_target_state().FixValue(
        &state_sender_ctx, VectorXd::Zero(n_x_full));
    q_contact_input =
        &state_sender->get_input_port_final_target_state().FixValue(
            &state_sender_ctx, VectorXd::Zero(n_x_full));
  }

  // tau_input: updated every control step in the main loop (see "LCM
  // telemetry" comment there) with whatever torque is actually being
  // applied that tick, in both phases.
  drake::systems::FixedInputPortValue* tau_input = nullptr;
  if (tau_sender != nullptr) {
    auto& tau_sender_ctx =
        sim_diagram->GetMutableSubsystemContext(*tau_sender, &sim_ctx);
    tau_input = &tau_sender->get_input_port_target_state().FixValue(
        &tau_sender_ctx, VectorXd::Zero(sim_plant.num_positions(sim_allegro)));
  }

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
  // PUSH TEST: identical table in the LCS plant so the factory builds a
  // cube-table contact (weight support + friction that resists the push).
  const GeometryId lcs_table_geom = lcs_plant.RegisterCollisionGeometry(
      lcs_plant.world_body(),
      drake::geometry::HalfSpace::MakePose(Vector3d::UnitZ(),
                                           Vector3d(0, 0, kTableZ)),
      drake::geometry::HalfSpace(), "table_collision", kTableFriction);
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
  // PUSH TEST: thumb removed; contacts are index-cube, middle-cube, and
  // cube-table (the ground). 3 contacts. Contact ORDER matters: the two
  // finger contacts come first (the executor realizes only those), the
  // cube-table contact is last (physical, realized by the sim's table).
  const std::vector<SortedPair<GeometryId>> contact_pairs{
      SortedPair<GeometryId>(lcs_index_geom,  lcs_cube_geom),
      SortedPair<GeometryId>(lcs_middle_geom, lcs_cube_geom),
      SortedPair<GeometryId>(lcs_cube_geom,   lcs_table_geom)};

  // ══════════════════════════════════════════════════════════════════════════
  // 4. IK: q_contact (fingertips 3 mm inside faces) and q_pregrasp (1 cm
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

  // Point (expressed in each "*_tip" frame) actually constrained by IK — the
  // frame ORIGIN does not coincide with the true fingertip collision
  // surface, which sits --tip_surface_offset_z along the frame's own local
  // +Z (confirmed empirically via the /tip_frame/* Meshcat triads). Passing
  // this to SolveGraspIK makes the IK target the real surface point instead
  // of the origin, so "3 mm inside the face" means the fingertip SURFACE is
  // 3 mm inside, not the frame origin (which was the punch-through bug).
  const Vector3d tip_surface_pt(0, 0, FLAGS_tip_surface_offset_z);

  auto solve_ik = [&](const char* label, const VectorXd& targets) {
    SetThumbSeed(sim_plant, sim_allegro, &plant_ctx);
    sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
    sim_plant.SetPositions(
        &plant_ctx, SolveGraspIK(sim_plant, &plant_ctx, targets,
                                 tip_surface_pt));
    const VectorXd q = sim_plant.GetPositions(plant_ctx, sim_allegro);
    double err = 0;
    for (int i = 0; i < 3; ++i) {
      // FK of the constrained SURFACE point (frame origin transformed by
      // tip_surface_pt), not the raw frame-origin translation, so this
      // error check matches what SolveGraspIK actually solved for.
      const Vector3d p =
          sim_plant
              .EvalBodyPoseInWorld(plant_ctx,
                                   sim_plant.get_body(tip_bodies[i])) *
          tip_surface_pt;
      err += (p - targets.template segment<3>(3 * i)).norm();
    }
    std::cout << "IK " << label << " total FK error = " << err << " m\n";
    return q;
  };

  // q_contact target: asymmetric penetration, per finger — index/middle
  // --penetration_index_middle inside the -Y face, thumb
  // --penetration_thumb inside the +Y face. GetGraspPositions only takes a
  // single shared margin (same inset on both faces), so build the 9-vector
  // target directly here, mirroring its internal formula.
  const double h_cube = cube_size / 2.0;
  VectorXd q_contact_targets(9);
  q_contact_targets << X_WC0 * Vector3d(a, -(h_cube - FLAGS_penetration_index_middle), 0),
                       X_WC0 * Vector3d(b, -(h_cube - FLAGS_penetration_index_middle), 0),
                       X_WC0 * Vector3d(c_off, h_cube - FLAGS_penetration_thumb, 0);
  const VectorXd q_contact = solve_ik("contact", q_contact_targets);

  // Publish q_contact once on GRASP_Q_CONTACT (see the LCM wiring above) —
  // it's a static IK target, doesn't change after this point. Only the
  // hand_q* entries are meaningful; the rest of the n_x_full-sized vector
  // is left zero.
  if (q_contact_input != nullptr) {
    VectorXd q_contact_full =
        VectorXd::Zero(sim_plant.num_positions() + sim_plant.num_velocities());
    q_contact_full.head(q_contact.size()) = q_contact;
    q_contact_input->GetMutableVectorData<double>()->SetFromVector(
        q_contact_full);
  }

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

    // Live "*_tip" body-frame-origin triads, offset by --tip_frame_viz_offset_z
    // along the frame's OWN local Z (i.e. X_WF * Translation(0,0,offset), not
    // a world-frame offset) — a visualization-only probe for where the
    // fingertip's true collision surface sits relative to the frame origin
    // that SolveGraspIK actually constrains. Does not affect q_contact/
    // q_pregrasp or anything physical, purely a Meshcat marker offset.
    const std::array<const char*, 3> tip_paths{
        "/tip_frame/index", "/tip_frame/middle", "/tip_frame/thumb"};
    const RigidTransform<double> X_offset(
        Vector3d(0, 0, FLAGS_tip_surface_offset_z));
    for (int i = 0; i < 3; ++i)
      meshcat->SetTransform(
          tip_paths[i],
          sim_plant.EvalBodyPoseInWorld(
              plant_ctx, sim_plant.get_body(tip_bodies[i])) *
              X_offset);
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
  lcs_opts.num_contacts = 3;  // PUSH TEST: index-cube, middle-cube, cube-table
  lcs_opts.num_friction_directions = FLAGS_num_friction_directions;
  lcs_opts.mu = FLAGS_mu;

  const int n_x = lcs_plant.num_positions() + lcs_plant.num_velocities();
  const int n_u = lcs_plant.num_actuators();
  const int n_lambda = LCSFactory::GetNumContactVariables(
      GetContactModelMap().at(lcs_opts.contact_model), 3,
      FLAGS_num_friction_directions);
  // Of the 3 contacts, only the first 2 (index, middle) are FINGER contacts
  // the executor realizes; contact 2 (cube-table) is physical.
  const int n_finger_contacts = 2;
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
  //   k_hold     on hand joint positions  → keeps fingers on cube faces
  //   w_cube     on cube POSE             → world-frame cube anchor (key term)
  //   w_vel      on hand joint velocities → light damping
  //   w_cube_vel on cube VELOCITY         → split from w_cube, 100× lower
  MatrixXd Q_knot = MatrixXd::Zero(n_x, n_x);
  for (int i = 0; i < n_hand_q; ++i)          Q_knot(i, i) = FLAGS_k_hold;
  for (int i = n_hand_q; i < n_pos; ++i)      Q_knot(i, i) = FLAGS_w_cube;
  for (int i = n_pos; i < n_pos + n_hand_v; ++i) Q_knot(i, i) = FLAGS_w_vel;
  for (int i = n_pos + n_hand_v; i < n_x; ++i)   Q_knot(i, i) = FLAGS_w_cube_vel;

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

  std::array<bool, 3>    arrived{false, false, false};
  std::array<double, 3>  arrived_time{-1.0, -1.0, -1.0};  // set when arrived[i] latches
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

  // Profiling + C3-cadence bookkeeping (kC3 phase only). c3_iter counts control
  // steps spent in kC3 and drives the decimation; the *_ms_* accumulators are
  // reported in the throttled diagnostics so you can see where the time goes.
  long c3_iter = 0;
  long relin_calls = 0, solve_calls = 0;
  double relin_ms_sum = 0.0, relin_ms_max = 0.0;
  double solve_ms_sum = 0.0, solve_ms_max = 0.0;

  // Previous solve's full state plan + the sim time it was solved at, kept
  // only for --plan_debug's linearization-accuracy check (compares that
  // plan's knot-ahead prediction against the actual measured state now).
  std::vector<VectorXd> xplan_prev;
  double t_prev_solve = -1.0;

  // Scratch plant context for forward-kinematics of C3's planned hand config
  // (--fk_target). Standalone (not the live sim context) so setting it does not
  // disturb the simulation; FK needs no scene-graph query object.
  auto fk_ctx = sim_plant.CreateDefaultContext();

  // --contact_force_log cadence: control_dt is 1kHz, throttle down to
  // --contact_force_log_hz.
  long contact_log_iter = 0;
  const int contact_log_period_steps = std::max(
      1, static_cast<int>(
             std::lround(1.0 / (FLAGS_contact_force_log_hz * control_dt))));

  for (double t = control_dt; t < FLAGS_sim_time; t += control_dt) {
    // Heartbeat (every 0.25 s): cube (y,z), index/middle fingertip Y vs the
    // cube's -Y face (they must approach it to contact), and the total
    // number of contact pairs (a large count would mean the hand is
    // spuriously hitting the table).
    if (std::llround(t / control_dt) % 250 == 0) {
      const VectorXd q_cube_hb = sim_plant.GetPositions(plant_ctx, sim_cube);
      const double face_y = q_cube_hb(5) - 0.03;  // -Y face (cube half 0.03)
      const double idx_y =
          sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                        sim_plant.get_body(tip_bodies[0]))
              .translation().y();
      const double mid_y =
          sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                        sim_plant.get_body(tip_bodies[1]))
              .translation().y();
      const int n_pp = sim_plant.get_contact_results_output_port()
                           .Eval<ContactResults<double>>(plant_ctx)
                           .num_point_pair_contacts();
      std::cout << "[t=" << t << "] phase="
                << (phase == kReach ? "reach" : "C3")
                << "  cube_y=" << q_cube_hb(5) << " face_y=" << face_y
                << "  idx_tip_y=" << idx_y << " mid_tip_y=" << mid_y
                << "  contacts=" << n_pp << "\n";
    }

    const VectorXd v_hand = sim_plant.GetVelocities(plant_ctx, sim_allegro);

    // LCM telemetry: push the current full state into state_sender's fixed
    // input every control step. The LcmPublisherSystem wired to it (see
    // "Live LCM state telemetry" above) is on its own --lcm_publish_hz
    // period and reads whatever is here whenever simulator.AdvanceTo()
    // crosses its next scheduled publish time.
    if (state_input != nullptr) {
      state_input->GetMutableVectorData<double>()->SetFromVector(
          sim_plant.GetPositionsAndVelocities(plant_ctx));
    }

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

    // Raw resolved contact force per colliding pair, straight from
    // ContactResults — no threshold filtering, no C3 involved. Throttled to
    // --contact_force_log_hz, printed from t=0 regardless of contact count
    // (the leading "contacts=N" lets you see the count climb toward 3 over
    // time and compare that against the "finger i arrived" / handoff prints).
    if (FLAGS_contact_force_log &&
        contact_log_iter % contact_log_period_steps == 0) {
      std::cout << "[t=" << t << "] contacts="
                << contacts.num_point_pair_contacts();
      for (int k = 0; k < contacts.num_point_pair_contacts(); ++k) {
        const auto& info = contacts.point_pair_contact_info(k);
        const Vector3d f = info.contact_force();
        std::cout << "  [" << sim_plant.get_body(info.bodyA_index()).name()
                  << "-" << sim_plant.get_body(info.bodyB_index()).name()
                  << " f=(" << f.x() << "," << f.y() << "," << f.z()
                  << ") |f|=" << f.norm() << "]";
      }
      std::cout << "\n";
    }
    ++contact_log_iter;

    VectorXd tau_hand(n_hand);

    // ── Phase 1: reach ──────────────────────────────────────────────────────
    if (phase == kReach) {
      const VectorXd q_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);

      // Latch any finger that just made contact (after the guard time).
      for (int i = 0; i < 3; ++i) {
        if (!arrived[i] && touching[i] && t > FLAGS_contact_enable_t) {
          arrived[i] = true;
          arrived_time[i] = t;
          std::cout << "[t=" << t << "] finger " << i << " arrived\n";
        }
      }

      const double tl = std::clamp(t, 0.0, traj.end_time());
      VectorXd q_tgt   = traj.value(tl).col(0);
      VectorXd qd_tgt  = traj_dot.value(tl).col(0);
      for (int i = 0; i < 3; ++i) {
        if (arrived[i]) {
          // Hold the grasping finger at q_contact (3 mm inside the face) so the
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

      // ── Handoff: all three fingers arrived, settle time elapsed ─────────
      // Gated on a fixed dwell time since the LAST finger's arrived[] latch,
      // not on any per-step contact measurement — both the LCS gap φ and the
      // instantaneous contact-force detector proved too noisy at the 1ms
      // scale (see handoff_settle_time flag). The PD hold has already brought
      // the fingertips to a static equilibrium well within this window.
      // PUSH TEST: only index+middle matter (thumb is filtered and never
      // registers contact, so arrived[2] never latches).
      const double last_arrival_t =
          std::max(arrived_time[0], arrived_time[1]);
      if (arrived[0] && arrived[1] &&
          t >= last_arrival_t + FLAGS_handoff_settle_time) {
        std::cout << "[t=" << t
                  << "] index+middle settled → C3 handoff\n";

        // Read full state from the sim plant at this instant.
        const VectorXd x_contact =
            sim_plant.GetPositionsAndVelocities(plant_ctx);

        // Sync the lcs plant context and build the LCS linearized here.
        lcs_plant.SetPositionsAndVelocities(&lcs_ctx, x_contact);

        LCS lcs_init = LCSFactory::LinearizePlantToLCS(
            lcs_plant, lcs_ctx, *lcs_plant_ad, *lcs_ctx_ad,
            contact_pairs, lcs_opts, x_contact, VectorXd::Zero(n_u));

        // Contact-gap printout, informational only (not used to gate the
        // handoff — see handoff_settle_time above for why).
        const VectorXd eta0 =
            lcs_init.E()[0] * x_contact + lcs_init.c()[0];
        if (FLAGS_contact_model == "stewart_and_trinkle") {
          // Normal gaps φ live in the λ_n block [n_contacts, 2·n_contacts).
          const VectorXd phi_contact =
              eta0.segment(n_contacts, n_contacts);
          std::cout << "  contact gaps φ [index, middle, table] = "
                    << phi_contact.transpose()
                    << "  (<=0 means active, informational only)\n";
        } else {
          std::cout << "  (Anitescu: η is the cone-velocity constraint, not a "
                        "per-finger signed distance; gaps not shown)\n";
        }

        // True fingertip→cube distance via forward kinematics of the SIM
        // plant (independent of the LCS gap convention). Negative = fingertip
        // is inside the cube; this is the geometric ground truth to compare
        // against the LCS φ printed above — a mismatch reveals a reference
        // geometry / IK error rather than a planner error.
        {
          const RigidTransform<double> X_WC_cube =
              sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                            sim_plant.get_body(cube_body));
          const double h = cube_size / 2.0;
          std::cout << "  FK fingertip→cube dist [idx,mid,thu] (neg=inside): ";
          for (int i = 0; i < 3; ++i) {
            const Vector3d p = sim_plant
                .EvalBodyPoseInWorld(plant_ctx,
                                     sim_plant.get_body(tip_bodies[i]))
                .translation();
            const Vector3d pc = X_WC_cube.inverse() * p;
            const Vector3d q = pc.cwiseAbs() - Vector3d(h, h, h);
            const double d = (q.maxCoeff() <= 0.0)
                                 ? q.maxCoeff()
                                 : q.cwiseMax(0.0).norm();
            std::cout << d << " ";
          }
          std::cout << "\n";
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

        // Desired state: hand at q_contact for the grasping fingers.
        VectorXd q_des_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);
        for (int i = 0; i < 3; ++i)
          q_des_hand.segment(finger_start[i], 4) =
              q_contact.segment(finger_start[i], 4);

        VectorXd x_des = VectorXd::Zero(n_x);
        x_des.head(n_hand_q) = q_des_hand;
        x_des.segment(n_hand_q, 7) = q_cube0;
        // PUSH TEST: the objective is to move the cube +push_distance in Y
        // (toward where the thumb was), NOT to hold it at q_cube0. Cube pose
        // layout is [qw,qx,qy,qz,x,y,z], so y sits at n_hand_q+5. z stays
        // 0.58 (kinematically fixed — the "table"), x stays 0.
        x_des(n_hand_q + 5) = q_cube0(5) + FLAGS_push_distance;
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
          // PUSH TEST: 2 contacts (index, middle), both target alpha_m.
          const std::array<double, 2> alpha{FLAGS_alpha_m, FLAGS_alpha_m};
          std::vector<MatrixXd> W_lam(FLAGS_N,
                                      MatrixXd::Zero(n_lambda, n_lambda));
          std::vector<VectorXd> lambda_des(FLAGS_N,
                                           VectorXd::Zero(n_lambda));
          for (int k = 0; k < FLAGS_N; ++k) {
            for (int i = 0; i < n_finger_contacts; ++i) {
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

        // Diagnostic: track whether the linearization itself is stable
        // across consecutive relin calls on a near-identical state (during
        // cube_pinned, the cube's part of x_current is bit-for-bit frozen
        // and the hand is under a converged PD hold — so |A|/|B|/|D| should
        // barely move between prints if the linearization is well-behaved).
        if (!FLAGS_contact_force_log) {
          std::cout << "[t=" << t << "] relin  cube_pinned=" << cube_pinned
                    << "  |A|=" << lcs_new.A()[0].norm()
                    << "  |B|=" << lcs_new.B()[0].norm()
                    << "  |D|=" << lcs_new.D()[0].norm()
                    << "  |c|=" << lcs_new.c()[0].norm()
                    << "  |v_hand|=" << v_hand.norm()
                    << "  |v_hand|_inf=" << v_hand.cwiseAbs().maxCoeff()
                    << "\n";
        }

        std::vector<MatrixXd> B_sc = lcs_new.B();
        for (auto& Bk : B_sc) Bk *= s_u;
        lcs_new.set_B(B_sc);

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

        const std::vector<VectorXd> xplan     = c3->GetStateSolution();
        const std::vector<VectorXd> lam_plan  = c3->GetForceSolution();
        const std::vector<VectorXd> u_plan    = c3->GetInputSolution();

        if (FLAGS_contact_force_log) {
          // --contact_force_log suppresses both the C3 PLAN table and the
          // --plan_debug SOLVER DIAG output — see xplan_prev caching below.
        } else if (!FLAGS_plan_debug) {
          // ── Plan printout ───────────────────────────────────────────────
          // The ONLY thing printed per solve: for each of the 3 contacts, the
          // gap/closing variable γ, the normal contact force λ_n, and the
          // complementarity/gap slack η — across every horizon knot k=0..N.
          // Stewart-Trinkle λ layout: [γ(nc) | λ_n(nc) | β(...)], so γ = λ[0..nc-1]
          // and λ_n = λ[nc..2nc-1]. C3Plus z = [x, λ, u, η], so the η that relaxes
          // the normal-force complementarity sits at dual-δ offset
          // (n_x + n_lambda + n_u) + nc.
          // States span N+1 knots; forces (λ) span N knots. Bound by the minimum
          // so every indexed vector stays in range (otherwise lam_plan[N] is out
          // of bounds → segfault).
          const int n_knots = std::min(static_cast<int>(xplan.size()),
                                       static_cast<int>(lam_plan.size()));
          if (n_knots > 0) {
            const int nc = n_contacts;
            const LCS& lcs = c3->GetLCS();
            const auto& E = lcs.E();
            const auto& cc = lcs.c();
            std::vector<VectorXd> gap_plan(n_knots), lamn_plan(n_knots);
            for (int k = 0; k < n_knots; ++k) {
              // Contact gap φ = E·x + c, per-contact block (meters). The LCS
              // contact-gap lives at segment(n_contacts, n_contacts) of E·x + c.
              const VectorXd phi_full = E.at(k) * xplan[k] + cc.at(k);
              gap_plan[k]  = phi_full.segment(n_contacts, n_contacts);
              lamn_plan[k] = lam_plan[k].segment(nc, nc);
            }

            auto print_table = [&](const std::string& title,
                                   const std::vector<VectorXd>& sol,
                                   const std::vector<std::string>& labels) {
              const int nk = static_cast<int>(sol.size());
              if (nk == 0) return;
              const int nrows = std::min(static_cast<int>(sol[0].size()),
                                         static_cast<int>(labels.size()));
              std::cout << "\n-- " << title << " --\n";
              std::cout << std::setw(10) << "var";
              for (int k = 0; k < nk; ++k)
                std::cout << std::setw(12) << ("k" + std::to_string(k));
              std::cout << "\n";
              for (int r = 0; r < nrows; ++r) {
                std::cout << std::setw(10) << labels[r];
                for (int k = 0; k < nk; ++k)
                  std::cout << std::setw(12) << std::fixed
                            << std::setprecision(4) << sol[k](r);
                std::cout << "\n";
              }
            };

            std::cout << "\n=== C3 PLAN @ t=" << t << " s  (" << n_knots
                      << " knots, dt=" << FLAGS_c3_dt << " s) ===";

            std::vector<std::string> c_labels{"idx", "mid", "table"};

            print_table("CONTACT GAP φ (per contact, m)", gap_plan, c_labels);
            print_table("NORMAL FORCE λ_n (per contact, force)", lamn_plan, c_labels);

            // PUSH TEST: the objective is in Y (push toward +Y), so show
            // cube_y. State layout: cube xyz sits at [n_hand_q+4 (x),
            // n_hand_q+5 (y), n_hand_q+6 (z)]. Target y = start + push_distance.
            std::vector<VectorXd> cubey_plan(xplan.size());
            for (size_t k = 0; k < xplan.size(); ++k)
              cubey_plan[k] = xplan[k].segment(n_hand_q + 5, 1);
            print_table("CUBE Y POSITION (state knots, target="
                            + std::to_string(q_cube0(5) + FLAGS_push_distance)
                            + ")",
                        cubey_plan, {"cube_y"});

            // ── Cost breakdown per knot ──────────────────────────────────
            // Splits the objective C3 is actually minimizing into its terms,
            // per horizon knot, so you can see WHERE the cost lives.
            //   state-tracking (Q):  (x_k − x_des_k)ᵀ Q (x_k − x_des_k),
            //     split by state block — hand joint positions (k_hold),
            //     cube pose (w_cube), hand joint velocities (w_vel), cube
            //     velocity (w_cube). Q is diagonal here, so each block is
            //     Σ_{i∈block} Q(i,i)·dx_i².
            //   input (R):  u_kᵀ R u_k. R = s_u²·w_R, u is the scaled input,
            //     so this reads in PHYSICAL torque-cost units (w_R·τᵀτ).
            //   TOTAL = sum of the above.
            // NOT included: the ADMM augmented-Lagrangian G/U terms (solver
            // internals, not the physical objective) and the w_lambda
            // force-tracking cost (a separate cost, 0 unless --w_lambda>0).
            const auto& costs = c3->GetCostMatrices();
            const std::vector<VectorXd> x_des_plan = c3->GetDesiredState();
            const int nk_state = static_cast<int>(xplan.size());
            std::vector<VectorXd> cost_plan(nk_state);
            for (int k = 0; k < nk_state; ++k) {
              VectorXd row = VectorXd::Zero(6);  // handq,cubepose,handv,cubev,input,total
              if (k < static_cast<int>(costs.Q.size()) &&
                  k < static_cast<int>(x_des_plan.size())) {
                const VectorXd dx = xplan[k] - x_des_plan[k];
                const VectorXd qd = costs.Q[k].diagonal();
                for (int i = 0; i < n_hand_q; ++i)
                  row(0) += qd(i) * dx(i) * dx(i);
                for (int i = n_hand_q; i < n_pos; ++i)
                  row(1) += qd(i) * dx(i) * dx(i);
                for (int i = n_pos; i < n_pos + n_hand_v; ++i)
                  row(2) += qd(i) * dx(i) * dx(i);
                for (int i = n_pos + n_hand_v; i < n_x; ++i)
                  row(3) += qd(i) * dx(i) * dx(i);
              }
              if (k < static_cast<int>(u_plan.size()) &&
                  k < static_cast<int>(costs.R.size())) {
                row(4) = u_plan[k].dot(costs.R[k] * u_plan[k]);
              }
              row(5) = row(0) + row(1) + row(2) + row(3) + row(4);
              cost_plan[k] = row;
            }
            print_table(
                "COST PER KNOT (Q state-tracking by block + R input)",
                cost_plan,
                {"hand_q", "cube_pose", "hand_v", "cube_v", "input", "TOTAL"});
          }
        } else {
          // ── Solver-quality diagnostics ──────────────────────────────────
          std::cout << "\n=== SOLVER DIAG @ t=" << t << " s ===\n";

          // (1) LINEARIZATION ACCURACY: compare the PREVIOUS solve's
          // knot-ahead state prediction to the state actually measured now.
          // knot_ahead is how many c3_dt's have elapsed since that solve —
          // normally 1 (c3_period_steps == relin_period_steps == c3_dt), but
          // computed generally in case the cadence flags are unequal.
          if (!xplan_prev.empty() && t_prev_solve >= 0.0) {
            const int knot_ahead = std::clamp(
                static_cast<int>(std::lround((t - t_prev_solve) /
                                             FLAGS_c3_dt)),
                1, static_cast<int>(xplan_prev.size()) - 1);
            const VectorXd& x_pred = xplan_prev[knot_ahead];
            const VectorXd dx = x_current - x_pred;
            const double cube_z_err =
                x_current(n_hand_q + 6) - x_pred(n_hand_q + 6);
            const double cube_pos_err =
                (x_current.segment<3>(n_hand_q + 4) -
                 x_pred.segment<3>(n_hand_q + 4)).norm();
            std::cout << "  [linearization] knot_ahead=" << knot_ahead
                      << "  full-state ‖error‖=" << dx.norm()
                      << "  cube_pos ‖error‖=" << cube_pos_err
                      << "  cube_z error=" << cube_z_err << " m\n";
          } else {
            std::cout << "  [linearization] no previous plan yet "
                         "(first solve this phase)\n";
          }

          // Unit note: GetLCS()'s D/E/F/H/c are baked with C3's internal
          // λ-scaling (C3Options::scale_lcs → LCS::ScaleComplementarityDynamics),
          // and GetForceSolution()/GetFullSolution()'s λ block is the
          // RESCALED-to-physical value (λ_physical = AnDn_ · λ_internal —
          // see c3.cc: `lambda_sol_->at(i) *= AnDn_`). Combining physical λ
          // with the scaled matrices double-counts the scale factor, so we
          // divide back by AnDn_ below. GetDualDeltaSolution() (δ) is NEVER
          // rescaled and is already in the same internal units as GetLCS() —
          // no correction needed there.
          const double an_dn = c3->GetLambdaScaling();

          // (2) DYNAMICS RESIDUAL: for the SOLVED (z-side) trajectory, how
          // well does x_{k+1} match A x_k + B u_k + D λ_k + d? This is an
          // equality constraint inside C3's QP, so it should be ~0 for a
          // converged solve; a large residual means the final QP didn't
          // actually satisfy its own constraint (non-convergence / failure).
          {
            const LCS& lcs = c3->GetLCS();
            const auto& A = lcs.A();
            const auto& B = lcs.B();
            const auto& D = lcs.D();
            const auto& d = lcs.d();
            const int n_knots = std::min(
                {static_cast<int>(xplan.size()) - 1,
                 static_cast<int>(lam_plan.size()),
                 static_cast<int>(u_plan.size())});

            double dyn_resid_max = 0.0, dyn_resid_sum = 0.0;
            for (int k = 0; k < n_knots; ++k) {
              const VectorXd x_next_pred =
                  A.at(k) * xplan[k] + B.at(k) * u_plan[k] +
                  D.at(k) * (lam_plan[k] / an_dn) + d.at(k);
              const double dyn_resid = (xplan[k + 1] - x_next_pred).norm();
              dyn_resid_max = std::max(dyn_resid_max, dyn_resid);
              dyn_resid_sum += dyn_resid;
            }
            std::cout << "  [dynamics residual] max‖x_{k+1}−f(x_k,u_k,λ_k)‖="
                      << dyn_resid_max << "  mean="
                      << (n_knots > 0 ? dyn_resid_sum / n_knots : 0.0) << "\n";
          }

          // (3) COMPLEMENTARITY RESIDUAL, on δ (the projection-step copy —
          // the copy actually meant to satisfy 0 ≤ λ ⊥ η ≥ 0; the z/QP-side
          // λ used above is NOT expected to be complementary by
          // construction — see GetForceSolution()'s doc comment). δ_k is a
          // GetZSize()-vector laid out [x | λ | u], same as z_sol_.
          //
          // (4) ADMM CONSENSUS RESIDUAL ‖z−δ‖: the actual "has ADMM
          // converged" signal — z enforces dynamics, δ enforces
          // complementarity, and ADMM is trying to drive them together.
          // Large values mean the two copies still disagree substantially.
          {
            const LCS& lcs = c3->GetLCS();
            const auto& E = lcs.E();
            const auto& F = lcs.F();
            const auto& H = lcs.H();
            const auto& cc = lcs.c();
            const std::vector<VectorXd> delta = c3->GetDualDeltaSolution();
            const std::vector<VectorXd> z = c3->GetFullSolution();
            const int n_knots = std::min(
                {static_cast<int>(delta.size()), static_cast<int>(z.size()),
                 static_cast<int>(E.size())});

            double lam_min = std::numeric_limits<double>::infinity();
            double eta_min = std::numeric_limits<double>::infinity();
            double compl_gap_max = 0.0;
            double consensus_max = 0.0, consensus_sum = 0.0;
            for (int k = 0; k < n_knots; ++k) {
              const VectorXd x_d = delta[k].segment(0, n_x);
              const VectorXd lam_d = delta[k].segment(n_x, n_lambda);
              const VectorXd u_d = delta[k].segment(n_x + n_lambda, n_u);
              const VectorXd eta =
                  E.at(k) * x_d + F.at(k) * lam_d + H.at(k) * u_d + cc.at(k);
              lam_min = std::min(lam_min, lam_d.minCoeff());
              eta_min = std::min(eta_min, eta.minCoeff());
              compl_gap_max = std::max(
                  compl_gap_max, lam_d.cwiseProduct(eta).cwiseAbs().maxCoeff());

              // Rescale z's λ block back to internal units to match δ before
              // differencing (see unit note above).
              VectorXd z_internal = z[k];
              z_internal.segment(n_x, n_lambda) /= an_dn;
              const double consensus = (z_internal - delta[k]).norm();
              consensus_max = std::max(consensus_max, consensus);
              consensus_sum += consensus;
            }
            std::cout << "  [complementarity(δ)]  min(λ)=" << lam_min
                      << "  min(η)=" << eta_min
                      << "  max|λ⊙η|=" << compl_gap_max
                      << "  (want: both mins ≥ ~0, gap ≈ 0)\n";
            std::cout << "  [ADMM consensus ‖z−δ‖]  max=" << consensus_max
                      << "  mean="
                      << (n_knots > 0 ? consensus_sum / n_knots : 0.0)
                      << "  (want: ≈ 0 — large means ADMM hasn't converged)\n";
          }
        }

        // Cache this solve's plan for next solve's linearization-accuracy
        // check, regardless of which printout mode is active.
        xplan_prev = xplan;
        t_prev_solve = t;
      }

      if (cube_pinned) {
        // Warm-up period: cube is still pinned, so C3's cost gradient on the
        // cube is near zero and it finds near-zero torques as "optimal". This
        // causes fingers to drift away from the cube. Instead, use PD to hold
        // fingers at q_contact (3 mm inside the faces) while C3 runs in the
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

        // Diagnostic: is the commanded torque actually saturating at
        // ±tau_max before the clamp below? If so, kp/kd values stop
        // mattering — the delivered torque is just the bound, every tick,
        // which looks exactly like a gain-independent limit cycle.
        if (do_relin && !FLAGS_contact_force_log) {
          const int n_sat = (tau_hand.array().abs() >= FLAGS_tau_max - 1e-6)
                                 .count();
          std::cout << "[t=" << t << "] tau(unclamped)  |tau|=" << tau_hand.norm()
                    << "  max|tau_i|=" << tau_hand.cwiseAbs().maxCoeff()
                    << "  saturated_joints=" << n_sat << "/" << n_hand
                    << "  (tau_max=" << FLAGS_tau_max << ")\n";
        }
      } else {
        // Table phase — task-space PD + Jacobian-transpose push executor.
        // PUSH TEST: only index+middle apply force (2 contacts). Desired
        // per-fingertip normal force = C3's projected λ, floored so the push
        // survives the contact-gate hover band.
        const VectorXd lam_proj =
            c3->GetDualDeltaSolution()[0].segment(n_x, n_lambda);
        const std::array<double, 2> floor{FLAGS_force_floor,
                                          FLAGS_force_floor};
        std::array<double, 2> fn_des{};
        for (int i = 0; i < n_finger_contacts; ++i) {
          double fn = 0.0;
          for (int idx : normal_groups[i]) fn += lam_proj(idx);
          fn_des[i] = std::max(std::max(fn, 0.0), floor[i]);
        }

        // Fingertip position targets. --fk_target: forward-kinematics of C3's
        // planned next hand config q1 (where the plan says the fingertips go
        // next). Otherwise: fixed grasp points on the upright reference cube
        // (the geometric hold).
        VectorXd p_des_all(9);
        if (FLAGS_fk_target) {
          const std::vector<VectorXd> xplan = c3->GetStateSolution();
          const VectorXd q1 = xplan.size() > 1
                                  ? VectorXd(xplan[1].head(n_hand_q))
                                  : sim_plant.GetPositions(plant_ctx, sim_allegro);
          sim_plant.SetPositions(fk_ctx.get(), sim_allegro, q1);
          for (int i = 0; i < n_finger_contacts; ++i)
            p_des_all.segment<3>(3 * i) =
                sim_plant
                    .EvalBodyPoseInWorld(*fk_ctx,
                                         sim_plant.get_body(tip_bodies[i]))
                    .translation();
        } else {
          p_des_all = GetGraspPositions(X_WC0, cube_size - 0.006, a, b, c_off);
        }

        // Task-space PD + Jacobian-transpose grip, no QP:
        //   τ = τ_g + Σ Jᵢᵀ[ Kp(p_des−p) − Kd·ṗ + fₙ·n̂ ],
        // plus C3's raw input solution as feedforward.
        const VectorXd tau_g = sim_plant.GetVelocitiesFromArray(
            sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));
        const RotationMatrix<double> R_WC =
            CubePoseFromPositions(sim_plant.GetPositions(plant_ctx, sim_cube))
                .rotation();
        // PUSH TEST: index+middle both press +Y (into the -Y face), pushing
        // the cube toward +Y (where the thumb was). Thumb gets only gravity
        // comp (tau_g), no active push, and is collision-filtered.
        const std::array<Vector3d, 2> press_C{
            Vector3d(0, 1, 0), Vector3d(0, 1, 0)};
        VectorXd tau = tau_g;
        for (int i = 0; i < n_finger_contacts; ++i) {
          Eigen::MatrixXd J(3, sim_plant.num_velocities());
          sim_plant.CalcJacobianTranslationalVelocity(
              plant_ctx, drake::multibody::JacobianWrtVariable::kV,
              sim_plant.get_body(tip_bodies[i]).body_frame(), Vector3d::Zero(),
              sim_plant.world_frame(), sim_plant.world_frame(), &J);
          const Eigen::MatrixXd Jh = J.leftCols(n_hand_v);
          const Vector3d p_i =
              sim_plant
                  .EvalBodyPoseInWorld(plant_ctx,
                                       sim_plant.get_body(tip_bodies[i]))
                  .translation();
          const Vector3d pdot = Jh * v_hand;
          const Vector3d f_task =
              FLAGS_task_kp * (p_des_all.segment<3>(3 * i) - p_i) -
              FLAGS_task_kd * pdot + fn_des[i] * (R_WC * press_C[i]);
          tau += Jh.transpose() * f_task;
        }
        tau_hand = tau + s_u * c3->GetInputSolution()[0];
      }
    }

    // Clamp only in kC3: the realization/C3 output must never exceed the
    // physical torque budget (an unclamped 321 Nm spike at pin release wrecked
    // the grasp). The reach phase stays unclamped — it needs 30-80 Nm peaks to
    // overcome the initial SAP transient and track the spline.
    if (phase == kC3) {
      tau_hand = tau_hand.cwiseMin(FLAGS_tau_max).cwiseMax(-FLAGS_tau_max);
    }

    // LCM telemetry: the actually-applied torque (post-clamp), every
    // control step, both phases — see GRASP_TAU wiring above.
    if (tau_input != nullptr) {
      tau_input->GetMutableVectorData<double>()->SetFromVector(tau_hand);
    }

    act_fixed.GetMutableVectorData<double>()->SetFromVector(tau_hand);

    simulator.AdvanceTo(t);

    // Kinematic pin: FULL pin (cube frozen at q_cube0) during reach + a 0.5 s
    // warm-up so the fingers seat and C3 converges. After that it's RELEASED —
    // the cube rests on the real frictional table (no kinematic constraint),
    // and index+middle physically push it in +Y, resisted by table friction.
    if (cube_pinned && phase == kC3 && t >= handoff_t + 0.5) {
      cube_pinned = false;
      std::cout << "[t=" << t << "] warm-up done → cube released onto table\n";
    }
    if (cube_pinned) {
      sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
      sim_plant.SetVelocities(&plant_ctx, sim_cube, VectorXd::Zero(6));
    }

    // Update Meshcat cube frame and grasp markers.
    const RigidTransform<double> X_WC_now = CubePoseFromPositions(
        sim_plant.GetPositions(plant_ctx, sim_cube));
    update_markers(X_WC_now);
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
