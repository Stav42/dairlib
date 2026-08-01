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
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_forces.h>
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
#include "c3/multibody/geom_geom_collider.h"
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
using c3::multibody::GeomGeomCollider;
using c3::multibody::LCSFactory;

// ── Reach-phase flags ────────────────────────────────────────────────────────
DEFINE_double(kp, 70.0, "PD proportional gain (Nm/rad) for the reach phase.");
DEFINE_double(kd, 0.0,
              "PD derivative gain (Nm·s/rad) for the reach phase.");
DEFINE_double(tau_max, 1000.0,
              "Per-joint torque box bound (Nm). Applied as INPUT constraints "
              "inside the C3 QP to keep it bounded (prevents DualInfeasible).");
DEFINE_double(contact_force_thresh, 0.02,
              "Per-finger contact force threshold to count as touching (N).");
DEFINE_double(penetration_index_middle, -0.0,
              "q_contact target: how far the TRUE fingertip surface (see "
              "--tip_surface_offset_z) sits inside the -Y cube face for the "
              "index and middle fingers (m).");
DEFINE_double(penetration_thumb, 0.005,
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
DEFINE_bool(track_cube_contact, false,
            "Re-solve the 3-point grasp IK against the CURRENT cube pose "
            "every relin (once unpinned), so the C3 cost's hand-q reference "
            "(k_hold) AND the OSC anchor track the cube as it moves, instead "
            "of pointing at the static t=0 q_contact. Default false keeps the "
            "static q_contact (A/B toggle).");
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
DEFINE_double(k_hold, 0.0,
              "Q cost weight on hand-joint deviations from q_contact. Keeps "
              "fingertips on the cube faces inside the ADMM solve.");
DEFINE_double(w_cube, 10000.0,
              "Q cost weight on cube POSE only (world-frame anchor). This is "
              "the load-bearing term: C3 must keep the cube here against "
              "gravity, which forces Σμλ_n ≥ mg. Was 1000 (shared with cube "
              "velocity); now 10× higher and pose-only, since the cost "
              "breakdown showed the velocity term (below) dominating and "
              "drowning out the position anchor.");
DEFINE_double(w_cube_vel, 0.0,
              "Q cost weight on cube VELOCITY (split out from w_cube). Was "
              "folded into w_cube at 1000; now 100× lower (10) so it no "
              "longer dominates the objective over the pose anchor.");
DEFINE_double(w_vel, 0.0,
              "Q cost weight on hand joint velocities (light damping).");
DEFINE_double(w_R, 0.01, "R cost weight on joint torques.");
DEFINE_double(w_G, 0.5, "ADMM augmented-Lagrangian G weight.");
DEFINE_double(w_U, 1.0, "ADMM augmented-Lagrangian U weight.");
DEFINE_double(w_lambda, 0.0,
              "Force-reference cost on the three normal contact forces "
              "λ_n[3,4,5]. Regularises the squeeze nullspace (§11.5) and gives "
              "the solve a physical grip target. 0 = disabled (default).");
DEFINE_double(alpha_m, 1.0,
              "Per-finger normal grip-force target (N) for the λ_n reference. "
              "Thumb target = 2·alpha_m (force closure vs index+middle).");
DEFINE_string(cube_motion_mode, "none",
              "Cube reference-pose motion after pin release (generalizes the "
              "old z-only --cube_bob_amp sine to a full 6-DOF profile):\n"
              "  none - static hold at X_WC0 (default).\n"
              "  step - min-jerk ramp from X_WC0 to X_WC0 composed with the "
              "--cube_move_{dx,dy,dz,roll,pitch,yaw} offset, over "
              "--cube_move_duration seconds, then holds there.\n"
              "  sine - the same offset oscillates sinusoidally about X_WC0 "
              "at --cube_move_period, its amplitude ramped in over "
              "--cube_move_duration so the reference starts at rest instead "
              "of stepping the velocity the instant the pin releases.");
DEFINE_double(cube_move_dx, 0.0,
              "Step target / sine amplitude, world-frame x translation (m).");
DEFINE_double(cube_move_dy, 0.0,
              "Step target / sine amplitude, world-frame y translation (m).");
DEFINE_double(cube_move_dz, 0.0,
              "Step target / sine amplitude, world-frame z translation (m).");
DEFINE_double(cube_move_roll, 0.0,
              "Step target / sine amplitude, roll (rad) about X_WC0's own "
              "x-axis (rotation is about the cube's own center, not the "
              "world origin).");
DEFINE_double(cube_move_pitch, 0.0,
              "Step target / sine amplitude, pitch (rad) about X_WC0's own "
              "y-axis.");
DEFINE_double(cube_move_yaw, 0.0,
              "Step target / sine amplitude, yaw (rad) about X_WC0's own "
              "z-axis.");
DEFINE_double(cube_move_period, 1.0, "sine mode: oscillation period (s).");
DEFINE_double(cube_move_duration, 0.5,
              "step: min-jerk ramp time to the target (s). sine: amplitude "
              "ramp-in time (s) — avoids a velocity step at release.");
DEFINE_double(cube_ik_lead_pos_max, 0.05,
              "Safety clamp (m): the moving-contact IK target pose is capped "
              "to this far from the CURRENTLY MEASURED cube pose, so if the "
              "cube lags the reference the fingers don't chase an "
              "unreachable target. Generous default — only binds if the cube "
              "falls far behind a fast/large commanded motion.");
DEFINE_double(cube_ik_lead_rot_max, 0.3,
              "Safety clamp (rad): same as --cube_ik_lead_pos_max for the "
              "rotational part of the IK lead.");
DEFINE_bool(show_cube_target, true,
            "Draw a translucent CYAN 'ghost' cube in Meshcat at the raw "
            "(pre-clamp) --cube_motion_mode target pose, updated every "
            "control step post-handoff. Lets you see tracking quality — "
            "sag, lag, drift — directly: compare the real (opaque) cube "
            "against the ghost. Set false for a clean recording.");
DEFINE_bool(show_cube_start, true,
            "Draw a translucent RED 'ghost' cube in Meshcat at the fixed "
            "starting/nominal pose X_WC0 — the pose every --cube_move_* "
            "offset is measured from. Static (set once, never updated), "
            "unlike --show_cube_target's live ghost, so it stays put as a "
            "constant visual anchor for how far the cube has moved.");
// ── Low-level realization layer ──────────────────────────────────────────────
// C3 PLANS (low rate); a task-space PD + Jacobian-transpose grip EXECUTES (high
// rate, every control tick): τ = τ_g + Σ Jᵢᵀ[Kp(p_des−p) − Kd·ṗ + fₙ·n̂], with
// C3's raw input solution added as feedforward. p_des is a fingertip position
// target (fixed grasp points, or C3's planned config — see --fk_target); fₙ is
// the desired per-fingertip normal force (C3's projected λ, optionally floored
// — see --force_floor).
DEFINE_bool(lambda_map_debug, false,
            "Print the '=== D: cube xyz <- lambda ===' breakdown (physical D "
            "matrix restricted to cube rows, per contact) after every solve. "
            "Off by default — verbose, rarely needed once the D-map's "
            "structure (which columns move cube x/y/z) is already known.");
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
DEFINE_string(exec_mode, "task_space",
              "Low-level controller that turns the C3 plan into applied "
              "torque, selectable so more can be added later:\n"
              "  direct     - Architecture A: pure input playback. "
              "tau = s_u*u0, receding-horizon. No tau_g (gravity is in the "
              "LCS), no PD, no task-space grip. The plan IS the controller. "
              "Runs in both the pinned warm-up and post-release.\n"
              "  task_space - (default) legacy hybrid: PD-to-q_contact while "
              "pinned, Jacobian-transpose grip + task-space PD + s_u*u0 "
              "feedforward after release.\n"
              "  osc        - Architecture B: joint-space inverse-dynamics "
              "tracking of the plan's next hand config (exact nonlinear M/C/G "
              "via CalcInverseDynamics — no LCS-linearization trust needed) "
              "PLUS a task-space feedforward normal force at each fingertip "
              "equal to C3's planned lambda_n (Jacobian-transpose). Ignores "
              "u0 entirely — realizes (x1, lambda0) on the exact plant "
              "instead of replaying the LCS's own input. The position target "
              "is q_contact_live (live grasp-IK against the desired cube "
              "pose, Mods 2-4) if --track_cube_contact, else C3's own "
              "knot-1 hand config (x1).");
DEFINE_double(osc_kp, 300.0,
              "OSC joint-space position gain (Nm/rad), applied directly in "
              "torque space as kp*(q_des-q) — NOT an inverse-dynamics "
              "acceleration gain despite the units its name suggests; the "
              "current exec_mode=osc law never calls CalcInverseDynamics.");
DEFINE_double(osc_kd, 15.0,
              "OSC joint-space damping gain (Nm*s/rad), applied directly in "
              "torque space as -kd*qdot (implicit qd_des=0 — pure damping, "
              "same convention as the reach-phase --kd). Previously defined "
              "but unused: exec_mode=osc was simplified to a bare P law after "
              "an earlier inverse-dynamics/kd variant injected energy at pin "
              "release and kicked the cube loose. This direct-torque form "
              "skips M(q) entirely so it's architecturally gentler than that "
              "one, but is still untested — A/B against --osc_kd=0.");
DEFINE_bool(exec_grav_comp, false,
            "Post-release C3 executor: add explicit gravity compensation "
            "(tau_g) on top of s_u*u0. Default false: gravity is already "
            "baked into the LCS, so C3's u0 carries it — adding tau_g here "
            "double-counts gravity. Toggle true to A/B the old behavior. "
            "(Does NOT affect the pinned-phase PD, which has no u0 and needs "
            "its own tau_g.)");
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
DEFINE_bool(warm_start_admm, false,
            "Carry the ADMM consensus solution (delta) across consecutive C3 "
            "solves instead of cold-starting each from zero. Anchors each "
            "solve to the previous one — a low-pass on the plan itself — which "
            "kills solve-to-solve churn where cold starts fall into different "
            "local optima of the non-convex complementarity problem. Persists "
            "across relinearizations (c3->UpdateLCS keeps the object). Lets "
            "you cut --admm_iter once the plan stops re-deriving from scratch. "
            "Distinct from --warm_start (that's the within-solve QP guess).");
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
DEFINE_double(rho_scale, 1.2,
              "Per-ADMM-iteration growth of the augmented-Lagrangian weight "
              "(G *= rho_scale each iteration). Movement effectively stops "
              "once G has grown ~1e3x, so useful iterations ~= "
              "log(1000)/log(rho_scale). Match to --admm_iter: 1.2 -> ~40, "
              "1.08 -> ~90, 1.04 -> ~175.");
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
  systems::C3StateSender* idx_sender = nullptr;
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

    // Fourth channel: index-finger joint tracking (GRASP_IDX_TRACK). 12
    // slots — the 4 actual index joint angles, then the C3 plan's knot-0 and
    // knot-1 index joint angles. The OSC PD chases the knot-1 config
    // (q_des = GetStateSolution()[1]); knot-0 is the plan's initial condition
    // (≈ the current state). Streaming all three lets a plotter overlay
    // actual vs. the planned setpoint per joint to see if the PD is tracking.
    std::vector<std::string> idx_names{
        "act_q0",   "act_q1",   "act_q2",   "act_q3",
        "plan0_q0", "plan0_q1", "plan0_q2", "plan0_q3",
        "plan1_q0", "plan1_q1", "plan1_q2", "plan1_q3"};
    idx_sender = sim_builder.AddSystem<systems::C3StateSender>(12, idx_names);
    idx_sender->set_name("idx_sender");  // unique name (see tau_sender note)
    auto* idx_pub = sim_builder.AddSystem(
        LcmPublisherSystem::Make<dairlib::lcmt_c3_state>(
            "GRASP_IDX_TRACK", lcm_iface, 1.0 / FLAGS_lcm_publish_hz));
    sim_builder.Connect(idx_sender->get_output_port_target_c3_state(),
                        idx_pub->get_input_port());
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

  // idx_input: index-finger tracking, updated every control step (see the
  // GRASP_IDX_TRACK wiring above and the loop update below).
  drake::systems::FixedInputPortValue* idx_input = nullptr;
  if (idx_sender != nullptr) {
    auto& idx_sender_ctx =
        sim_diagram->GetMutableSubsystemContext(*idx_sender, &sim_ctx);
    idx_input = &idx_sender->get_input_port_target_state().FixValue(
        &idx_sender_ctx, VectorXd::Zero(12));
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

  // Translucent "ghost" cube at the --cube_motion_mode commanded target pose
  // (see the kC3-phase update below) — makes tracking quality (sag, lag,
  // drift) visible directly against the real, opaque cube. Slightly
  // inflated so its faces don't z-fight the real cube's when the two
  // coincide (perfect tracking, or before any motion is commanded).
  if (FLAGS_show_cube_target) {
    const drake::geometry::Rgba kGhostCyan(0.0, 0.9, 1.0, 0.35);
    meshcat->SetObject(
        "/cube_target",
        drake::geometry::Box(cube_size + 0.002, cube_size + 0.002,
                             cube_size + 0.002),
        kGhostCyan);
    meshcat->SetTransform("/cube_target", X_WC0);
  }

  // Static red start-pose ghost, set once — X_WC0 never changes during a
  // run, so (unlike /cube_target) this needs no per-tick update anywhere.
  // Inflated further than /cube_target (+4mm vs +2mm) so the two ghosts
  // render as distinct concentric shells around the real cube instead of
  // z-fighting when both sit at X_WC0 (e.g. before any motion starts).
  if (FLAGS_show_cube_start) {
    const drake::geometry::Rgba kGhostRed(1.0, 0.0, 0.0, 0.35);
    meshcat->SetObject(
        "/cube_start",
        drake::geometry::Box(cube_size + 0.004, cube_size + 0.004,
                             cube_size + 0.004),
        kGhostRed);
    meshcat->SetTransform("/cube_start", X_WC0);
  }

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

  // Apply the user-facing physical normal-force reference in C3's current
  // internal lambda units. C3 may change AnDn_ whenever UpdateLCS() rescales a
  // relinearized LCS, so this must be called after construction and every
  // subsequent LCS update.
  auto update_force_tracking = [&]() {
    if (!c3 || FLAGS_w_lambda <= 0.0) return;

    const std::array<double, 3> alpha{FLAGS_alpha_m, FLAGS_alpha_m,
                                      2.0 * FLAGS_alpha_m};
    const double lambda_scale = c3->GetLambdaScaling();
    std::vector<MatrixXd> W_lam(
        FLAGS_N, MatrixXd::Zero(n_lambda, n_lambda));
    std::vector<VectorXd> lambda_des(
        FLAGS_N, VectorXd::Zero(n_lambda));
    for (int k = 0; k < FLAGS_N; ++k) {
      for (int i = 0; i < 3; ++i) {
        const std::vector<int>& g = normal_groups[i];
        for (int a : g) {
          for (int b : g) {
            // w*(lambda_physical-alpha)^2 expressed in internal lambda.
            W_lam[k](a, b) =
                FLAGS_w_lambda * lambda_scale * lambda_scale;
          }
          lambda_des[k](a) =
              alpha[i] / (static_cast<double>(g.size()) * lambda_scale);
        }
      }
    }
    c3->SetForceTrackingWeight(W_lam);
    c3->UpdateForceTarget(lambda_des);
  };

  // Base C3 state target (built at handoff) and the cube's nominal z. The C3
  // phase overwrites the cube pose reference per --cube_motion_mode.
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

  // --track_cube_contact: the hand-q reference the cost pulls toward (k_hold)
  // and the OSC anchor pulls toward. Initialized to the static t=0 q_contact;
  // if tracking is enabled, re-solved every relin against the live cube pose
  // (below) so it follows the cube. With tracking off it stays == q_contact,
  // so every consumer can read q_contact_live unconditionally.
  VectorXd q_contact_live = q_contact;
  // Horizon-end IK solve (t_ref_now + N*c3_dt) — Mod 3 linearly interpolates
  // the plan's per-knot hand-q reference between q_contact_live (k=0) and
  // this (k=N) instead of freezing the whole horizon at a single instant.
  VectorXd q_contact_live_end = q_contact;

  // Dedicated scratch context for the moving-contact IK. Standalone so the
  // solve never disturbs the live sim context (plant_ctx) or the LCS context.
  auto ik_ctx = sim_plant.CreateDefaultContext();

  // Re-solve the 3-point grasp IK so the three fingertips land on the current
  // cube faces (same per-finger penetration + tip-surface offset as the t=0
  // q_contact). Warm-started from the last solution for speed and to keep the
  // hand config from jumping between IK branches. Returns hand-only joints.
  auto resolve_contact_ik =
      [&](const RigidTransform<double>& X_WC) -> VectorXd {
    const double h = cube_size / 2.0;
    VectorXd targets(9);
    targets << X_WC * Vector3d(a, -(h - FLAGS_penetration_index_middle), 0),
        X_WC * Vector3d(b, -(h - FLAGS_penetration_index_middle), 0),
        X_WC * Vector3d(c_off, h - FLAGS_penetration_thumb, 0);
    sim_plant.SetPositions(ik_ctx.get(), sim_allegro, q_contact_live);
    const VectorXd q_full =
        SolveGraspIK(sim_plant, ik_ctx.get(), targets, tip_surface_pt);
    sim_plant.SetPositions(ik_ctx.get(), q_full);
    return sim_plant.GetPositions(*ik_ctx, sim_allegro);
  };

  // ── Cube reference-pose generator (--cube_motion_mode) ──────────────────
  // alpha(t_ref) in [0,1] scales BOTH the translation and rotation offsets so
  // they move in lockstep; dalpha/dt rides along for the translational-
  // velocity feedforward below. "step" is a min-jerk ramp (zero velocity AND
  // zero acceleration at both ends) from 0 to 1 that then holds; "sine" is
  // the same min-jerk ramp used as an AMPLITUDE envelope on a sinusoid, so
  // oscillation eases in from rest instead of stepping velocity the instant
  // the pin releases.
  auto motion_alpha = [&](double t_ref) -> std::pair<double, double> {
    const double T = std::max(1e-6, FLAGS_cube_move_duration);
    const double s = std::clamp(t_ref / T, 0.0, 1.0);
    const double ramp = 10.0 * s * s * s - 15.0 * s * s * s * s +
                        6.0 * s * s * s * s * s;
    const double dramp_dt =
        (t_ref <= 0.0 || t_ref >= T)
            ? 0.0
            : (30.0 * s * s - 60.0 * s * s * s + 30.0 * s * s * s * s) / T;
    if (FLAGS_cube_motion_mode == "step") {
      return {ramp, dramp_dt};
    } else if (FLAGS_cube_motion_mode == "sine") {
      const double w = 2.0 * M_PI / std::max(1e-6, FLAGS_cube_move_period);
      const double sn = std::sin(w * t_ref), cs = std::cos(w * t_ref);
      return {ramp * sn, dramp_dt * sn + ramp * w * cs};
    }
    return {0.0, 0.0};  // "none"
  };

  // Desired cube pose + its translational velocity, t_ref seconds after pin
  // release. Rotation is composed about the CUBE's own nominal frame
  // (X_WC0's axes) about its own center, then the whole cube (rotated) is
  // translated in world frame — so --cube_move_roll/pitch/yaw spin the cube
  // in place and --cube_move_dx/dy/dz then carries it there.
  // NOTE: only translational velocity is populated (matching the convention
  // this file already verified for the old z-only bob term, below); the
  // angular-velocity feedforward is left at zero. Drake's floating-base
  // generalized-velocity frame convention for the rotational block was not
  // re-derived here, and getting it wrong would inject a wrong-frame signal
  // into the cost — at the small, slow angles this flag targets, the
  // resulting tracking lag is negligible against --w_cube_vel's light
  // weight, and it costs nothing extra to add later if it matters.
  auto cube_target_pose =
      [&](double t_ref) -> std::pair<RigidTransform<double>, Vector3d> {
    const std::pair<double, double> ad = motion_alpha(t_ref);
    const double alpha = ad.first, dalpha = ad.second;
    const Vector3d delta_p(FLAGS_cube_move_dx, FLAGS_cube_move_dy,
                           FLAGS_cube_move_dz);
    const RotationMatrix<double> R_delta(
        drake::math::RollPitchYaw<double>(alpha * FLAGS_cube_move_roll,
                                          alpha * FLAGS_cube_move_pitch,
                                          alpha * FLAGS_cube_move_yaw));
    const RigidTransform<double> X_WC_des =
        RigidTransform<double>(alpha * delta_p) * X_WC0 *
        RigidTransform<double>(R_delta);
    return {X_WC_des, Vector3d(dalpha * delta_p)};
  };

  // Mod 2 safety clamp: cap the IK target pose to within
  // --cube_ik_lead_{pos,rot}_max of the CURRENTLY MEASURED cube pose, so a
  // cube that lags a fast/large commanded motion doesn't yank the fingers
  // toward a target the physical grasp can't reach yet.
  auto clamp_ik_lead =
      [&](const RigidTransform<double>& X_des,
          const RigidTransform<double>& X_meas) -> RigidTransform<double> {
    const Vector3d dp = X_des.translation() - X_meas.translation();
    const double dp_norm = dp.norm();
    const Vector3d p_clamped =
        (dp_norm > FLAGS_cube_ik_lead_pos_max)
            ? Vector3d(X_meas.translation() +
                       dp * (FLAGS_cube_ik_lead_pos_max / dp_norm))
            : X_des.translation();

    const Eigen::Quaterniond q_meas = X_meas.rotation().ToQuaternion();
    const Eigen::Quaterniond q_des = X_des.rotation().ToQuaternion();
    const Eigen::AngleAxisd aa_diff(q_des * q_meas.inverse());
    RotationMatrix<double> R_clamped = X_des.rotation();
    if (aa_diff.angle() > FLAGS_cube_ik_lead_rot_max) {
      const Eigen::AngleAxisd aa_capped(FLAGS_cube_ik_lead_rot_max,
                                        aa_diff.axis());
      R_clamped =
          RotationMatrix<double>(Eigen::Quaterniond(aa_capped) * q_meas);
    }
    return RigidTransform<double>(R_clamped, p_clamped);
  };

  // --contact_force_log cadence: control_dt is 1kHz, throttle down to
  // --contact_force_log_hz.
  long contact_log_iter = 0;
  const int contact_log_period_steps = std::max(
      1, static_cast<int>(
             std::lround(1.0 / (FLAGS_contact_force_log_hz * control_dt))));

  // Print the current contact-force-to-cube-motion map after every C3 solve.
  // C3 scales the LCS D matrix internally, so divide by AnDn_ to report the
  // physical map from force (N) to state increment (m or m/s).
  auto print_cube_lambda_map = [&](const LCS& lcs, double t) {
    const int nc = n_contacts;
    const int i_cx = n_hand_q + 4;
    const int i_cy = n_hand_q + 5;
    const int i_cz = n_hand_q + 6;
    const int i_vx = n_pos + n_hand_v + 3;
    const int n_fd = FLAGS_num_friction_directions;
    const int n_beta_per = 2 * n_fd;
    const double an_dn = c3->GetLambdaScaling();
    const MatrixXd D = lcs.D()[0] / an_dn;
    const MatrixXd D_xyz = D.block(i_cx, 0, 3, n_lambda);
    const MatrixXd D_vxyz = D.block(i_vx, 0, 3, n_lambda);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== D: cube xyz <- lambda @ t=" << t
              << " s (physical, unscaled LCS) ===\n"
              << "  state rows: cube_x=" << i_cx << " cube_y=" << i_cy
              << " cube_z=" << i_cz << "   (n_x=" << n_x
              << ", n_lambda=" << n_lambda << ")\n"
              << "  lambda cols: gamma[0.." << (nc - 1) << "]  lambda_n["
              << nc << ".." << (2 * nc - 1) << "]  beta[" << (2 * nc)
              << ".." << (n_lambda - 1) << "]  (nc=" << nc
              << ", n_fd=" << n_fd << " -> " << n_beta_per
              << " beta/contact)\n"
              << "  C3 AnDn_ (internal lambda scale)=" << an_dn
              << "  - printed D is physical; internal D = AnDn_ * this\n";

    const char* row_names[3] = {"cube_x", "cube_y", "cube_z"};
    auto print_block = [&](const char* title, int c0, int ncols) {
      std::cout << "  -- " << title << " (cols " << c0 << ".."
                << (c0 + ncols - 1) << ") --\n";
      std::cout << "           ";
      for (int j = 0; j < ncols; ++j)
        std::cout << std::setw(11) << ("c" + std::to_string(c0 + j));
      std::cout << "\n";
      for (int r = 0; r < 3; ++r) {
        std::cout << "  " << std::setw(8) << row_names[r];
        for (int j = 0; j < ncols; ++j)
          std::cout << std::setw(11) << D_xyz(r, c0 + j);
        std::cout << "\n";
      }
    };

    if (FLAGS_contact_model == "stewart_and_trinkle") {
      print_block("gamma (slacks; expect ~0)", 0, nc);
      print_block("lambda_n [idx, mid, thu]", nc, nc);
      for (int i = 0; i < nc; ++i) {
        const int c0 = 2 * nc + i * n_beta_per;
        const std::string title =
            std::string("beta contact ") + std::to_string(i) +
            (i == 0 ? " (index)" : i == 1 ? " (middle)" : " (thumb)");
        print_block(title.c_str(), c0, n_beta_per);
      }
    } else {
      print_block("combined cone forces", 0, n_lambda);
    }

    if (FLAGS_contact_model == "stewart_and_trinkle") {
      const VectorXd Dz = D_xyz.row(2).transpose();
      const VectorXd Dzn = Dz.segment(nc, nc);
      const VectorXd Dzb = Dz.tail(n_lambda - 2 * nc);
      std::cout << "  -- cube_z row summaries --\n"
                << "  D_z[lambda_n] (m per N) [idx,mid,thu] = "
                << Dzn.transpose() << "\n"
                << "  |D_z[gamma]|_max="
                << Dz.head(nc).cwiseAbs().maxCoeff()
                << "  |D_z[lambda_n]|_max=" << Dzn.cwiseAbs().maxCoeff()
                << "  |D_z[beta]|_max=" << Dzb.cwiseAbs().maxCoeff() << "\n"
                << "  Delta z from unit lambda_n=[1,1,1]: " << Dzn.sum()
                << " m/step\n"
                << "  Delta z from alpha_m target lambda_n=["
                << FLAGS_alpha_m << "," << FLAGS_alpha_m << ","
                << (2.0 * FLAGS_alpha_m) << "]: "
                << (FLAGS_alpha_m * Dzn(0) + FLAGS_alpha_m * Dzn(1) +
                    2.0 * FLAGS_alpha_m * Dzn(2))
                << " m/step\n"
                << "  freefall Delta z over c3_dt (0.5*g*dt^2, g=9.81): "
                << (-0.5 * 9.81 * FLAGS_c3_dt * FLAGS_c3_dt) << " m\n";

      std::cout << "  -- cube linear vel rows (vx,vy,vz) <- lambda_n only --\n"
                << "  D_v[lambda_n] (m/s per N):\n"
                << "    vx: " << D_vxyz.block(0, nc, 1, nc) << "\n"
                << "    vy: " << D_vxyz.block(1, nc, 1, nc) << "\n"
                << "    vz: " << D_vxyz.block(2, nc, 1, nc) << "\n"
                << "  |D_vz[beta]|_max="
                << D_vxyz.row(2).tail(n_lambda - 2 * nc).cwiseAbs().maxCoeff()
                << "  (non-zero beta->vz means friction can oppose gravity)\n\n";
    }
    std::cout << std::defaultfloat;
  };

  for (double t = control_dt; t < FLAGS_sim_time; t += control_dt) {
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
                 FLAGS_kd * (qd_tgt - v_hand)
                  + tau_g_hand;
      
      

      // ── Handoff: all three fingers arrived, settle time elapsed ─────────
      // Gated on a fixed dwell time since the LAST finger's arrived[] latch,
      // not on any per-step contact measurement — both the LCS gap φ and the
      // instantaneous contact-force detector proved too noisy at the 1ms
      // scale (see handoff_settle_time flag). The PD hold has already brought
      // the fingertips to a static equilibrium well within this window.
      const double last_arrival_t =
          std::max({arrived_time[0], arrived_time[1], arrived_time[2]});
      if (arrived[0] && arrived[1] && arrived[2] &&
          t >= last_arrival_t + FLAGS_handoff_settle_time) {
        std::cout << "[t=" << t
                  << "] all fingers settled → C3 handoff\n";

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
          std::cout << "  contact gaps φ [index, middle, thumb] = "
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
        c3_opts.rho_scale  = FLAGS_rho_scale;
        c3_opts.warm_start = FLAGS_warm_start;
        c3_opts.scale_lcs  = true;
        c3_opts.gamma      = 1.0;

        c3 = std::make_unique<C3Plus>(
            lcs_init, C3::CostMatrices(Q_vec, R_vec, G_vec, U_vec),
            x_desired, c3_opts);
        // Carry ADMM state across solves (see --warm_start_admm). The object
        // outlives relinearizations (UpdateLCS below), so the warm start
        // persists; the first solve after this fresh construction cold-starts.
        c3->SetAdmmWarmStartAcrossSolves(FLAGS_warm_start_admm);

        // Force reference on each contact's NORMAL force — seeds a real grip
        // and regularises the squeeze nullspace. A contact's normal force is
        // the sum over normal_groups[i] (a single λ_n entry for Stewart-
        // Trinkle, the whole cone block for Anitescu), so the per-contact cost
        // w_lambda·(Σ_{j∈g} λ_j − αᵢ)² is built from the rank-1 block 1·1ᵀ over
        // the group with λ_des spreading αᵢ evenly across it. For Stewart-
        // Trinkle (singleton group) this collapses to the old diagonal cost on
        // λ_n. Targets: index=middle=alpha_m, thumb=2·alpha_m (force closure).
        //
        // C3 scales lambda internally: lambda_physical = AnDn_ * lambda_int.
        // update_force_tracking() keeps alpha_m in user-facing physical
        // Newton units by converting both the target and its quadratic weight
        // into the current internal coordinates.
        if (FLAGS_w_lambda > 0.0) {
          update_force_tracking();
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
        // UpdateLCS() may change AnDn_; preserve the physical force target.
        update_force_tracking();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        relin_ms_sum += ms;
        relin_ms_max = std::max(relin_ms_max, ms);
        ++relin_calls;
      }

      // t_ref_now: seconds since pin release, clamped to 0 beforehand — the
      // single time base shared by the IK target, the plan's cube-pose
      // reference, and (one horizon ahead) the plan's hand-q reference.
      const double t_ref_now = std::max(0.0, t - (handoff_t + 0.5));
      const double t_ref_end = t_ref_now + FLAGS_N * FLAGS_c3_dt;

      // Ghost cube: raw (unclamped) commanded target, same call the IK lead
      // and the C3 cube-pose cost use at k=0 — so it always matches what's
      // actually being commanded, including when the IK-lead clamp is
      // silently capping how far the fingers can chase it.
      if (FLAGS_show_cube_target) {
        meshcat->SetTransform("/cube_target",
                              cube_target_pose(t_ref_now).first);
      }

      // Moving contact reference (Mod 2 + Mod 3): re-solve the 3-point grasp
      // IK against the DESIRED cube pose — not the measured one — so the
      // fingers LEAD the commanded motion instead of re-wrapping around
      // wherever the cube currently is (which is what made --cube_bob_amp
      // produce lateral chasing instead of vertical motion: the old IK call
      // tracked the MEASURED pose while the cube's z-target moved
      // independently, so the two references fought). Two solves per relin —
      // now (k=0) and one horizon ahead (k=N) — so the per-knot hand-q
      // reference below can interpolate instead of freezing the whole
      // horizon at a single instant. Done at relin cadence (an IK solve per
      // control tick would be wasteful). Pinned → cube frozen, skip.
      if (FLAGS_track_cube_contact && !cube_pinned && do_relin) {
        const auto ik_t0 = std::chrono::steady_clock::now();
        const RigidTransform<double> X_WC_meas = CubePoseFromPositions(
            sim_plant.GetPositions(plant_ctx, sim_cube));
        const std::pair<RigidTransform<double>, Vector3d> pose_now =
            cube_target_pose(t_ref_now);
        const std::pair<RigidTransform<double>, Vector3d> pose_end =
            cube_target_pose(t_ref_end);
        q_contact_live =
            resolve_contact_ik(clamp_ik_lead(pose_now.first, X_WC_meas));
        q_contact_live_end =
            resolve_contact_ik(clamp_ik_lead(pose_end.first, X_WC_meas));
        const double ik_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - ik_t0)
                                 .count();
        // Save/restore stream format — a bare setprecision here leaks into
        // every later print (the "dt=0.0" / "|A|=70.3" corruption).
        const auto cout_flags = std::cout.flags();
        const auto cout_prec = std::cout.precision();
        std::cout << "  [track IK] " << std::fixed << std::setprecision(1)
                  << ik_ms << " ms\n";
        std::cout.flags(cout_flags);
        std::cout.precision(cout_prec);
      }

      // Update C3's desired-state trajectory. Two overrides on top of
      // x_des_base, either of which triggers a rebuild — and, unlike before,
      // both now come from the SAME --cube_motion_mode pose function, so
      // they can no longer disagree about where the cube should be:
      //   track:  the 3 grasping fingers' per-knot q-target ← linear
      //           interpolation between the k=0 and k=N IK solves above
      //           (Mod 3), instead of one IK solve pasted across every knot.
      //   motion: the cube's full pose (quaternion + xyz) and translational
      //           velocity, per knot, from cube_target_pose(tk). State
      //           layout: cube quaternion at n_hand_q..+3, cube xyz at
      //           n_hand_q+4..+6, cube linear velocity at
      //           n_pos+n_hand_v+3..+5. Horizon k=0..N filled with the
      //           look-ahead so C3 tracks ahead, not a lagged step.
      const bool track = (FLAGS_track_cube_contact && !cube_pinned);
      const bool motion = (FLAGS_cube_motion_mode != "none" && !cube_pinned);
      if (track || motion) {
        std::vector<VectorXd> x_des_traj(FLAGS_N + 1, x_des_base);
        if (track) {
          for (int k = 0; k <= FLAGS_N; ++k) {
            const double frac =
                FLAGS_N > 0 ? static_cast<double>(k) / FLAGS_N : 0.0;
            const VectorXd q_k =
                (1.0 - frac) * q_contact_live + frac * q_contact_live_end;
            for (int i = 0; i < 3; ++i)
              x_des_traj[k].segment(finger_start[i], 4) =
                  q_k.segment(finger_start[i], 4);
          }
        }
        if (motion) {
          for (int k = 0; k <= FLAGS_N; ++k) {
            const double tk = t_ref_now + k * FLAGS_c3_dt;
            const std::pair<RigidTransform<double>, Vector3d> pose_vel =
                cube_target_pose(tk);
            Eigen::Quaterniond quat = pose_vel.first.rotation().ToQuaternion();
            if (quat.w() < 0.0) quat.coeffs() *= -1.0;  // match q_cube0's +w
            x_des_traj[k](n_hand_q + 0) = quat.w();
            x_des_traj[k](n_hand_q + 1) = quat.x();
            x_des_traj[k](n_hand_q + 2) = quat.y();
            x_des_traj[k](n_hand_q + 3) = quat.z();
            x_des_traj[k].segment(n_hand_q + 4, 3) = pose_vel.first.translation();
            x_des_traj[k](n_pos + n_hand_v + 3) = pose_vel.second.x();
            x_des_traj[k](n_pos + n_hand_v + 4) = pose_vel.second.y();
            x_des_traj[k](n_pos + n_hand_v + 5) = pose_vel.second.z();
          }
          last_zref = x_des_traj[0](n_hand_q + 6);
        }
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

        // Print the map for every completed solve, including after each
        // relinearization, so changes in contact geometry are visible.
        // Gated behind --lambda_map_debug (off by default) — see flag doc.
        if (FLAGS_lambda_map_debug) {
          print_cube_lambda_map(c3->GetLCS(), t);
        }

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
            const auto& F = lcs.F();
            const auto& H = lcs.H();
            const auto& cc = lcs.c();
            // Units: GetLCS()'s F/H/c/E are in C3's internal (scale_lcs)
            // units, and GetForceSolution()'s λ is rescaled back to physical
            // (× AnDn_) — so divide it back by AnDn_ to combine with the
            // matrices. GetStateSolution / GetInputSolution are already in
            // matching units. (Same convention as the SOLVER DIAG block.)
            const double an_dn = c3->GetLambdaScaling();
            std::vector<VectorXd> gap_plan(n_knots), lamn_plan(n_knots);
            for (int k = 0; k < n_knots; ++k) {
              // TRUE contact gap η = E·x + F·λ + H·u + c — the actual
              // complementarity variable paired with λ_n (0 ≤ λ_n ⊥ η_n),
              // NOT the partial E·x+c. The normal-force block lives at
              // segment(n_contacts, n_contacts). Where λ_n > 0 (contact
              // active), η_n should be ≈ 0.
              const VectorXd eta_full = E.at(k) * xplan[k] +
                                        F.at(k) * (lam_plan[k] / an_dn) +
                                        H.at(k) * u_plan[k] + cc.at(k);
              gap_plan[k]  = eta_full.segment(n_contacts, n_contacts);
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

            std::vector<std::string> c_labels{"idx", "mid", "thu"};

            print_table("CONTACT GAP η=Ex+Fλ+Hu+c (per contact; ⊥ λ_n)",
                        gap_plan, c_labels);

            // FK GAP: forward-kinematics of the PLANNED joint+cube trajectory
            // (xplan[k], NOT the current sim state) onto lcs_plant, then the
            // REAL Drake signed-distance query (GeomGeomCollider, the exact
            // same call lcs_factory.cc uses to build phi/eta) — not a
            // hand-rolled box-distance approximation. xplan[k].head(n_pos) is
            // already [hand_q(16); cube_q(7)] in lcs_plant's own position
            // order (see the state-layout comment above), so it drops
            // straight into SetPositions with no reassembly.
            //
            // η above is the model's complementarity gap using the contact
            // Jacobian/normal frozen at the LAST relinearization — the same
            // witness point/normal is reused for every knot k=0..N even
            // though hand_q is planned to keep moving. FK GAP recomputes the
            // TRUE geometry fresh at each knot's own planned configuration,
            // so the two should roughly agree at k=0 (both anchored at the
            // relinearization point) and can diverge by k=N — a direct
            // measure of how stale the single-linearization-over-the-horizon
            // assumption has become by the end of the plan.
            std::vector<VectorXd> fk_gap_plan(xplan.size());
            for (size_t k = 0; k < xplan.size(); ++k) {
              lcs_plant.SetPositions(&lcs_ctx, xplan[k].head(n_pos));
              VectorXd d(3);
              for (int i = 0; i < 3; ++i) {
                GeomGeomCollider<double> collider(lcs_plant, contact_pairs[i]);
                d(i) = collider.GetGeometryQueryResult(lcs_ctx).distance;
              }
              fk_gap_plan[k] = d;
            }
            // Restore lcs_ctx to the real current state — it's shared with
            // the relinearization logic elsewhere in the loop.
            lcs_plant.SetPositionsAndVelocities(&lcs_ctx, x_current);
            print_table(
                "FK GAP (planned q_k → true signed distance; neg=inside; cf. η above)",
                fk_gap_plan, c_labels);

            print_table("NORMAL FORCE λ_n (per contact, force)", lamn_plan, c_labels);

            // NET FRICTION FORCE (global +Z): the vertical support the grasp
            // actually provides against gravity. Normal forces are horizontal
            // (±Y) so give ~0 vertical support — ALL vertical hold is friction.
            // Reconstructed from the LCS's own contact model: the physical D
            // matrix maps physical friction β to the cube's z-velocity
            // increment Δvz, so the friction force is Fz = m_cube/dt · Δvz.
            // Per contact + total; compare total to mg (printed). β for contact
            // i occupies λ cols [2nc + i·(2·n_fd) .. +2·n_fd); the cube's linear
            // z-velocity is state row n_pos+n_hand_v+5.
            {
              const MatrixXd D_phys = lcs.D()[0] / an_dn;
              const int i_vz = n_pos + n_hand_v + 5;
              const int n_beta_per = 2 * FLAGS_num_friction_directions;
              const double m_cube =
                  lcs_plant.get_body(lcs_plant.GetBodyIndices(lcs_cube)[0])
                      .get_mass(lcs_ctx);
              std::vector<VectorXd> ffz_plan(n_knots);
              std::vector<double> ffz_total(n_knots, 0.0);
              for (int k = 0; k < n_knots; ++k) {
                VectorXd fz(nc);
                for (int i = 0; i < nc; ++i) {
                  const int b0 = 2 * nc + i * n_beta_per;
                  double dvz = 0.0;
                  for (int j = 0; j < n_beta_per; ++j)
                    dvz += D_phys(i_vz, b0 + j) * lam_plan[k](b0 + j);
                  fz(i) = m_cube / FLAGS_c3_dt * dvz;
                }
                ffz_plan[k] = fz;
                ffz_total[k] = fz.sum();
              }
              print_table("NET FRICTION FORCE Fz (global +Z, per contact, N)",
                          ffz_plan, c_labels);
              std::cout << std::setw(10) << "total";
              for (int k = 0; k < n_knots; ++k)
                std::cout << std::setw(12) << std::fixed << std::setprecision(4)
                          << ffz_total[k];
              std::cout << "    (mg=" << std::setprecision(4) << (m_cube * 9.81)
                        << " N)\n";
            }

            // Cube z-position from the STATE plan (N+1 knots, unlike the N-knot
            // force tables above). State layout: cube xyz sits at
            // [n_hand_q+4, n_hand_q+5, n_hand_q+6].
            std::vector<VectorXd> cubez_plan(xplan.size());
            for (size_t k = 0; k < xplan.size(); ++k)
              cubez_plan[k] = xplan[k].segment(n_hand_q + 6, 1);
            print_table("CUBE Z POSITION (state knots)", cubez_plan,
                        {"cube_z"});

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

      if (FLAGS_exec_mode == "direct") {
        // Architecture A — pure C3 input playback (receding horizon). The
        // plan IS the controller: apply the first input u0 as generalized
        // torque. Gravity is baked into the LCS, so u0 already accounts for
        // it — NO tau_g, NO PD, NO task-space grip. Runs in both the pinned
        // warm-up and post-release, so it directly tests whether C3's own
        // plan holds the cube. Clamped to the torque budget below (kC3).
        tau_hand = s_u * c3->GetInputSolution()[0];
      } else if (FLAGS_exec_mode == "osc") {
        // Simple joint-space controller (replaces the inverse-dynamics OSC):
        //   tau = tau_grav + kp*(q_target - q_hand) + Σ_i J_iᵀ (fn_i · n_i)
        // Three additive pieces, nothing more:
        //   tau_grav : hand gravity compensation (holds static under gravity).
        //   kp*(..)  : pure proportional pull toward q_target (below). No mass
        //              matrix, no Coriolis, no velocity damping term.
        //   J^T·fn   : C3's planned normal grip lambda_n at each fingertip,
        //              applied along the (rotating) contact normal via
        //              Jacobian-transpose — the task-space force piece.
        //
        // q_target (Mod 4): previously xplan_now[0] — C3's knot-0 hand config,
        // which the QP's own initial-condition constraint HARD-PINS to the
        // measured state at every solve. That made this PD term a structural
        // no-op (q_target ≈ q_hand always, by construction — no plan, however
        // good, can ever move it): with a moving --cube_motion_mode reference,
        // the cube's target visibly moved in the plan/IK, but nothing here
        // ever read it, so the hand never chased it. Two real targets used
        // instead, both able to actually move:
        //   --track_cube_contact on:  q_contact_live, the live grasp-IK
        //     solution against the DESIRED cube pose (Mods 2/3) — the
        //     fingertips lead the commanded motion, and the cube rides along
        //     via contact/friction, exactly as a real grasp transports an
        //     object.
        //   --track_cube_contact off: xplan_now[1], C3's own solved NEXT
        //     knot — the fix the exec_mode=osc flag's docstring already
        //     promised ("realizes (x1, lambda0)") but the code never
        //     implemented. Knot 1 is the earliest knot the QP can actually
        //     move (knot 0 can't, per above), so this is the plan's real
        //     one-step-ahead intent, not a frozen restatement of "now."
        const VectorXd q_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);
        const std::vector<VectorXd> xplan_now = c3->GetStateSolution();
        VectorXd q_des = q_hand;
        if (FLAGS_track_cube_contact) {
          q_des = q_contact_live;
        } else if (xplan_now.size() > 1) {
          q_des = xplan_now[1].head(n_hand_q);
        }

        // Gravity compensation: tau = -tau_gravity holds the hand static.
        const VectorXd tau_grav = sim_plant.GetVelocitiesFromArray(
            sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));

        // Pure joint-space PD, implicit qd_des=0 (same "hold still" damping
        // convention as the reach-phase PD) — NOT via CalcInverseDynamics/
        // M(q), which was the earlier, heavier architecture that injected
        // energy at pin release and kicked the cube loose. See --osc_kd's
        // doc: this direct-torque damping is gentler than that but untested.
        const VectorXd tau_pd =
            FLAGS_osc_kp * (q_des - q_hand) - FLAGS_osc_kd * v_hand;

        // Feedforward contact normal force from C3's own planned lambda_n
        // (physical units: GetForceSolution() / GetLambdaScaling()).
        const std::vector<VectorXd> lam_plan_now = c3->GetForceSolution();
        const VectorXd lam0 = lam_plan_now[0] / c3->GetLambdaScaling();
        const RotationMatrix<double> R_WC =
            CubePoseFromPositions(sim_plant.GetPositions(plant_ctx, sim_cube))
                .rotation();
        const std::array<Vector3d, 3> press_C{
            Vector3d(0, 1, 0), Vector3d(0, 1, 0), Vector3d(0, -1, 0)};
        VectorXd tau_force = VectorXd::Zero(n_hand_v);
        for (int i = 0; i < 3; ++i) {
          double fn = 0.0;
          for (int idx : normal_groups[i]) fn += lam0(idx);
          Eigen::MatrixXd J(3, sim_plant.num_velocities());
          sim_plant.CalcJacobianTranslationalVelocity(
              plant_ctx, drake::multibody::JacobianWrtVariable::kV,
              sim_plant.get_body(tip_bodies[i]).body_frame(), Vector3d::Zero(),
              sim_plant.world_frame(), sim_plant.world_frame(), &J);
          tau_force +=
              J.leftCols(n_hand_v).transpose() * (fn * (R_WC * press_C[i]));
        }

        tau_hand = tau_grav + tau_pd + tau_force;
      } else if (cube_pinned) {
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
        // Pin released — task-space PD + Jacobian-transpose grip executor.
        // Desired per-fingertip normal force = C3's projected λ, optionally
        // floored so the grip survives the contact-gate hover band. Thumb
        // floor = 2× (force closure).
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
          for (int i = 0; i < 3; ++i)
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
        const std::array<Vector3d, 3> press_C{
            Vector3d(0, 1, 0), Vector3d(0, 1, 0), Vector3d(0, -1, 0)};
        // Gravity is already inside the LCS (baked into A/d), so C3's u0
        // (added below as s_u*u0) already accounts for it. Adding tau_g here
        // double-counts gravity — off by default, toggleable to A/B.
        VectorXd tau =
            FLAGS_exec_grav_comp ? tau_g : VectorXd::Zero(tau_g.size());
        for (int i = 0; i < 3; ++i) {
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

    // LCM telemetry: index-finger joint tracking — actual angles vs. the C3
    // plan's knot-0 / knot-1 index angles (the OSC PD setpoint is knot-1).
    // See GRASP_IDX_TRACK wiring above. Before C3 has a solution (reach
    // phase), the plan slots mirror the actual angles so the traces overlap.
    if (idx_input != nullptr) {
      const VectorXd q_hand_now =
          sim_plant.GetPositions(plant_ctx, sim_allegro);
      const VectorXd idx_act = q_hand_now.segment(finger_start[0], 4);
      VectorXd idx_vec(12);
      idx_vec.segment(0, 4) = idx_act;
      idx_vec.segment(4, 4) = idx_act;  // plan knot0 (overwritten if available)
      idx_vec.segment(8, 4) = idx_act;  // plan knot1 (overwritten if available)
      if (c3 && solve_calls > 0) {
        const std::vector<VectorXd> xpl = c3->GetStateSolution();
        if (xpl.size() > 0)
          idx_vec.segment(4, 4) = xpl[0].segment(finger_start[0], 4);
        if (xpl.size() > 1)
          idx_vec.segment(8, 4) = xpl[1].segment(finger_start[0], 4);
      }
      idx_input->GetMutableVectorData<double>()->SetFromVector(idx_vec);
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
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
