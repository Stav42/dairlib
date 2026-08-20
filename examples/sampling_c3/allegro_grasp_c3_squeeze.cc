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
DEFINE_double(ring_tip_surface_offset_z, 0.0115,
              "Same meaning as --tip_surface_offset_z (offset along "
              "link_11_tip's own local +Z from its frame origin to the "
              "true fingertip collision surface), but for ring specifically "
              "— started from the SAME empirically-found value as "
              "index/middle/thumb (0.0115, ring is mechanically the same "
              "finger module), pulled in slightly per visual tuning via the "
              "/tip_frame/ring_surface cube. Used for real: "
              "SolveGraspIKWithRing constrains this point (combined with "
              "--ring_tip_surface_offset_y) when solving q_contact/"
              "q_pregrasp/q_release_middle/q_regrasp_ring, not just for "
              "the cube marker.");
DEFINE_double(ring_tip_surface_offset_y, 0.005,
              "Offset (m) along link_11_tip's own local +Y from the frame "
              "origin to the true fingertip collision surface — confirmed "
              "by eye via the /tip_frame/ring_surface cube (X tried first, "
              "moved to Y; 0.005 is the visually-confirmed value, same "
              "convention as --tip_surface_offset_z for the other three). "
              "Now used for real, not just visualization: "
              "SolveGraspIKWithRing constrains this point — not "
              "link_11_tip's raw origin — when solving q_contact/"
              "q_pregrasp/q_release_middle/q_regrasp_ring.");
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
              "cube lags the reference the fingers don't chase a target it "
              "has not reached. Generous default — only binds if the cube "
              "falls far behind a fast/large commanded motion.");
DEFINE_double(cube_ik_lead_rot_max, 0.3,
              "Safety clamp (rad): same as --cube_ik_lead_pos_max for the "
              "rotational part of the IK lead.\n"
              "NOTE 0.3 rad is 17.2 deg — SMALLER than one --gait cycle's "
              "rotation, so a gait will bind against it, and bind SILENTLY: "
              "C3's cube-pose target, the Meshcat ghost and --rot_log's cmd "
              "all read the RAW schedule, not the clamped pose the fingers "
              "actually receive. A clamped run therefore looks identical in "
              "the logs while tracking worse than the numbers imply. Raise "
              "it (1.0-1.5, or 3.2 to disable — the angle between two "
              "orientations never exceeds pi) for any run whose commanded "
              "rotation exceeds ~17 deg.");
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
DEFINE_bool(rot_log, false,
            "Log cube ORIENTATION tracking every --rot_log_period seconds "
            "post-handoff. Everything else in this file logs cube z only "
            "(--cube_bob_amp's heritage), which says nothing about how far "
            "the cube actually TURNED. Prints, all relative to X_WC0: the "
            "commanded and measured rotation angles, the residual "
            "orientation error, how far the measured rotation axis has "
            "drifted off the commanded one (the slip signature — a grasp "
            "losing the cube rotates about the wrong axis, not just by the "
            "wrong amount), and the position drift. Use it to find how "
            "large a single-grasp rotation the fingers can actually "
            "deliver.");
DEFINE_double(rot_log_period, 0.25,
              "Seconds between --rot_log lines.");
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
DEFINE_bool(c3_joint_plan_log, false,
            "Print C3's complete planned hand-joint trajectory q[0..N] "
            "after every solve, together with measured q and per-finger "
            "q1-minus-measured norms. Intended to diagnose whether "
            "--osc_target_source=c3 is producing sensible PD targets.");
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
DEFINE_string(osc_target_source, "auto",
              "Joint-position target used by --exec_mode=osc: "
              "'auto' preserves the legacy rule (live IK when "
              "--track_cube_contact, otherwise C3 knot 1); 'ik' always uses "
              "q_contact_live; 'c3' always uses C3's planned knot-1 hand "
              "configuration. This is independent of --track_cube_contact, "
              "so a gait can keep updating C3's moving grasp/cube reference "
              "while the executor PD tracks C3 q1 instead of IK.");
DEFINE_double(osc_kp, 300.0,
              "OSC joint-space position gain (Nm/rad), applied directly in "
              "torque space as kp*(q_des-q) — NOT an inverse-dynamics "
              "acceleration gain despite the units its name suggests; the "
              "current exec_mode=osc law never calls CalcInverseDynamics.");
DEFINE_double(osc_kd, 15.0,
              "OSC joint-space damping gain (Nm*s/rad), applied as "
              "-kd*(qdot - qdot_des) — see --osc_qd_filter_tau for how "
              "qdot_des is estimated.\n"
              "CORRECTION to this flag's earlier doc: it used to be applied "
              "as -kd*qdot with qdot_des IMPLICITLY zero, described as 'same "
              "convention as the reach-phase --kd' — that was wrong, the "
              "reach phase's PD already used a real qd_tgt from its spline "
              "(FLAGS_kd * (qd_tgt - v_hand)), never implicit-zero. The "
              "implicit-zero form here meant ANY --osc_kd > 0 fought the "
              "fingers' own legitimate tracking motion during an active "
              "gait rotation — it cannot tell 'moving because tracking "
              "correctly' from 'moving because of an unwanted kick', and "
              "opposed both equally. Measured: even --osc_kd=0.1 was enough "
              "to weaken grip below what active rotation needs. Fixed by "
              "giving this a real qdot_des instead of assuming zero.");
DEFINE_double(osc_qd_filter_tau, 0.05,
              "Time constant (s) for a low-pass filter on the finite-"
              "differenced velocity of q_des (--osc_kd's qdot_des). A RAW "
              "finite difference would inherit every discontinuity in "
              "q_des directly — including --track_ik_period_steps' ~5ms "
              "position-target jumps, so the 'velocity' estimate would "
              "spike exactly on those, and --osc_kd would then read that "
              "spike as 'not moving fast enough' and add torque right at "
              "the jump — the opposite of gentle. This filter turns each "
              "step into a smooth ramp over roughly this time constant "
              "instead of an instant jump, so the derived qdot_des tracks "
              "real, sustained motion (a rotating grasp, a gait leg's "
              "spline) without inheriting the sampling artifacts of "
              "whichever source is currently driving q_des.");
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
DEFINE_double(lambda_torque_scale, 1.0,
              "Scale applied to the Jacobian-transpose torque generated from "
              "C3's planned lambda forces. 1 = normal behavior; 0 disables "
              "lambda-force feedforward while leaving position PD active.");
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
DEFINE_int32(track_ik_period_steps, 40,
             "Re-solve the live --track_cube_contact IK (q_contact_live/"
             "q_contact_live_end, via resolve_contact_ik) once every this "
             "many control steps — independent of --relin_period_steps. "
             "The two used to share relin's cadence for no real reason "
             "other than convenience: resolve_contact_ik is a plain "
             "kinematic IK solve (SolveGraspIK, no dynamics, no AutoDiff), "
             "nothing like relin's LCSFactory::LinearizePlantToLCS (which "
             "differentiates through the full contact dynamics and is much "
             "more expensive). Drop this on its own for smoother fine-"
             "manipulation tracking (see --relin_period_steps's docs on "
             "why a coarse update looks like a staircase during continuous "
             "cube motion) without paying to relinearize that often too. "
             "The OSC executor's q_des no longer reads that staircase raw: "
             "it ramps linearly from one solve to the next over this same "
             "period (q_contact_live_ramp_from/track_ik_ramp_t0), so this "
             "flag also sets the ramp's duration — the actual fix for the "
             "continuous, rotation-only jerk --jerk_log traced to this "
             "cadence meeting a --osc_kd=0 (no velocity term) PD law.");
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

// ── Single-finger release (step 1 of finger-gaiting, built incrementally) ──
// 4-finger triangle grasp (index/middle/ring on -Y, thumb on +Y). At
// --release_middle_t seconds after the cube unpins, retract --release_finger
// (middle or ring) off the cube and rebuild C3 to solve with only the
// remaining 3 contacts, not a stale 4-contact problem with the released one
// masked downstream. --release_finger=middle stays retracted forever (no
// recontact built yet); --release_finger=ring instead free-space PD-drives
// toward a new point (same X, middle's Z height) and rejoins C3 once it
// touches there — see q_regrasp_ring/ring_touch_latched/finger_rejoined.
DEFINE_bool(release_middle, false,
            "If true, run the 4-finger triangle grasp (index/middle/ring "
            "on -Y, thumb on +Y) and retract one of {middle, ring} "
            "partway through the run — see --release_finger, "
            "--release_middle_t/--release_middle_offset. Default false is "
            "a no-op — behavior is unchanged from before this flag "
            "existed.");
DEFINE_string(release_finger, "middle",
              "Which finger --release_middle retracts at "
              "--release_middle_t: \"middle\" or \"ring\". If \"ring\", it "
              "PD-moves to a new point (same X, middle's Z height) and "
              "rejoins C3 on touch (see --regrasp_settle_time); if "
              "\"middle\", it just stays retracted. The two fingers not "
              "picked keep holding at their triangle positions throughout.");
DEFINE_double(release_middle_t, 3.0,
              "Seconds after the cube unpins (not absolute sim time) to "
              "release --release_finger.");
DEFINE_double(release_middle_offset, 0.02,
              "How far outward (m), along --release_finger's current "
              "face normal, to retract it once released — just enough to "
              "break contact.");
DEFINE_double(regrasp_settle_time, 0.15,
              "Seconds to hold contact at a regrasp's new point (ring's, "
              "then middle's — see q_regrasp_ring/q_regrasp_middle) before "
              "trusting it and rejoining C3 — same rationale as "
              "--handoff_settle_time, independently tunable. Shared by "
              "both regrasp legs.");
DEFINE_double(regrasp_duration, 1.0,
              "Seconds for a regrasp's PD motion from its old contact "
              "point to the new one to cover, via a CubicShapePreserving "
              "spline built at the moment of release from wherever the "
              "finger actually is then — not an instant position-target "
              "jump. Shared by both regrasp legs (ring, then middle).");
DEFINE_double(regrasp_touch_tol, 0.01,
              "Max distance (m) from a regrasp's target point (ring_"
              "regrasp_target, then middle_regrasp_target) for a "
              "touching[] reading to count as the real regrasp. Without "
              "this, a sagging cube (weaker grip with a finger out of the "
              "LCS) can swing back into contact with it while it's still "
              "near its OLD point early in the slow-starting spline, and "
              "that false touch would rejoin C3 there instead of at the "
              "new point. Shared by both regrasp legs.");
DEFINE_double(regrasp_arc_clearance, 0.03,
              "How far outward (m) beyond the face, along its -Y normal, "
              "a regrasp path's midpoint bulges before sliding to the new "
              "Z height and back in. A plain 2-knot joint-space spline "
              "from the old point straight to the new one has no notion "
              "of the cube's geometry and can stay close to (or drag "
              "across) the face the whole way; routing through this "
              "lifted midpoint forces a real arc that clears the surface. "
              "Shared by both regrasp legs (q_regrasp_ring_mid, then "
              "q_regrasp_middle_mid).");
// (Superseded: ring used to target the +X face independently, with its own
// --release_middle_ring_penetration/offset_{y,z} flags. Now it sits on the
// -Y face as the triangle's base-right point, mirroring index — see
// --release_middle_tri_* below and ring_target's construction.)

// ── Triangle topology: index/middle/ring, all on -Y (--release_middle) ────
// The real --release_middle grasp target: index and ring level with each
// other (the triangle's base), middle above them (the apex), all three on
// the -Y face — wired into q_contact_targets/pregrasp/IK/LCS, not just a
// preview. Thumb is unchanged, still on +Y.
DEFINE_double(release_middle_tri_spread, 0.015,
              "Half-distance (m) between index and ring along the face's "
              "lateral (X) axis — the triangle's base half-width. Face "
              "half-width is 0.03 (cube_size/2), so this leaves ~15mm "
              "margin to the edge. Was 0.022 (~8mm margin) until "
              "empirical testing found index/ring too far apart.");
DEFINE_double(release_middle_tri_base_z, -0.008,
              "Index/ring's vertical (Z) offset (m) from the -Y face's "
              "center — the triangle's base, below center by default.");
DEFINE_double(release_middle_tri_apex_z, 0.02,
              "Middle's vertical (Z) offset (m) from the -Y face's center "
              "— the triangle's apex, above center by default. Face "
              "half-width is 0.03, so this leaves ~1cm margin to the edge.");

// ── Gait: reorient the cube by repeated rotate-then-regrasp cycles ───────────
// Rotates the cube about the GRASP AXIS (the cube's own Y, the thumb-to-
// fingers pinch direction) by --gait_delta, then walks index/middle/ring back
// to their starting configurations one at a time, then rotates again. Each
// regrasp restores the finger travel budget the rotation consumed, so N cycles
// accumulate N*--gait_delta of cube rotation with no finger ever having to
// reach further than one cycle's worth.
//
// Why the cycle is exactly repeatable: faces 3/4 have normals +/-cube-Y, and
// rotation about cube-Y leaves those normals fixed. The contact faces
// therefore occupy the SAME world plane at every angle — only the material
// points on them rotate. So each fingertip ends every rotate phase at the same
// world point and returns to the same world point after every regrasp, which
// makes the regrasp destination just that finger's t=0 configuration
// (q_contact's own segment) on every cycle, with no per-cycle IK.
//
// The thumb never moves: it sits at the center of the +Y face (c_off = 0), so
// the rotation axis passes through its contact and it only spins in place.
// That matters beyond convenience — the thumb is the sole opposing contact,
// and releasing it while all three others are on one face would drop the cube.
DEFINE_bool(gait, false,
            "Run the rotate-then-regrasp gait. Implies the 4-finger triangle "
            "topology (as --release_middle builds it) and suppresses the "
            "one-shot --release_finger chain, which this replaces with a "
            "repeatable cycle. Requires --exec_mode=osc and "
            "--track_cube_contact.");
DEFINE_double(gait_delta, 0.7854,
              "Cube rotation per gait cycle (rad, about the grasp axis). "
              "Default 45 deg: measured single-grasp tracking delivers 44.1 "
              "of a commanded 45.3 deg with 5.7 deg of off-axis tilt and "
              "4.6 mm drift, so two cycles make a 90 deg quarter turn. The "
              "shortfall is a near-constant ~1.2 deg regardless of angle "
              "(a friction/stiffness deadband, not a range limit), and it "
              "does not accumulate: each cycle re-derives its target from "
              "the absolute commanded angle, not from where the cube got to.");
DEFINE_int32(gait_cycles, 2,
             "How many rotate+regrasp cycles to run. 2 x --gait_delta=45deg "
             "= a 90 deg quarter turn, which brings a side face to the top.");
DEFINE_double(gait_rotate_duration, 4.0,
              "Min-jerk ramp time (s) for one cycle's rotation. Same angular "
              "rate as the validated 45 deg step test at the default 45 deg "
              "--gait_delta.");
DEFINE_string(gait_scheme, "triangle",
              "Which gait to run:\n"
              "  triangle - all FOUR fingers grip throughout (index/middle/"
              "ring on -Y, thumb on +Y). Rotation happens on four contacts "
              "and each cycle regrasps ring, middle, index in turn.\n"
              "  relay    - rotation happens on THREE (index/middle/thumb, "
              "the layout measured at 97% tracking); ring is off the cube "
              "and comes down only to hold while index and middle step "
              "round, then lifts again. No finger ever crosses the face's "
              "centre line, which is what put ring out of reach in the "
              "triangle scheme.");
DEFINE_double(relay_ring_hold_z, 0.0,
              "relay: where on the -Y face ring holds, as a height (m) above "
              "the face centre. Negative is below.\n"
              "0.0 puts it dead centre, directly opposite the thumb, which is "
              "the tidiest two-point pinch and the only point that is "
              "rotation-invariant. It is also the WORST place to resist the "
              "cube unwinding: a contact on the rotation axis has no lever "
              "arm about that axis. With ring and thumb both centred, a leg "
              "leaves just one off-axis finger holding, and the rotation "
              "wound in against friction lets go — measured as 31 deg of "
              "backspin in 70 ms during index's leg.\n"
              "Offsetting ring gives a second off-axis contact, opposite the "
              "remaining one, so the pair forms a couple that resists twist. "
              "The cost is that ring's footprint is no longer the same point "
              "every cycle, which nothing depends on: its leg IK solves "
              "against the measured cube pose anyway.");
DEFINE_double(relay_ring_press, 0.001,
              "relay: how far past the -Y face (m) ring's holding target sits, "
              "so touching down produces real contact force rather than a "
              "graze. index/middle inherit their seating from the reach "
              "phase, where full-hand collision pushes them into place; ring "
              "arrives cold and has only this to press it home. Too small and "
              "the contact force never crosses "
              "--contact_force_thresh, so the ENGAGE leg times out with the "
              "fingertip resting on the cube. Too LARGE and ring drives into "
              "the cube and knocks it: this was briefly 3 mm while a "
              "contact-detection bug was misread as weak contact, and the "
              "touchdown visibly jerked the cube.");
DEFINE_double(relay_ring_retract, 0.03,
              "relay: how far outside the -Y face (m) ring parks while it is "
              "not holding. Far enough to be clear of the rotating cube, "
              "near enough that its approach is short.");
DEFINE_bool(gait_realign, true,
            "After the cycles finish, walk index/middle/ring back to their "
            "ORIGINAL cube-frame footprints, each finger to its own vertex, "
            "so the contact triangle's base is parallel to the cube's base "
            "edge again. The gait leaves them rotated: every cycle walks a "
            "footprint backwards by --gait_delta, so a 90 deg turn leaves the "
            "triangle turned 90 deg within the face. Cube is held still "
            "throughout; this moves fingers only. Ring's leg is the demanding "
            "one (it must cross the face), so watch the setup probe's "
            "residuals before trusting it.");
DEFINE_bool(gait_log, true,
            "One consolidated status line every --gait_log_period seconds "
            "while --gait runs, replacing the scattered per-relin and "
            "per-solve chatter. Reports, in one place: which cycle and gait "
            "state, which finger is moving and what kind of leg it is, how "
            "far that fingertip still is from its target and whether it is "
            "touching yet, the commanded vs achieved cube rotation, the "
            "off-axis tilt, the per-axis drift, and which fingers C3 is "
              "currently modelling as gripping.");
DEFINE_bool(gait_handoff_debug, false,
            "Print OSC position error, torque-component norms and per-finger "
            "C3 normal forces every 5 ms around a relay ring handoff.");
DEFINE_double(gait_log_period, 0.02,
              "Seconds between --gait_log lines. Was 0.25 — too coarse to "
              "see a fast transient (a jerk plays out in tens of ms), only "
              "enough for a slow trend. --jerk_log catches the exact tick "
              "regardless of this; this just controls how much of the "
              "surrounding context comes along for free without re-running "
              "at an even tighter period by hand.");
DEFINE_bool(jerk_log, true,
            "Flag a sudden jump in the cube's linear acceleration, checked "
            "every control tick — independent of --gait_log_period, so it "
            "catches the exact tick something abrupt happens rather than "
            "the nearest sampled one. Acceleration (one finite difference "
            "of the cube's own SIMULATED velocity, a clean, directly "
            "tracked signal), not literal jerk (which would need a second "
            "difference of that, or three of position — amplifying "
            "whatever numerical noise the contact solver already has). A "
            "real step in commanded force or torque shows up as a step in "
            "acceleration just as plainly. Prints once per rising edge "
            "(armed again once it drops back below threshold), with full "
            "context — gait state/leg, cmd/got/off/drift, current grip — "
            "so the line reads standalone.");
DEFINE_double(jerk_accel_thresh, 3.0,
              "Cube linear-acceleration threshold (m/s^2) --jerk_log flags "
              "on. Meant to sit above ordinary simulation/contact noise and "
              "below what a real single-finger push can produce, so it "
              "fires on a visible shake and stays quiet through normal "
              "tracking. No run has characterized the actual noise floor "
              "yet — tune from a run's own numbers if it fires on nothing "
              "or never fires on a shake you can see.");
DEFINE_bool(legacy_log, false,
            "Restore the old high-rate diagnostics: the per-relinearization "
            "'[t=..] relin |A|=..' line, the '[track IK] N ms' timing, and "
            "the per-solve 'C3 PLAN' header. At --relin_period_steps=5 and "
            "--track_ik_period_steps=5 the first two print at 200 Hz each, "
            "which buries everything the gait says. Off by default; the "
            "gait's own state prints and --gait_log carry what matters.");
DEFINE_int32(gait_seek_period_steps, 5,
             "Re-solve a regrasp leg's destination against the cube's MEASURED "
             "pose once every this many control steps (control_dt=1ms), for "
             "the part of the leg after its arc spline has played out.\n"
             "The spline is aimed at the pose measured when the leg STARTED, "
             "and the cube keeps moving during the leg — fastest of all right "
             "then, since lifting a finger drops the grasp one contact "
             "lighter. Without re-aiming, the fingertip arrives beside a "
             "surface that has moved out from under it, never contacts, and "
             "the leg waits until --gait_leg_timeout.\n"
             "Cost is a full 4-point IK per update (3-6 ms measured), so this "
             "is the most expensive thing the gait does per unit time. "
             "Matches --track_ik_period_steps by default; raise it if solve "
             "time becomes the constraint. Measured cube drift is under "
             "1 mm/s, so even 100 (10 Hz) stays well inside "
             "--regrasp_touch_tol.");
DEFINE_double(gait_leg_gap, 0.3,
              "Seconds to hold still between one leg finishing and the next "
              "starting, with every contact of the moment gripping.\n"
              "Without it the two run back to back: a leg ends by rebuilding "
              "C3 with its finger added, and the next begins on the very next "
              "tick by rebuilding again with a different finger removed — two "
              "full reconstructions 1 ms apart, both cold-started, both "
              "linearizing the dynamics while the cube is still ringing from "
              "the touchdown. Measured |A| went from 122 at rest to 657 and "
              "359 across such a pair, i.e. C3 was handed a badly conditioned "
              "model and 15 ADMM iterations to solve it, and the grip force it "
              "returned kicked the cube further. This gap lets the impact die "
              "and gives C3 several clean solves at each contact set before "
              "the next change.");
DEFINE_double(gait_touch_offset, 0.003,
              "How far outside the true contact point (m) a leg's LAST "
              "waypoint sits, for any leg that ends touching the cube "
              "(ENGAGE, REGRASP — not DISENGAGE, which ends in free air). "
              "This is the standoff the final, slow approach starts from: "
              "the fast arc/transit gets the finger to within this distance "
              "of the surface, then --gait_touch_duration governs a much "
              "shorter, slower final leg from there into contact.");
DEFINE_double(gait_touch_duration, 0.4,
              "Seconds allotted to a contact-ending leg's FINAL segment — "
              "from --gait_touch_offset outside the surface, to the true "
              "contact point — added ON TOP of --regrasp_duration, which "
              "still governs everything before that. Covering a few mm in "
              "this many seconds is what makes touchdown gentle: the same "
              "cubic spline mechanism as the rest of the leg, just given a "
              "short distance and a deliberately generous slice of time, so "
              "average speed over the last stretch is far below the "
              "transit's. Was un-tunable before — touchdown speed was "
              "whatever --osc_kp's PD delivered when it slammed into the "
              "spline's frozen final knot.");
DEFINE_int32(gait_rebuild_solve_passes, 3,
             "Every rebuild_c3 call now solves the fresh object once before "
             "anything reads it (closes the zero-force window a brand-new, "
             "never-solved object would otherwise sit in). But that one "
             "solve is still just --admm_iter iterations, COLD — the first "
             "answer from a new object, replacing whatever a DIFFERENT, "
             "well-settled object had been outputting for however long it "
             "held the previous contact set. Measured: a real force step "
             "at the handoff, ~10 m/s^2 cube acceleration, ringing for "
             "~150ms before later solves settle it out.\n"
             "This runs Solve() this many times in a row on the fresh "
             "object BEFORE the executor ever sees it, with ADMM "
             "warm-start-across-solves temporarily forced on so each pass "
             "continues the previous one's consensus instead of resetting "
             "to zero — --gait_rebuild_solve_passes x --admm_iter total "
             "ADMM depth (45 at the defaults) landing on the FIRST value "
             "anything reads, instead of --admm_iter alone. Warm-start is "
             "restored to whatever --warm_start_admm says immediately "
             "after, so nothing about this object's ONGOING, per-tick "
             "behavior changes — only the one handoff instant gets more "
             "solver depth, not the steady state.");
DEFINE_double(gait_force_ramp_time, 0.2,
              "Seconds over which a newly-joined finger's grip-force "
              "feedforward ramps from 0 to full, instead of snapping to "
              "C3's freshly-computed value the instant it joins "
              "active_fingers.\n"
              "Tried more ADMM depth on the rebuild first (see "
              "--gait_rebuild_solve_passes) on the theory the step was an "
              "under-converged solve. Measured: 45 iterations produced the "
              "IDENTICAL downstream trajectory as 15 — same accel timing, "
              "same settled state, tick for tick. That rules out solver "
              "noise: the 15-iteration answer was already converged, so "
              "the step is the CORRECT optimum changing, because 3 "
              "contacts and 4 contacts genuinely have different optimal "
              "grip distributions for the same state. No amount of solving "
              "harder removes a step that isn't an error.\n"
              "So: ramp the COMMANDED value instead, at the executor, the "
              "same way --gait_touch_duration already stops the newly-"
              "landed finger's PD target from slamming straight to contact "
              "— this is the identical idea for force instead of "
              "position. Spreads the same total force change over more "
              "time, which directly caps peak acceleration regardless of "
              "why the target changed.");
DEFINE_double(gait_torque_ramp_time, 0.2,
              "Seconds to cross-fade the full applied OSC PD torque vector "
              "from its pre-handoff value to the live post-handoff value "
              "when relay adds ring to C3. The four-point IK can change the "
              "surviving fingers' joint targets too, especially index, so "
              "smoothing ring alone is insufficient.");
DEFINE_double(gait_leg_timeout, 3.0,
              "Seconds a regrasp leg may keep seeking after its arc has "
              "played out before the gait gives up and stops. Without it a "
              "finger that cannot land waits forever and the run hangs with "
              "no indication of why. On timeout the leg reports how far the "
              "tip ended up from its target and whether it was touching "
              "anything, then C3 is rebuilt at four contacts so the grasp is "
              "left intact.");
DEFINE_double(gait_hold_time, 0.5,
              "Seconds to hold after a rotation completes before starting "
              "that cycle's regrasps — lets the cube settle so the first "
              "release doesn't happen while it is still moving.");

// ── Cube start pose ──────────────────────────────────────────────────────────
// X_WC0 — the nominal cube pose every grasp target, --cube_move_* offset and
// --gait rotation is measured from. Was hardcoded at (0, 0, 0.58).
//
// Height matters for how much rotation a grasp can deliver. Rotating about the
// grasp axis sweeps each footprint around its face, and ring's carries it
// DOWNWARD: from (x=+15, z=-8) mm it descends 8.3 mm over 45 deg, which is
// what puts ring out of reach (measured: the tracking IK starts failing at
// ~37 deg with ring 1.7-2.9 mm outside its 1 mm tolerance box, while
// index/middle/thumb stay comfortable). Raising the cube shifts that whole arc
// up in the hand's workspace. It moves ALL four contacts though, so it trades
// against whatever headroom the other three have upward.
DEFINE_double(cube_start_x, 0.0, "Cube nominal start position, world x (m).");
DEFINE_double(cube_start_y, 0.0, "Cube nominal start position, world y (m).");
DEFINE_double(cube_start_z, 0.58,
              "Cube nominal start position, world z (m). See the note above "
              "on how this interacts with ring's reach during --gait.");

// ── General ──────────────────────────────────────────────────────────────────
DEFINE_bool(preview, true,
            "Pause on the two static Meshcat previews (q_pregrasp, q_contact) "
            "and wait for Enter before starting the sim. Set false for "
            "unattended runs and for any command that pipes stdout: the "
            "prompt goes to stdout, so a pipe hides it and the run looks "
            "hung when it is really blocked on std::cin.get().");
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

  if (FLAGS_release_finger != "middle" && FLAGS_release_finger != "ring") {
    throw std::runtime_error("--release_finger must be 'middle' or 'ring'.");
  }
  if (FLAGS_osc_target_source != "auto" &&
      FLAGS_osc_target_source != "ik" &&
      FLAGS_osc_target_source != "c3") {
    throw std::runtime_error(
        "--osc_target_source must be 'auto', 'ik', or 'c3'.");
  }

  // --gait reuses --release_middle's 4-finger triangle topology wholesale
  // (grasp targets, reach-phase finger count, 4-point IK selection), so turn
  // it on rather than duplicating every one of those branches behind a second
  // flag. The one-shot --release_finger chain is separately suppressed
  // wherever it is gated, since --gait replaces it with a repeatable cycle.
  if (FLAGS_gait && FLAGS_gait_scheme != "triangle" &&
      FLAGS_gait_scheme != "relay") {
    throw std::runtime_error("--gait_scheme must be 'triangle' or 'relay'.");
  }
  if (FLAGS_gait) {
    // Only the triangle scheme wants --release_middle's 4-finger topology.
    // relay grips with index/middle/thumb, which IS the default layout
    // (index/middle at x=-+20mm, z=0, symmetric about the rotation axis) —
    // exactly the 3-finger set measured at 97% single-rotation tracking.
    // Ring stays out of the reach phase there and only ever touches down as
    // a temporary holder.
    FLAGS_release_middle = (FLAGS_gait_scheme == "triangle");
    if (!FLAGS_track_cube_contact) {
      // Without it the fingers never re-solve against the rotating cube, so
      // nothing drives the rotation the gait is built to produce.
      throw std::runtime_error("--gait requires --track_cube_contact.");
    }
    if (FLAGS_gait_cycles < 1) {
      throw std::runtime_error("--gait_cycles must be >= 1.");
    }
  }

  if (FLAGS_release_middle && FLAGS_exec_mode != "osc") {
    // The task_space executor branch's force-feedforward loop still
    // iterates a hardcoded 0..3 over normal_groups/press_C, which becomes
    // an out-of-bounds read once release_middle reduces the LCS to 2
    // contacts. Only the osc branch has been made release_middle-aware.
    throw std::runtime_error(
        "--release_middle is only implemented for --exec_mode=osc.");
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

  // Filter index/middle/ring links against each other. The --release_middle
  // triangle brings these three fingertips close together on one face, and
  // unlike a finger-cube contact, a finger-finger collision is invisible to
  // the LCS/C3 model (only finger-cube pairs are ever in contact_pairs) —
  // SAP would still resolve real impulses from it every step, the same
  // class of problem the thumb/palm filter above already exists to prevent.
  // Thumb is untouched (opposite face, never close enough to matter).
  // Harmless when --release_middle is off too: index/middle sit far enough
  // apart there that this filter just never activates.
  {
    std::vector<GeometryId> index_geoms, middle_geoms, ring_geoms;
    for (const char* name :
         {"link_0", "link_1", "link_2", "link_3", "link_3_tip"}) {
      const auto& g = sim_plant.GetCollisionGeometriesForBody(
          sim_plant.GetBodyByName(name, sim_allegro));
      index_geoms.insert(index_geoms.end(), g.begin(), g.end());
    }
    for (const char* name :
         {"link_4", "link_5", "link_6", "link_7", "link_7_tip"}) {
      const auto& g = sim_plant.GetCollisionGeometriesForBody(
          sim_plant.GetBodyByName(name, sim_allegro));
      middle_geoms.insert(middle_geoms.end(), g.begin(), g.end());
    }
    for (const char* name :
         {"link_8", "link_9", "link_10", "link_11", "link_11_tip"}) {
      const auto& g = sim_plant.GetCollisionGeometriesForBody(
          sim_plant.GetBodyByName(name, sim_allegro));
      ring_geoms.insert(ring_geoms.end(), g.begin(), g.end());
    }
    sim_scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet(index_geoms),
            drake::geometry::GeometrySet(middle_geoms)));
    sim_scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet(index_geoms),
            drake::geometry::GeometrySet(ring_geoms)));
    sim_scene_graph.collision_filter_manager().Apply(
        drake::geometry::CollisionFilterDeclaration().ExcludeBetween(
            drake::geometry::GeometrySet(middle_geoms),
            drake::geometry::GeometrySet(ring_geoms)));
    std::cout << "[setup] finger-finger collision filter APPLIED: "
                 "index/middle/ring excluded from colliding with each "
                 "other.\n";
  }

  // Isolate the cube: filter it against EVERY hand geometry except the four
  // fingertips this file knows how to place — index/middle/thumb always,
  // plus ring (link_11_tip) whenever --release_middle reaches it to the +X
  // face during the initial reach phase (see n_grasp_fingers). This makes
  // the physics world match the LCS model world by construction: ring joins
  // contact_pairs (via rebuild_c3) at the same handoff where it starts
  // physically touching, so there's no window where it's a real,
  // LCS-unmodeled contact. See grasp_questions_answered.html §Q2.
  //
  // CAVEAT: the reach phase relies on full-hand collision to seat the fingers,
  // so this can stop index/middle from ever reaching the cube. Gated behind
  // --isolate_cube for A/B testing.
  if (FLAGS_isolate_cube) {
    const std::array<std::string, 4> tip_names{
        "link_3_tip", "link_7_tip", "link_15_tip", "link_11_tip"};
    std::vector<GeometryId> non_tip_geoms;
    for (BodyIndex bi : sim_plant.GetBodyIndices(sim_allegro)) {
      const auto& body = sim_plant.get_body(bi);
      if (std::find(tip_names.begin(), tip_names.end(), body.name()) !=
          tip_names.end())
        continue;  // keep these fingertips able to hit the cube
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
                 "with the 4 fingertips (" << non_tip_geoms.size()
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

  // Ring's guessed fingertip-surface point (--ring_tip_surface_offset_z),
  // same idea as the /tip_frame/* triads above but a small cube instead of
  // a triad — easier to eyeball as a single point against the rendered
  // fingertip. Starts at the SAME offset already found for index/middle/
  // thumb (ring is the same finger module), independently adjustable if it
  // turns out not to match. --release_middle only.
  if (FLAGS_release_middle) {
    meshcat->SetObject("/tip_frame/ring_surface",
                       drake::geometry::Box(0.004, 0.004, 0.004),
                       drake::geometry::Rgba(1.0, 1.0, 0.0, 1.0));
  }

  const drake::geometry::Rgba kRed(1.0, 0.0, 0.0, 1.0);
  meshcat->SetObject("/grasp/index",  drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/middle", drake::geometry::Sphere(0.006), kRed);
  meshcat->SetObject("/grasp/thumb",  drake::geometry::Sphere(0.006), kRed);
  // Same dot, same size/color as the other three — where ring is meant to
  // touch (base-right of the triangle, -Y face). Only meaningful once
  // --release_middle has ring reaching for it, so only created in that
  // mode. (The separate green /grasp_tri/* preview markers from when this
  // was still just a proposal are gone now — /grasp/index,middle,ring
  // themselves show the real triangle target, making them redundant.)
  if (FLAGS_release_middle) {
    meshcat->SetObject("/grasp/ring", drake::geometry::Sphere(0.006), kRed);
  }
  // Where ring is currently supposed to touch, under --gait_scheme=relay.
  // A separate marker from /grasp/ring above (that one's the STATIC
  // triangle target under --release_middle; this one tracks a single,
  // moving point — relay_ring_hold_C mapped through the cube's CURRENT
  // pose — so it visibly rides around with the cube as it turns). Cyan to
  // stay distinct from the red grasp dots and the yellow tip-frame probes.
  // Created once here; update_markers below moves it and toggles
  // visibility every tick.
  if (FLAGS_gait && FLAGS_gait_scheme == "relay") {
    meshcat->SetObject("/gait/ring_touch_target", drake::geometry::Sphere(0.006),
                       drake::geometry::Rgba(0.0, 1.0, 1.0, 1.0));
  }

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
  const GeometryId lcs_ring_geom =
      lcs_plant.GetCollisionGeometriesForBody(
          lcs_plant.GetBodyByName("link_11_tip", lcs_allegro))[0];

  // Indexed by physical finger (0=index,1=middle,2=thumb,3=ring) — for
  // building whichever subset of contact_pairs is currently active (see
  // rebuild_c3 below). Fixed: all 4 geoms always exist, only which ones are
  // IN contact_pairs varies. Ring (--release_middle only, once triggered)
  // is the 4th entry.
  const std::array<GeometryId, 4> lcs_finger_geoms{
      lcs_index_geom, lcs_middle_geom, lcs_thumb_geom, lcs_ring_geom};

  // Mutable: normally all 3 pairs, temporarily 2 once --release_middle
  // triggers. rebuild_c3 (below) rebuilds this from active_fingers. Read by
  // BOTH the one-time (re)construction and the steady-state per-tick relin
  // call further down, so both always see the currently-active set.
  std::vector<SortedPair<GeometryId>> contact_pairs{
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
  // Index/middle's lateral (X) and vertical (Z) position on the -Y face:
  // normally (a,0)/(b,0) — same height, no triangle — but when
  // --release_middle is set, they instead sit at the triangle's base-left/
  // apex positions (ring will mirror index at base-right — see
  // ring_target below), forming a triangle with ring. Used from the very
  // start (reach phase, q_contact, q_pregrasp), not just after the
  // release trigger — so there's no separate jump for the two fingers
  // --release_finger DIDN'T pick; only the picked one retracts (see
  // q_release_middle/q_regrasp_ring).
  const double index_x =
      FLAGS_release_middle ? -FLAGS_release_middle_tri_spread : a;
  const double index_z =
      FLAGS_release_middle ? FLAGS_release_middle_tri_base_z : 0.0;
  const double middle_x = FLAGS_release_middle ? 0.0 : b;
  const double middle_z =
      FLAGS_release_middle ? FLAGS_release_middle_tri_apex_z : 0.0;
  // How many fingers take part in the initial reach phase (pregrasp spline,
  // arrived[]/touching[] latch, handoff). 3 normally; 4 when
  // --release_middle, so ring reaches and establishes contact on +X
  // alongside index/middle/thumb, instead of appearing in the LCS later
  // with no physical reach behind it. Referenced throughout the reach/
  // handoff logic below in place of a hardcoded 3.
  const int n_grasp_fingers = FLAGS_release_middle ? 4 : 3;
  const RigidTransform<double> X_WC0(
      RotationMatrix<double>(),
      Vector3d(FLAGS_cube_start_x, FLAGS_cube_start_y, FLAGS_cube_start_z));
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

  // 4th entry (ring, link_11_tip) is --release_middle only, but harmless to
  // always populate — every OTHER existing loop over tip_bodies/tip_names
  // is hardcoded `for i<3` and simply never reaches index 3.
  const std::array<std::string, 4> tip_names{"link_3_tip", "link_7_tip",
                                             "link_15_tip", "link_11_tip"};
  std::array<BodyIndex, 4> tip_bodies;
  for (int i = 0; i < 4; ++i)
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
  q_contact_targets << X_WC0 * Vector3d(index_x, -(h_cube - FLAGS_penetration_index_middle), index_z),
                       X_WC0 * Vector3d(middle_x, -(h_cube - FLAGS_penetration_index_middle), middle_z),
                       X_WC0 * Vector3d(c_off, h_cube - FLAGS_penetration_thumb, 0);

  // Ring's contact/pregrasp points: base-right of the triangle on the -Y
  // face, mirroring index's base-left position (see index_x/index_z
  // above) — index and ring level (the base), middle above (the apex).
  // Same --penetration_index_middle depth convention as index/middle,
  // since ring is now doing the same kind of face-press they are (no
  // longer a separate --release_middle_ring_penetration for a different
  // face). Computed here, not just at the --release_middle trigger,
  // because ring now reaches and establishes contact during the SAME
  // initial reach phase as the other three when --release_middle is set
  // — see n_grasp_fingers below.
  const Vector3d ring_target = X_WC0 * Vector3d(
      FLAGS_release_middle_tri_spread,
      -(h_cube - FLAGS_penetration_index_middle), FLAGS_release_middle_tri_base_z);

  // ── Contact footprints, in the CUBE's own frame ──────────────────────────
  // Indexed like finger_start/tip_bodies: 0=index, 1=middle, 2=thumb, 3=ring.
  // Single source of truth for "where on the cube is each finger touching":
  // resolve_contact_ik maps these through the current cube pose to get world
  // IK targets, and both the --release_finger chain and the --gait cycle
  // update an entry when that finger re-establishes contact somewhere new.
  // Previously this lived as scattered index_x/index_z/middle_z/c_off reads
  // plus the index_z_live/middle_z_live patch inside resolve_contact_ik,
  // which could only express the two specific moves the release chain made.
  //
  // Rotation about the grasp axis (cube Y) leaves every one of these on its
  // own face: face 3/4's normals are +/-cube-Y, which R_y fixes, so only the
  // in-face (x,z) part ever changes. That is what lets --gait update a
  // footprint with a plain R_y and know the finger is still on its face.
  std::array<Vector3d, 4> footprint_C{
      Vector3d(index_x, -(h_cube - FLAGS_penetration_index_middle), index_z),
      Vector3d(middle_x, -(h_cube - FLAGS_penetration_index_middle), middle_z),
      Vector3d(c_off, h_cube - FLAGS_penetration_thumb, 0.0),
      Vector3d(FLAGS_release_middle_tri_spread,
               -(h_cube - FLAGS_penetration_index_middle),
               FLAGS_release_middle_tri_base_z)};
  // The t=0 footprints, kept unmodified. --gait's regrasp returns each
  // fingertip to the WORLD point it started at, whose hand configuration is
  // exactly q_contact's own segment for that finger — see the gait state
  // machine. Retained separately so that stays true after footprint_C has
  // been rotated by a cycle.
  const std::array<Vector3d, 4> footprint_C0 = footprint_C;

  // Ring's holding footprint under relay: the centre of the -Y face. That
  // point lies ON the rotation axis, so it is the same point in world and in
  // cube frame at every angle — ring lands on the identical spot every cycle
  // and never needs repositioning. It also sits directly opposite the thumb,
  // making the cleanest two-point pinch available while index and middle are
  // both away. Its one weakness is that two holders on the axis have no
  // lever arm about it, so they resist the cube spinning by friction alone.
  const Vector3d relay_ring_hold_C(
      0.0, -(h_cube - FLAGS_relay_ring_press), FLAGS_relay_ring_hold_z);
  // Park directly out from the hold point, so ring approaches and leaves
  // along the face normal instead of sweeping across the face.
  const Vector3d relay_ring_park_C(
      0.0, -(h_cube + FLAGS_relay_ring_retract), FLAGS_relay_ring_hold_z);
  // 1cm outside, same convention as the other 3 fingers' pregrasp.
  const Vector3d ring_pregrasp_target = X_WC0 * Vector3d(
      FLAGS_release_middle_tri_spread, -(h_cube + 0.01),
      FLAGS_release_middle_tri_base_z);
  // Ring's regrasp destination (--release_finger=ring): same X as
  // ring_target, raised to middle's Z height. Declared here (not just
  // inline where q_regrasp_ring is solved below) because the main loop's
  // touch-detection also needs it, as a world-frame point to confirm ring
  // actually got there — see FLAGS_regrasp_touch_tol below.
  const Vector3d ring_regrasp_target = X_WC0 * Vector3d(
      FLAGS_release_middle_tri_spread,
      -(h_cube - FLAGS_penetration_index_middle), middle_z);
  // Arc midpoint for the regrasp path: same X as ring_regrasp_target
  // (old and new points already share X — the whole move is a Y/Z arc),
  // pulled outward past the face by --regrasp_arc_clearance, at the Z
  // height halfway between the old and new points. q_regrasp_ring_mid
  // (solved below) is this midpoint's hand config — the 2nd of 3 spline
  // knots the main loop builds at release time.
  const Vector3d ring_regrasp_mid_target = X_WC0 * Vector3d(
      FLAGS_release_middle_tri_spread,
      -(h_cube + FLAGS_regrasp_arc_clearance),
      0.5 * (FLAGS_release_middle_tri_base_z + middle_z));
  // Middle's regrasp destination, chained after ring's: once ring rejoins
  // C3 (finger_rejoined), middle loosens and moves DOWN to index's
  // height. Same X as middle's current position (middle_x, unchanged —
  // this move is a Y/Z arc, same shape as ring's) but Z lowered to the
  // triangle-base height index already sits at.
  const Vector3d middle_regrasp_target = X_WC0 * Vector3d(
      middle_x, -(h_cube - FLAGS_penetration_index_middle),
      FLAGS_release_middle_tri_base_z);
  // Arc midpoint for middle's regrasp path — same construction as ring's
  // (ring_regrasp_mid_target above).
  const Vector3d middle_regrasp_mid_target = X_WC0 * Vector3d(
      middle_x, -(h_cube + FLAGS_regrasp_arc_clearance),
      0.5 * (middle_z + FLAGS_release_middle_tri_base_z));
  // Index's regrasp destination, chained after middle's rejoin: once
  // middle is back in the LCS (middle_rejoined), index loosens and moves
  // UP to the top line (middle's original apex height — same height ring
  // moved to earlier). Same X as index's current position (index_x,
  // unchanged — same Y/Z-arc shape as ring's and middle's moves).
  const Vector3d index_regrasp_target = X_WC0 * Vector3d(
      index_x, -(h_cube - FLAGS_penetration_index_middle), middle_z);
  // Arc midpoint for index's regrasp path — same construction as ring's
  // and middle's above.
  const Vector3d index_regrasp_mid_target = X_WC0 * Vector3d(
      index_x, -(h_cube + FLAGS_regrasp_arc_clearance),
      0.5 * (index_z + middle_z));

  // Ring's own frame-origin-to-true-surface offset — confirmed by eye via
  // the /tip_frame/ring_surface cube (--ring_tip_surface_offset_{y,z}).
  // Different from tip_surface_pt (index/middle/thumb's offset, pure +Z) —
  // this is what SolveGraspIKWithRing now constrains to ring_target/
  // ring_pregrasp_target below, instead of incorrectly reusing
  // tip_surface_pt for ring the way the first version of this did.
  const Vector3d ring_surface_offset(0, FLAGS_ring_tip_surface_offset_y,
                                     FLAGS_ring_tip_surface_offset_z);

  // 4-point IK (index/middle/thumb + ring), used instead of solve_ik
  // whenever --release_middle is set. Mirrors solve_ik's own structure
  // (thumb-seed reset, FK-error check) but calls SolveGraspIKWithRing
  // instead of the 3-point SolveGraspIK.
  auto solve_ik_with_ring = [&](const char* label, const VectorXd& targets9,
                                const Vector3d& ring_pt) {
    SetThumbSeed(sim_plant, sim_allegro, &plant_ctx);
    sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
    sim_plant.SetPositions(
        &plant_ctx,
        SolveGraspIKWithRing(sim_plant, &plant_ctx, targets9, ring_pt,
                             tip_surface_pt, ring_surface_offset));
    const VectorXd q = sim_plant.GetPositions(plant_ctx, sim_allegro);
    double err = (sim_plant.EvalBodyPoseInWorld(
                      plant_ctx,
                      sim_plant.GetBodyByName("link_11_tip", sim_allegro)) *
                  ring_surface_offset - ring_pt).norm();
    for (int i = 0; i < 3; ++i)
      err += (sim_plant
                  .EvalBodyPoseInWorld(plant_ctx,
                                       sim_plant.get_body(tip_bodies[i])) *
                  tip_surface_pt -
              targets9.template segment<3>(3 * i))
                 .norm();
    std::cout << "IK " << label << " total FK error = " << err << " m\n";
    return q;
  };

  const VectorXd q_contact =
      FLAGS_release_middle
          ? solve_ik_with_ring("contact", q_contact_targets, ring_target)
          : solve_ik("contact", q_contact_targets);

  // Retracted-finger target for --release_middle/--release_finger: the
  // other three fingers stay at their established points — nothing further
  // changes for them at trigger time. Middle's segment (--release_finger=
  // middle) moves outward along the -Y face normal FROM ITS TRIANGLE
  // POSITION by --release_middle_offset — just enough to break contact —
  // and stays there forever; no recontact/regrasp for middle yet, that's
  // unbuilt. Ring's segment (--release_finger=ring) instead targets a NEW
  // point on the same face — same X (its triangle base-right spot) but
  // raised to middle's Z height, at the SAME penetration as a normal touch
  // (not just outside) — so PD naturally drives it to make contact there;
  // the main loop below detects that touch and rejoins ring to C3 (see
  // ring_touch_latched/finger_rejoined). Only the selected finger's target
  // is solved (the other stays an empty, unused vector).
  VectorXd q_release_middle, q_regrasp_ring, q_regrasp_ring_mid,
      q_regrasp_middle, q_regrasp_middle_mid, q_regrasp_index,
      q_regrasp_index_mid;
  if (FLAGS_release_middle && FLAGS_release_finger == "middle") {
    // Post-retraction the remaining 3-contact grasp is index+ring (the
    // triangle's base, now the sole -Y contacts) + thumb.
    VectorXd release_targets = q_contact_targets;
    release_targets.segment<3>(3) = X_WC0 * Vector3d(
        middle_x, -(h_cube + FLAGS_release_middle_offset), middle_z);
    q_release_middle =
        solve_ik_with_ring("release_middle", release_targets, ring_target);
  } else if (FLAGS_release_middle && FLAGS_release_finger == "ring") {
    // While released/moving, the remaining C3 grasp is index+middle (back
    // to the classic two-fingers-one-face layout) + thumb; ring itself is
    // PD-driven (not C3) toward this new point until it rejoins. Two
    // targets solved: the lifted arc midpoint (q_regrasp_ring_mid) and
    // the final new point (q_regrasp_ring) — the main loop's spline
    // routes through both, not straight to the final one.
    q_regrasp_ring_mid = solve_ik_with_ring(
        "regrasp_ring_mid", q_contact_targets, ring_regrasp_mid_target);
    q_regrasp_ring = solve_ik_with_ring("regrasp_ring", q_contact_targets,
                                        ring_regrasp_target);
    // Middle's regrasp targets, chained after ring's (see
    // middle_regrasp_target above) — solved here too, at setup time,
    // even though middle doesn't loosen until ring actually rejoins at
    // runtime, same as how q_regrasp_ring itself is precomputed before
    // release ever happens. index/thumb held at their normal
    // q_contact_targets slots; ring held at ITS new point
    // (ring_regrasp_target), not its original ring_target — by the time
    // middle loosens, ring is already there.
    VectorXd middle_regrasp_targets9_mid = q_contact_targets;
    middle_regrasp_targets9_mid.segment<3>(3) = middle_regrasp_mid_target;
    q_regrasp_middle_mid = solve_ik_with_ring(
        "regrasp_middle_mid", middle_regrasp_targets9_mid, ring_regrasp_target);
    VectorXd middle_regrasp_targets9 = q_contact_targets;
    middle_regrasp_targets9.segment<3>(3) = middle_regrasp_target;
    q_regrasp_middle = solve_ik_with_ring(
        "regrasp_middle", middle_regrasp_targets9, ring_regrasp_target);
    // Index's regrasp targets, chained after middle's (see
    // index_regrasp_target above) — solved here too, at setup time.
    // middle held at ITS new point (middle_regrasp_target, not
    // middle_x/middle_z's original apex) and ring held at ITS new point
    // (ring_regrasp_target) — by the time index loosens, both have
    // already moved there. Thumb unchanged, from q_contact_targets.
    VectorXd index_regrasp_targets9_mid = q_contact_targets;
    index_regrasp_targets9_mid.segment<3>(0) = index_regrasp_mid_target;
    index_regrasp_targets9_mid.segment<3>(3) = middle_regrasp_target;
    q_regrasp_index_mid = solve_ik_with_ring(
        "regrasp_index_mid", index_regrasp_targets9_mid, ring_regrasp_target);
    VectorXd index_regrasp_targets9 = q_contact_targets;
    index_regrasp_targets9.segment<3>(0) = index_regrasp_target;
    index_regrasp_targets9.segment<3>(3) = middle_regrasp_target;
    q_regrasp_index = solve_ik_with_ring(
        "regrasp_index", index_regrasp_targets9, ring_regrasp_target);
  }

  // ── --gait realignment reachability check ───────────────────────────────
  // Advisory only — nothing computed here reaches the controller. Every
  // regrasp destination is now solved live against the MEASURED cube pose at
  // the moment its leg starts (see solve_leg_ik), because the cube does not
  // track its commanded rotation closely enough for a t=0-anchored target to
  // still lie on its surface several cycles in.
  //
  // What this still answers, in one IK solve rather than a minute of sim, is
  // whether the END of the gait is reachable at all: --gait_realign sends
  // every finger to its OWN original cube-frame vertex, and after a 90 deg
  // turn ring's vertex sits diagonally across the face from where the gait
  // leaves it. If any finger is out of reach it is ring, and the residual
  // printed here says so before the run starts.
  if (FLAGS_gait && FLAGS_gait_realign) {
    const double total = FLAGS_gait_cycles * FLAGS_gait_delta;
    const RotationMatrix<double> R_total(
        drake::math::RollPitchYaw<double>(0.0, total, 0.0));
    VectorXd targets9(9);
    for (int i = 0; i < 3; ++i)
      targets9.segment<3>(3 * i) = X_WC0 * (R_total * footprint_C0[i]);
    // Under relay ring is a holder, not a gripping vertex: it sits at the
    // face centre, which is on the rotation axis and therefore the same
    // point at any angle. Checking it against a triangle vertex it never
    // occupies would report a failure that cannot happen.
    const Vector3d ring_pt =
        (FLAGS_gait_scheme == "relay")
            ? Vector3d(X_WC0 * relay_ring_hold_C)
            : Vector3d(X_WC0 * (R_total * footprint_C0[3]));

    SetThumbSeed(sim_plant, sim_allegro, &plant_ctx);
    sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
    bool ok = false;
    const VectorXd q_re =
        SolveGraspIKWithRing(sim_plant, &plant_ctx, targets9, ring_pt,
                             tip_surface_pt, ring_surface_offset, &ok);
    sim_plant.SetPositions(&plant_ctx, q_re);
    const char* names[4] = {"index", "middle", "thumb", "ring"};
    std::cout << "IK gait_realign (total " << total * 180.0 / M_PI << " deg) "
              << (ok ? "OK" : "INFEASIBLE") << ":";
    for (int i = 0; i < 4; ++i) {
      const Vector3d want =
          (i == 3) ? ring_pt : Vector3d(targets9.segment<3>(3 * i));
      const double e =
          (sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                         sim_plant.get_body(tip_bodies[i])) *
               (i == 3 ? ring_surface_offset : tip_surface_pt) -
           want)
              .norm();
      std::cout << "  " << names[i] << "=" << e * 1e3 << "mm";
    }
    std::cout << "\n";
    if (!ok) {
      std::cout << "  NOTE: the realign legs still run, but a finger that "
                   "cannot reach its vertex never contacts and its leg waits "
                   "forever. Raise --cube_start_z, shrink "
                   "--release_middle_tri_spread, or set --gait_realign=false.\n";
    }
  }

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
  // Built directly rather than via GetGraspPositions (cube_kinematics.h):
  // that helper hardcodes zero vertical offset for every point, so it
  // cannot represent the triangle's index_z/middle_z at all. 1cm outside
  // each face, same convention as GetGraspPositions(..., cube_size+0.02,
  // ...) used to encode via the halved margin.
  VectorXd pregrasp_targets(9);
  pregrasp_targets << X_WC0 * Vector3d(index_x, -(h_cube + 0.01), index_z),
                      X_WC0 * Vector3d(middle_x, -(h_cube + 0.01), middle_z),
                      X_WC0 * Vector3d(c_off, h_cube + 0.01, 0);
  const VectorXd q_pregrasp =
      FLAGS_release_middle
          ? solve_ik_with_ring("pregrasp", pregrasp_targets,
                               ring_pregrasp_target)
          : solve_ik("pregrasp", pregrasp_targets);

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

  // Physical finger indices (0=index,1=middle,2=thumb,3=ring) currently
  // modeled as LCS contacts. Normally {0,1,2}; {0,1,2,3} once
  // --release_middle brings ring in during the initial reach phase; drops
  // to 3 (excluding whichever --release_finger picked) once the release
  // trigger fires. normal_groups (and anything λ-derived) is indexed by
  // POSITION in this list, not by physical finger — the two only coincide
  // while all active fingers are present. Declared here (before
  // update_markers, update_force_tracking, and rebuild_c3 below) since all
  // three lambdas capture it by reference — a `[&]` capture only sees
  // names already in scope at the lambda's definition point, not ones
  // declared later in the function.
  std::vector<int> active_fingers{0, 1, 2};
  // When each finger last JOINED active_fingers — stamped by rebuild_c3,
  // read by the force-feedforward loop to ramp a newly-joined finger's grip
  // in over --gait_force_ramp_time rather than snapping to it. -1e9 so a
  // finger present from the very first rebuild reads as "joined forever
  // ago" (fraction 1) rather than triggering a ramp for no reason.
  std::array<double, 4> finger_joined_t{-1e9, -1e9, -1e9, -1e9};
  std::array<double, 4> force_crossfade_from{0.0, 0.0, 0.0, 0.0};
  bool finger_released = false;  // one-shot latch for --release_middle
  // --release_finger=ring regrasp bookkeeping (see q_regrasp_ring above).
  // ring_left_surface debounces: touching[3] can still read true for a
  // tick or two right as ring starts pulling away from its OLD point, so
  // the "re-contacted" latch below only arms once a genuine break has been
  // observed first — otherwise it could false-latch instantly at the old
  // contact instead of the new one.
  bool ring_left_surface = false;
  bool ring_touch_latched = false;
  double ring_touch_time = -1.0;
  bool finger_rejoined = false;  // true once ring is back in the LCS
  // Ring's old-point→new-point PD path (--release_finger=ring only): a
  // 3-knot (old point, lifted arc midpoint, new point) CubicShapePreserving
  // spline, built fresh at the moment of release from wherever ring's
  // actual joint config is then (not a precomputed/static target like
  // q_pregrasp/q_contact's spline, since the release moment — and so the
  // spline's start point — isn't known until runtime). Default-constructed
  // empty here; real content assigned once, at the trigger.
  PiecewisePolynomial<double> ring_regrasp_traj;
  double ring_regrasp_start_t = 0.0;

  // ── --gait state ────────────────────────────────────────────────────────
  // One cycle is kGaitRotate -> (kGaitMove, kGaitWait) x 3 fingers ->
  // next cycle, ending in kGaitDone. Declared here (ahead of
  // cube_target_pose) because the pose generator reads gait_theta_* to
  // produce the piecewise rotation reference the gait commands.
  enum GaitState { kGaitRotate, kGaitMove, kGaitDone };
  GaitState gait_state = kGaitRotate;
  // Undoes one cycle's rotation, applied to a footprint when its finger
  // re-establishes contact: the fingertip goes back to the world point it
  // started at, so in the cube's frame its footprint has walked backwards by
  // --gait_delta. Rotation preserves the footprint's radius from the face
  // center (17-20 mm here, on a 30 mm half-face), so footprints never walk
  // off the face no matter how many cycles run.
  const RotationMatrix<double> R_gait_back(
      drake::math::RollPitchYaw<double>(0.0, -FLAGS_gait_delta, 0.0));
  int gait_cycle = 0;   // which rotate+regrasp cycle we are in
  int gait_leg = 0;     // index into leg_plan for the current cycle
  // True once the cycles are done and the --gait_realign legs are running.
  // Those legs reuse kGaitMove wholesale — release, arc, touch, rejoin are
  // identical — and differ only in destination: a gait leg sends its finger
  // BACK to the world point it started the cycle at, a realign leg sends it
  // FORWARD to its own original cube-frame vertex at the final cube pose.
  bool gait_in_realign = false;
  // Destination for the leg in progress, solved live at leg entry against the
  // measured cube pose (see solve_leg_ik).
  VectorXd gait_leg_q_dest;
  Vector3d gait_leg_target_pt = Vector3d::Zero();
  Vector3d gait_leg_target_C = Vector3d::Zero();
  // Regrasp order. Thumb (2) is absent by design: it is the pivot and the
  // sole opposing contact, so releasing it would drop the cube. Every
  // intermediate state is thus thumb + 2 face-4 fingers — still an opposed
  // grasp, and the same 3-contact set the --release_finger chain already
  // holds with.
  // What each cycle does after its rotation, as (finger, leg kind) pairs.
  //
  //   triangle: regrasp ring, middle, index in turn. Four contacts hold
  //             throughout; each leg drops to three while its finger moves.
  //   relay:    ring comes DOWN to hold. Deliberately just the one leg for
  //             now — index/middle stepping round and ring lifting back off
  //             (kLegRegrasp / kLegDisengage) are cut from THIS plan while
  //             the rotate + ring-touch-down slice is isolated and proven
  //             graceful on its own; see git history (commit "Add relay
  //             gait scheme") for the full four-leg cycle to restore once
  //             this is solid. Run with --gait_cycles=1 --gait_realign=false
  //             so nothing tries to chain past this single leg.
  //
  // Thumb (2) never appears: it sits at the face centre, on the rotation
  // axis, so it only spins in place — and it is the sole opposing contact,
  // so releasing it drops the cube.
  enum LegKind { kLegRegrasp, kLegEngage, kLegDisengage };
  std::vector<std::pair<int, LegKind>> leg_plan;
  if (FLAGS_gait_scheme == "relay") {
    leg_plan = {{3, kLegEngage}};  // ring down to the face centre — only this
  } else {
    leg_plan = {{3, kLegRegrasp}, {1, kLegRegrasp}, {0, kLegRegrasp}};
  }
  // Which contacts C3 solves for during a rotation. relay rotates on three;
  // triangle rotates on all four.
  const std::vector<int> rotate_fingers =
      (FLAGS_gait_scheme == "relay") ? std::vector<int>{0, 1, 2}
                                     : std::vector<int>{0, 1, 2, 3};
  VectorXd q_ring_parked;  // solved on the first disengage, then held
  // Is ring currently one of the gripping contacts? Fixed for the triangle
  // scheme (always yes) but a live fact under relay, where ring is off the
  // cube during every rotation and only touches down to hold. The tracking
  // IK must solve 4-point exactly when it is engaged and 3-point when it is
  // not — solving for a finger that is in free space would drag the other
  // three toward a configuration built around a contact that does not exist.
  // Declared here (ahead of update_markers) rather than down by
  // resolve_contact_ik, which is its other reader, so the ring-touch-point
  // marker can read it too.
  bool ring_engaged = (n_grasp_fingers == 4);
  bool hand_tau_crossfade_active = false;
  VectorXd hand_tau_crossfade_from = VectorXd::Zero(n_hand);
  VectorXd last_tau_pd = VectorXd::Zero(n_hand);
  double gait_theta_start = 0.0;   // rad, rotation at the ramp's start
  double gait_theta_target = 0.0;  // rad, rotation at the ramp's end
  double gait_rotate_t0 = 0.0;     // t_ref at which the current ramp began
  bool gait_entered = false;       // has the current state run its entry code
  PiecewisePolynomial<double> gait_traj;  // moving finger's joint path
  double gait_traj_t0 = 0.0;
  bool gait_left_surface = false;
  bool gait_touch_latched = false;
  // When the previous leg completed. The next leg waits --gait_leg_gap past
  // this before touching anything, so the touchdown transient decays and C3
  // gets clean solves at the current contact set before it is rebuilt again.
  double gait_leg_done_t = -1e9;
  double gait_touch_time = -1.0;
  // Middle's regrasp bookkeeping, chained after ring's — mirrors every
  // ring_* variable above exactly, just triggered by finger_rejoined
  // (ring's own rejoin) instead of the --release_middle_t timer. See the
  // middle-loosen trigger block and middle-rejoin detection block below.
  bool middle_left_surface = false;
  bool middle_touch_latched = false;
  double middle_touch_time = -1.0;
  bool middle_loosened = false;   // true once middle has been dropped
  bool middle_rejoined = false;   // true once middle is back in the LCS
  PiecewisePolynomial<double> middle_regrasp_traj;
  double middle_regrasp_start_t = 0.0;
  // Index's regrasp bookkeeping, chained after middle's — mirrors every
  // middle_* variable above exactly, just triggered by middle_rejoined
  // instead of finger_rejoined.
  bool index_left_surface = false;
  bool index_touch_latched = false;
  double index_touch_time = -1.0;
  bool index_loosened = false;   // true once index has been dropped
  bool index_rejoined = false;   // true once index is back in the LCS
  PiecewisePolynomial<double> index_regrasp_traj;
  double index_regrasp_start_t = 0.0;

  auto update_markers = [&](const RigidTransform<double>& X_WC) {
    // index/middle computed directly (index_x/z, middle_x/z), not via
    // GetGraspPositions (cube_kinematics.h, shared by other binaries) —
    // that helper hardcodes zero vertical offset, so it can't show the
    // triangle's apex/base height. No --penetration_* inset here, matching
    // this block's existing convention of showing the nominal face point,
    // not the tiny IK inset.
    meshcat->SetTransform(
        "/grasp/index",
        RigidTransform<double>(X_WC * Vector3d(index_x, -h_cube, index_z)));
    meshcat->SetTransform(
        "/grasp/middle",
        RigidTransform<double>(X_WC * Vector3d(middle_x, -h_cube, middle_z)));
    meshcat->SetTransform(
        "/grasp/thumb",
        RigidTransform<double>(X_WC * Vector3d(c_off, h_cube, 0)));
    if (FLAGS_release_middle) {
      // Ring's dot: normally base-right of the triangle, mirroring index
      // (same -Y face, same convention as index/middle above). Once
      // --release_finger=ring has actually fired, ring is off the cube —
      // the dot instead shows the new desired point: same X (its original
      // base-right spot) but raised to middle's Z height. Visual-only for
      // now, not wired into any real IK/LCS/finger motion.
      const bool ring_relocated =
          finger_released && FLAGS_release_finger == "ring";
      const double ring_dot_z =
          ring_relocated ? middle_z : FLAGS_release_middle_tri_base_z;
      meshcat->SetTransform(
          "/grasp/ring",
          RigidTransform<double>(X_WC * Vector3d(
              FLAGS_release_middle_tri_spread, -h_cube, ring_dot_z)));
      // Guessed fingertip-surface cube: ring's tip BODY pose, offset along
      // its own local frame by --ring_tip_surface_offset_{y,z} — a
      // mechanical property of the finger itself, independent of which
      // face it's targeting, so this doesn't change with the topology.
      meshcat->SetTransform(
          "/tip_frame/ring_surface",
          sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                        sim_plant.get_body(tip_bodies[3])) *
              RigidTransform<double>(
                  Vector3d(0, FLAGS_ring_tip_surface_offset_y,
                          FLAGS_ring_tip_surface_offset_z)));
    }
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

    // /gait/ring_touch_target: where ring is either currently pressing or
    // about to. Visible from the moment its ENGAGE leg starts flying toward
    // the face (leg_plan[gait_leg] == {3, kLegEngage}) through the whole
    // hold (ring_engaged, which stays true across index/middle's own legs
    // in between) — and gone the instant DISENGAGE completes and
    // ring_engaged drops back to false. Position tracks the CURRENT cube
    // pose (X_WC, this function's argument), so the dot visibly rides
    // around with the cube as it turns rather than sitting fixed in world
    // space.
    if (FLAGS_gait && FLAGS_gait_scheme == "relay") {
      const bool ring_leg_flying =
          gait_state == kGaitMove && !leg_plan.empty() &&
          leg_plan[gait_leg].first == 3 &&
          leg_plan[gait_leg].second == kLegEngage;
      const bool ring_target_visible = ring_engaged || ring_leg_flying;
      meshcat->SetProperty("/gait/ring_touch_target", "visible",
                           ring_target_visible);
      if (ring_target_visible)
        meshcat->SetTransform(
            "/gait/ring_touch_target",
            RigidTransform<double>(X_WC * relay_ring_hold_C));
    }
  };

  auto preview = [&](const char* label) {
    update_markers(X_WC0);
    sim_diagram->ForcedPublish(sim_ctx);
    std::cout << "\n=== STATIC PREVIEW: " << label << " ===\n";
    if (!FLAGS_preview) return;
    std::cout << "Press Enter to continue...\n";
    std::cin.get();
  };

  sim_plant.SetPositions(&plant_ctx, sim_allegro, q_pregrasp);
  sim_plant.SetPositions(&plant_ctx, sim_cube, q_cube0);
  sim_plant.SetVelocities(&plant_ctx, sim_allegro, VectorXd::Zero(n_hand));
  sim_plant.SetVelocities(&plant_ctx, sim_cube, VectorXd::Zero(6));
  preview("q_pregrasp  (1 cm outside cube faces — sim starts here)");

  sim_plant.SetPositions(&plant_ctx, sim_allegro, q_contact);
  preview(FLAGS_release_middle
              ? "q_contact  (index/middle/ring triangle on -Y face, thumb "
                "on +Y — C3 target configuration)"
              : "q_contact  (on cube faces — C3 target configuration)");

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
  // n_lambda/n_z/n_contacts/normal_groups are mutable (not const): once
  // --release_middle triggers, rebuild_c3 (below) recomputes all four for
  // the reduced contact set, since C3's dimensions are fixed at
  // construction and must be rebuilt from scratch to change contact count
  // (UpdateLCS alone cannot resize them).
  int n_lambda = LCSFactory::GetNumContactVariables(
      GetContactModelMap().at(lcs_opts.contact_model), 3,
      FLAGS_num_friction_directions);
  // C3+ augments z with an explicit η variable (n_lambda extra), so its
  // z-size is n_x + n_u + 2·n_lambda (plain C3/C3QP would be n_x + n_u + n_lambda).
  int n_z = n_x + n_u + 2 * n_lambda;

  // Model-specific bookkeeping, reused by the force reference and the λ_n
  // diagnostic below.
  int n_contacts = lcs_opts.num_contacts.value();
  std::vector<std::vector<int>> normal_groups = NormalForceGroups(
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

  // active_fingers/finger_released now declared earlier, right before
  // update_markers — see the comment there.

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
  double next_rot_log_t = 0.0;  // --rot_log cadence, see the loop's tail
  double next_gait_log_t = 0.0;  // --gait_log cadence, same place
  // --jerk_log: previous tick's cube linear velocity (to finite-difference
  // into acceleration), whether that previous sample exists yet (skip the
  // very first tick, no baseline to diff against), and whether the LAST
  // check was already above threshold (so a sustained event prints once,
  // on the rising edge, rather than every tick it stays high).
  Vector3d v_cube_prev = Vector3d::Zero();
  bool v_cube_prev_valid = false;
  bool jerk_active = false;
  // When the most recent REGULAR (not rebuild_c3's own) relin/solve landed —
  // read by --jerk_log to show whether a jerk lines up with one of C3's
  // normal per-tick updates or neither, discriminating "this is the ongoing
  // 25Hz/200Hz solve cadence" from "this is something else entirely."
  double last_relin_t = -1e9;
  double last_solve_t = -1e9;
  // --osc_kd's velocity feedforward: q_des from the PREVIOUS tick (to
  // finite-difference), the filtered qdot_des estimate itself, and whether
  // a previous sample exists yet. See --osc_qd_filter_tau's doc for why
  // this is filtered rather than a raw finite difference.
  VectorXd q_des_prev = VectorXd::Zero(n_hand);
  VectorXd qd_des_filt = VectorXd::Zero(n_hand);
  bool qd_des_filt_valid = false;
  VectorXd tau_hand_prev_debug = VectorXd::Zero(n_hand);
  bool tau_hand_prev_debug_valid = false;

  // 4th entry (ring) only ever latches when n_grasp_fingers==4
  // (--release_middle); the loops below are bounded by n_grasp_fingers, not
  // hardcoded 3, so ring's entries are simply never touched otherwise.
  std::array<bool, 4>    arrived{false, false, false, false};
  std::array<double, 4>  arrived_time{-1.0, -1.0, -1.0, -1.0};  // set when arrived[i] latches
  // 4th entry (8) is ring's joint start — link_8..link_11, see
  // cube_kinematics.h's finger table. --release_middle only; every other
  // existing loop over finger_start is hardcoded `for i<3`.
  const std::array<int, 4> finger_start{0, 4, 12, 8};  // idx, mid, thu, ring

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
    // normal_groups is indexed by POSITION IN active_fingers (LCS/λ order),
    // not by physical finger — the two only coincide when all 3 fingers are
    // active. alpha[] IS indexed by physical finger (index=alpha_m,
    // middle=alpha_m, thumb=2*alpha_m), so af and finger=active_fingers[af]
    // are both needed here.
    for (int k = 0; k < FLAGS_N; ++k) {
      for (size_t af = 0; af < active_fingers.size(); ++af) {
        const int finger = active_fingers[af];
        const std::vector<int>& g = normal_groups[af];
        for (int a : g) {
          for (int b : g) {
            // w*(lambda_physical-alpha)^2 expressed in internal lambda.
            W_lam[k](a, b) =
                FLAGS_w_lambda * lambda_scale * lambda_scale;
          }
          lambda_des[k](a) =
              alpha[finger] / (static_cast<double>(g.size()) * lambda_scale);
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

  // ADMM options — fixed for the whole run, independent of contact count.
  // Read by rebuild_c3 below.
  C3Options c3_opts;
  c3_opts.admm_iter  = FLAGS_admm_iter;
  c3_opts.rho_scale  = FLAGS_rho_scale;
  c3_opts.warm_start = FLAGS_warm_start;
  c3_opts.scale_lcs  = true;
  c3_opts.gamma      = 1.0;

  // (Re)construct c3 from scratch for the given active contact set. Called
  // at the original reach→C3 handoff ({0,1,2}) and, once, when
  // --release_middle triggers (drops to {0,2}). C3's dimensions (n_lambda,
  // n_z, and the cost matrices sized off n_z) are fixed at construction and
  // never resized by UpdateLCS, so changing contact count means building a
  // new object, not relinearizing the old one. x_des_base/Q_knot/c3_opts
  // are read via closure, unchanged by which fingers are active.
  auto rebuild_c3 = [&](const std::vector<int>& fingers,
                        const VectorXd& x_now, double t_now) {
    // Snapshot the physical normal force actually applied by the old
    // contact set. Surviving fingers cross-fade from this value; a newly
    // added finger is absent here and therefore starts from zero.
    std::array<double, 4> applied_before{0.0, 0.0, 0.0, 0.0};
    if (c3) {
      const std::vector<VectorXd> old_lam_plan = c3->GetForceSolution();
      if (!old_lam_plan.empty()) {
        const VectorXd old_lam0 =
            old_lam_plan[0] / c3->GetLambdaScaling();
        for (size_t af = 0; af < active_fingers.size(); ++af) {
          const int finger = active_fingers[af];
          double old_target = 0.0;
          for (int idx : normal_groups[af]) old_target += old_lam0(idx);
          const double old_ramp =
              FLAGS_gait_force_ramp_time <= 0.0
                  ? 1.0
                  : std::clamp((t_now - finger_joined_t[finger]) /
                                   FLAGS_gait_force_ramp_time,
                               0.0, 1.0);
          applied_before[finger] =
              (1.0 - old_ramp) * force_crossfade_from[finger] +
              old_ramp * old_target;
        }
      }
    }
    force_crossfade_from = applied_before;

    // Stamp every finger in the new set: the whole optimum changes with the
    // contact count, so survivors and the new arrival all ramp together.
    for (int f : fingers) finger_joined_t[f] = t_now;
    active_fingers = fingers;
    lcs_opts.num_contacts = static_cast<int>(fingers.size());
    n_lambda = LCSFactory::GetNumContactVariables(
        GetContactModelMap().at(lcs_opts.contact_model),
        static_cast<int>(fingers.size()), FLAGS_num_friction_directions);
    n_z = n_x + n_u + 2 * n_lambda;
    n_contacts = static_cast<int>(fingers.size());
    normal_groups = NormalForceGroups(FLAGS_contact_model, n_contacts,
                                      FLAGS_num_friction_directions);

    contact_pairs.clear();
    for (int f : fingers)
      contact_pairs.push_back(
          SortedPair<GeometryId>(lcs_finger_geoms[f], lcs_cube_geom));

    lcs_plant.SetPositionsAndVelocities(&lcs_ctx, x_now);
    LCS lcs_new = LCSFactory::LinearizePlantToLCS(
        lcs_plant, lcs_ctx, *lcs_plant_ad, *lcs_ctx_ad, contact_pairs,
        lcs_opts, x_now, VectorXd::Zero(n_u));
    std::cout << "  C3 rebuild: active fingers=[";
    for (size_t k = 0; k < fingers.size(); ++k)
      std::cout << fingers[k] << (k + 1 < fingers.size() ? "," : "");
    std::cout << "]  n_lambda=" << n_lambda << "  n_z=" << n_z
              << "  |A|=" << lcs_new.A()[0].norm()
              << "  |B|=" << lcs_new.B()[0].norm() << "\n";

    // Input scaling s_u is derived ONCE, from the very first linearization
    // (the original reach→C3 handoff, all 3 contacts) — kept fixed across
    // a later release-triggered rebuild so the effective torque penalty
    // (R) and input scaling don't shift at the contact-count change. !c3
    // detects "is this the first-ever construction" (c3 starts null).
    if (!c3) {
      s_u = FLAGS_input_scale > 0.0
                ? FLAGS_input_scale
                : lcs_new.A()[0].norm() / lcs_new.B()[0].norm();
    }
    {
      std::vector<MatrixXd> B_sc = lcs_new.B();
      for (auto& Bk : B_sc) Bk *= s_u;
      lcs_new.set_B(B_sc);
    }
    std::cout << "  s_u=" << s_u << "  |B| now=" << lcs_new.B()[0].norm()
              << "\n";

    const std::vector<VectorXd> x_desired(FLAGS_N + 1, x_des_base);
    const std::vector<MatrixXd> Q_vec(FLAGS_N + 1, Q_knot);
    const std::vector<MatrixXd> R_vec(
        FLAGS_N, (s_u * s_u) * FLAGS_w_R * MatrixXd::Identity(n_u, n_u));
    const std::vector<MatrixXd> G_vec(
        FLAGS_N, FLAGS_w_G * MatrixXd::Identity(n_z, n_z));
    const std::vector<MatrixXd> U_vec(
        FLAGS_N, FLAGS_w_U * MatrixXd::Identity(n_z, n_z));

    c3 = std::make_unique<C3Plus>(
        lcs_new, C3::CostMatrices(Q_vec, R_vec, G_vec, U_vec), x_desired,
        c3_opts);
    c3->SetAdmmWarmStartAcrossSolves(FLAGS_warm_start_admm);

    // Force reference — update_force_tracking() self-gates (no-op unless
    // --w_lambda>0), same unconditional-call pattern as the existing
    // do_relin block further down, so this orthogonal, pre-existing feature
    // keeps working correctly across a release-triggered rebuild now that
    // its normal_groups indexing is remapped through active_fingers.
    update_force_tracking();

    // Box constraints on the torque — belong to the C3 instance, must be
    // re-applied on every fresh construction.
    {
      const double u_bound = FLAGS_tau_max / s_u;
      for (int i = 0; i < n_u; ++i) {
        Eigen::RowVectorXd Ai = Eigen::RowVectorXd::Zero(n_u);
        Ai(i) = 1.0;
        c3->AddLinearConstraint(Ai, -u_bound, u_bound,
                                c3::ConstraintVariable::INPUT);
      }
    }

    // OSQP options (same tuning as allegro_grasp_c3.cc) — also per-instance.
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

    // A fresh C3 object's force solution is zero-initialized in its own
    // constructor (lambda_sol_ = Zero) and stays that way until Solve() is
    // called on THIS object — which otherwise wouldn't happen until the
    // next scheduled do_solve tick, up to --c3_period_steps (40 ticks,
    // ~40ms) later. In between, every finger the executor reads grip force
    // for gets zero (not "whatever it was," genuinely zero, since it's a
    // different object with its own solution vector) — then snaps to a
    // real value the instant the first real solve lands. Solving once,
    // right here, closes that gap to ~0 ticks.
    //
    // But one solve is still just --admm_iter iterations, COLD — the fresh
    // object's very first answer, handed off in place of whatever a
    // DIFFERENT, well-settled object had been outputting. Measured: still
    // a real force step at that handoff, ~10 m/s^2 of cube acceleration,
    // ringing for ~150ms before later solves settle it. So: temporarily
    // force warm-start-across-solves ON for just this object, and solve it
    // --gait_rebuild_solve_passes times in a row — each pass, with that
    // flag on, continues the PREVIOUS pass's ADMM consensus instead of
    // resetting to zero (admm_state_valid_ only goes true once a solve has
    // actually completed), so the LAST pass is landing on
    // passes x admm_iter total depth, not admm_iter alone. Then restore
    // whatever --warm_start_admm says for this object's ONGOING, per-tick
    // solves — this is depth for the ONE handoff instant, not a standing
    // change to how the object behaves afterward.
    c3->SetAdmmWarmStartAcrossSolves(true);
    for (int i = 0; i < std::max(1, FLAGS_gait_rebuild_solve_passes); ++i)
      c3->Solve(x_now);
    c3->SetAdmmWarmStartAcrossSolves(FLAGS_warm_start_admm);
  };

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
  // q_contact_live is a zero-order-hold staircase in TIME: it sits frozen
  // for --track_ik_period_steps ticks, then jumps straight to the next IK
  // solve in a single 1 ms tick. With a pure-kp OSC law (--osc_kd=0, no
  // velocity term to absorb it) that jump lands on q_des whole, producing a
  // position-error — hence torque — spike every update period, silent only
  // once the commanded rotation plateaus and successive IK solves converge
  // to the same point. This is the continuous, rotation-only jerk traced
  // via --jerk_log. Fix: ramp q_des linearly from the last solve to the new
  // one over the same period, instead of snapping — see its use at the
  // --osc_kp/--osc_kd law below. ramp_from is the value the ramp departs
  // from; ramp_t0 is when that departure started (both set only when a
  // fresh solve actually lands, in the --track_cube_contact block below).
  VectorXd q_contact_live_ramp_from = q_contact_live;
  double track_ik_ramp_t0 = 0.0;

  // Dedicated scratch context for the moving-contact IK. Standalone so the
  // solve never disturbs the live sim context (plant_ctx) or the LCS context.
  auto ik_ctx = sim_plant.CreateDefaultContext();

  // Re-solve the 3-point grasp IK so the three fingertips land on the current
  // cube faces (same per-finger penetration + tip-surface offset as the t=0
  // q_contact). Warm-started from the last solution for speed and to keep the
  // hand config from jumping between IK branches. Returns hand-only joints.
  // --track_cube_contact IK health. A failed solve used to be fed straight to
  // the executor and to C3 (the helpers print "IK failed!" but return the
  // failed iterate anyway), which at --track_ik_period_steps=5 meant a run of
  // failures became a stream of garbage position targets that osc_kp drove the
  // hand to — measured as the cube being kicked 15 deg in 60 ms, ~23x the
  // commanded rate, followed by ~0.8 s of thrashing. Self-reinforcing, too:
  // each kick moves the cube further from where the IK expects it.
  long ik_fail_count = 0;
  double last_ik_fail_print_t = -1e9;
  auto resolve_contact_ik =
      [&](const RigidTransform<double>& X_WC, double t_now) -> VectorXd {
    // Footprints come from footprint_C, so a finger that has re-established
    // contact somewhere new is tracked at its NEW point automatically. This
    // replaced an index_z_live/middle_z_live patch that hardcoded the two
    // specific moves the --release_finger chain makes; footprint_C covers
    // those and --gait's repeated rotations with the same mechanism.
    VectorXd targets(9);
    targets << X_WC * footprint_C[0], X_WC * footprint_C[1],
        X_WC * footprint_C[2];
    sim_plant.SetPositions(ik_ctx.get(), sim_allegro, q_contact_live);
    // Ring must be solved too whenever it is actually gripping, i.e. under
    // the 4-finger triangle. It used to be omitted here unconditionally
    // (3-point SolveGraspIK), which meant that with --track_cube_contact and
    // a MOVING cube, ring alone was never told the cube had moved: it held
    // its t=0 world pose and anchored the cube while the other three tried
    // to turn it. Measured effect was a commanded +9.7deg rotation coming
    // out as -10.6deg — the cube driven BACKWARDS by the stale finger. Any
    // 4-contact grasp whose cube moves needs the 4-point solve.
    bool ik_ok = false;
    const VectorXd q_full =
        ring_engaged
            ? SolveGraspIKWithRing(sim_plant, ik_ctx.get(), targets,
                                   X_WC * footprint_C[3], tip_surface_pt,
                                   ring_surface_offset, &ik_ok)
            : SolveGraspIK(sim_plant, ik_ctx.get(), targets, tip_surface_pt,
                           &ik_ok);
    if (!ik_ok) {
      // Hold the last good target rather than command the failed iterate.
      // The cube then simply lags a reference the hand cannot reach, which
      // is recoverable, instead of being thrown by a configuration jump.
      ++ik_fail_count;
      if (t_now >= last_ik_fail_print_t + 0.25) {
        last_ik_fail_print_t = t_now;
        // Per-finger residual of the FAILED iterate: the constraint left
        // furthest from its target is the one that could not be reached,
        // which is what says whether a given --gait_delta is past a
        // particular finger's workspace rather than merely hard to solve.
        sim_plant.SetPositions(ik_ctx.get(), q_full);
        const char* names[4] = {"index", "middle", "thumb", "ring"};
        int worst = 0;
        double worst_err = -1.0;
        for (int i = 0; i < n_grasp_fingers; ++i) {
          const Vector3d want =
              (i == 3) ? Vector3d(X_WC * footprint_C[3])
                       : Vector3d(targets.segment<3>(3 * i));
          const double e =
              (sim_plant.EvalBodyPoseInWorld(*ik_ctx,
                                             sim_plant.get_body(tip_bodies[i])) *
                   (i == 3 ? ring_surface_offset : tip_surface_pt) -
               want)
                  .norm();
          if (e > worst_err) {
            worst_err = e;
            worst = i;
          }
        }
        std::cout << "  [track IK] FAILED at t=" << t_now << " (" << ik_fail_count
                  << " so far) — holding last target; worst finger: "
                  << names[worst] << " off by " << worst_err * 1e3 << " mm\n";
      }
      return q_contact_live;
    }
    sim_plant.SetPositions(ik_ctx.get(), q_full);
    return sim_plant.GetPositions(*ik_ctx, sim_allegro);
  };

  // Hand configuration putting finger f's tip on target_C (a point in the
  // CUBE's frame) while the other three stay on their current footprints,
  // everything mapped through the cube pose passed in.
  //
  // --gait's regrasp legs use this against the MEASURED cube pose rather than
  // a world point precomputed at t=0. The t=0 anchoring only holds while the
  // cube tracks its commanded rotation; it does not. The cube lags ~17 deg
  // and drifts ~13 mm, and by the third cycle a t=0-anchored destination is
  // no longer on the cube's surface at all — the finger flies to it, touches
  // nothing, and the leg waits for a contact that can never happen. Solving
  // against where the cube actually IS makes every leg land on the real
  // surface no matter how far the rotation has fallen behind.
  auto solve_leg_ik = [&](const RigidTransform<double>& X_WC, int f,
                          const Vector3d& target_C, bool* ok) -> VectorXd {
    VectorXd targets(9);
    for (int i = 0; i < 3; ++i)
      targets.segment<3>(3 * i) = X_WC * (i == f ? target_C : footprint_C[i]);
    const Vector3d ring_pt = X_WC * (f == 3 ? target_C : footprint_C[3]);
    sim_plant.SetPositions(ik_ctx.get(), sim_allegro, q_contact_live);
    const VectorXd q_full = SolveGraspIKWithRing(
        sim_plant, ik_ctx.get(), targets, ring_pt, tip_surface_pt,
        ring_surface_offset, ok);
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
    // --gait overrides the --cube_motion_mode profiles entirely: instead of
    // one ramp to a fixed offset, the reference is PIECEWISE — ramp by
    // --gait_delta during a rotate phase, then hold at that angle for the
    // whole regrasp sequence, then ramp again. The state machine owns
    // gait_theta_start/target/rotate_t0; this just evaluates the min-jerk
    // ramp between them, which makes the function still a pure function of
    // t_ref and so safe for the horizon look-aheads that call it at
    // t_ref_end and per-knot tk.
    //
    // Rotation only, about the cube's own y (the grasp axis), and zero
    // velocity feedforward — the same convention the rotational part of the
    // --cube_move_* path already uses.
    if (FLAGS_gait) {
      const double T = std::max(1e-6, FLAGS_gait_rotate_duration);
      const double s = std::clamp((t_ref - gait_rotate_t0) / T, 0.0, 1.0);
      const double ramp =
          10.0 * s * s * s - 15.0 * s * s * s * s + 6.0 * s * s * s * s * s;
      const double theta =
          gait_theta_start + ramp * (gait_theta_target - gait_theta_start);
      return {X_WC0 * RigidTransform<double>(RotationMatrix<double>(
                          drake::math::RollPitchYaw<double>(0.0, theta, 0.0))),
              Vector3d::Zero()};
    }
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
    std::array<bool, 4> touching{false, false, false, false};
    for (int k = 0; k < contacts.num_point_pair_contacts(); ++k) {
      const auto& info = contacts.point_pair_contact_info(k);
      if (info.contact_force().norm() < FLAGS_contact_force_thresh) continue;
      // All four tips, NOT n_grasp_fingers. n_grasp_fingers counts the
      // fingers that take part in the REACH phase, which is 3 under
      // --gait_scheme=relay because ring joins later — but ring still
      // physically touches the cube every time it comes down to hold. Bounding
      // this loop by n_grasp_fingers made touching[3] impossible to set, so
      // relay's ENGAGE leg could never see the contact it was waiting for and
      // timed out with the fingertip 2.4 mm from target and visibly resting on
      // the cube. Detection must cover every tip that CAN touch, which is all
      // four; who is currently a planned contact is active_fingers' job.
      for (int i = 0; i < 4; ++i) {
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
      // n_grasp_fingers is 4 (includes ring) whenever --release_middle, 3
      // otherwise — ring's slot is simply never touched in the 3 case.
      for (int i = 0; i < n_grasp_fingers; ++i) {
        if (!arrived[i] && touching[i] && t > FLAGS_contact_enable_t) {
          arrived[i] = true;
          arrived_time[i] = t;
          std::cout << "[t=" << t << "] finger " << i << " arrived\n";
        }
      }

      const double tl = std::clamp(t, 0.0, traj.end_time());
      VectorXd q_tgt   = traj.value(tl).col(0);
      VectorXd qd_tgt  = traj_dot.value(tl).col(0);
      for (int i = 0; i < n_grasp_fingers; ++i) {
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
      // arrived_time[3] (ring) stays -1.0 and never wins this max() unless
      // n_grasp_fingers==4 actually latched it — harmless to always include.
      const double last_arrival_t = std::max(
          {arrived_time[0], arrived_time[1], arrived_time[2], arrived_time[3]});
      if (arrived[0] && arrived[1] && arrived[2] &&
          (!FLAGS_release_middle || arrived[3]) &&
          t >= last_arrival_t + FLAGS_handoff_settle_time) {
        std::cout << "[t=" << t
                  << "] all fingers settled → C3 handoff\n";

        // Read full state from the sim plant at this instant.
        const VectorXd x_contact =
            sim_plant.GetPositionsAndVelocities(plant_ctx);

        // Desired state: q_contact for the three grasping fingers, cube at
        // the world-frame reference q_cube0. Computed once, here, and held
        // fixed across a later release-triggered rebuild_c3 call.
        VectorXd q_des_hand = sim_plant.GetPositions(plant_ctx, sim_allegro);
        for (int i = 0; i < n_grasp_fingers; ++i)
          q_des_hand.segment(finger_start[i], 4) =
              q_contact.segment(finger_start[i], 4);
        VectorXd x_des = VectorXd::Zero(n_x);
        x_des.head(n_hand_q) = q_des_hand;
        x_des.segment(n_hand_q, 7) = q_cube0;
        // Hand and cube velocities desired = 0 (zero-initialized above).
        x_des_base = x_des;

        // Ring joins the LCS right here, at the same instant it starts
        // physically touching (n_grasp_fingers==4 means it just reached
        // and arrived alongside the other three) — not later at the
        // --release_middle trigger, so there's no window where it's a
        // real, unmodeled contact.
        rebuild_c3(
            FLAGS_release_middle ? std::vector<int>{0, 1, 2, 3}
                                 : std::vector<int>{0, 1, 2},
            x_contact, t);

        // Diagnostic-only checks that the reach phase actually seated the
        // fingers correctly (not used to gate the handoff — see
        // handoff_settle_time above for why). Read off the just-constructed
        // c3's own LCS, which — because C3Options::scale_lcs is true — is
        // scaled by AnDn_ internally; phi_contact's magnitude won't match a
        // pre-refactor log line-for-line, but its sign (<=0 = active) is
        // unaffected by a positive scale factor.
        {
          const LCS& lcs0 = c3->GetLCS();
          const VectorXd eta0 = lcs0.E()[0] * x_contact + lcs0.c()[0];
          if (FLAGS_contact_model == "stewart_and_trinkle") {
            const VectorXd phi_contact = eta0.segment(n_contacts, n_contacts);
            std::cout << "  contact gaps φ [index, middle, thumb] = "
                      << phi_contact.transpose()
                      << "  (<=0 means active, informational only)\n";
          } else {
            std::cout << "  (Anitescu: η is the cone-velocity constraint, "
                          "not a per-finger signed distance; gaps not "
                          "shown)\n";
          }
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

        phase = kC3;
        handoff_t = t;  // cube released after 0.5 s warm-up (see pin block)
      }

    } else {
      // ── Phase 2: C3 ─────────────────────────────────────────────────────
      const VectorXd x_current =
          sim_plant.GetPositionsAndVelocities(plant_ctx);

      // Decimate the expensive calls off the 1 kHz control loop. c3_iter
      // starts at 0, so the very first kC3 step always relinearizes, solves,
      // and re-tracks; thereafter each fires on its own period. Between
      // solves the cached input solution (c3->GetInputSolution) is reused
      // unchanged. do_track_ik is independent of do_relin — the live
      // --track_cube_contact IK re-solve (resolve_contact_ik) has nothing
      // to do with the LCS relinearization, they just used to share a gate
      // for convenience; --track_ik_period_steps lets it run on its own,
      // cheaper clock instead of paying full relin cost for fresh tracking.
      const bool do_relin =
          FLAGS_relinearize &&
          (c3_iter % std::max(1, FLAGS_relin_period_steps) == 0);
      const bool do_solve =
          (c3_iter % std::max(1, FLAGS_c3_period_steps) == 0);
      const bool do_track_ik =
          (c3_iter % std::max(1, FLAGS_track_ik_period_steps) == 0);
      ++c3_iter;

      if (do_relin) {
        last_relin_t = t;
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
        if (!FLAGS_contact_force_log && FLAGS_legacy_log) {
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

      // --release_middle: one-shot, at --release_middle_t seconds after the
      // cube unpins (t_ref_now is already exactly that time base). All 4
      // fingers have already been in the LCS since the original handoff
      // (they reached and arrived together — see n_grasp_fingers), so the
      // only thing that changes here is --release_finger dropping out: it
      // retracts off the face via q_release_middle/q_regrasp_ring (picked
      // up in the osc executor below); the other three have already been
      // at their triangle positions since the reach phase, so nothing
      // moves for them at trigger time.
      // !FLAGS_gait: --gait turns --release_middle on for its topology but
      // replaces this one-shot chain with its own repeatable cycle. Gating
      // the trigger alone disables the whole chain, since every later block
      // (and the executor's overrides) keys off finger_released/_rejoined.
      if (FLAGS_release_middle && !FLAGS_gait && !finger_released &&
          t_ref_now >= FLAGS_release_middle_t) {
        if (FLAGS_release_finger == "middle") {
          rebuild_c3({0, 2, 3}, x_current, t);
          std::cout << "[t=" << t << "] release_middle: middle finger "
                       "retracted, C3 now solving 3 contacts "
                       "(index, thumb, ring)\n";
        } else {  // "ring"
          rebuild_c3({0, 1, 2}, x_current, t);
          std::cout << "[t=" << t << "] release_middle: ring finger "
                       "retracted, C3 now solving 3 contacts "
                       "(index, middle, thumb)\n";
          // 3-knot arc from wherever ring actually is right now, through
          // the lifted midpoint (q_regrasp_ring_mid), to q_regrasp_ring —
          // CubicShapePreserving same as the reach phase's q_pregrasp→
          // q_contact spline (traj/traj_dot above), just with a real
          // waypoint in the middle instead of 2 knots straight-line-
          // interpolated in joint space (which has no notion of the
          // cube's geometry and can stay close to, or drag across, the
          // face the whole way). Only finger_start[3]'s segment is ever
          // read out of it (see the osc executor below). Built here, not
          // at setup time, since the start point (ring's actual config
          // right now) isn't known until release actually happens.
          // n_hand_q, not n_pos: solve_ik_with_ring (and so
          // q_regrasp_ring/q_regrasp_ring_mid) returns sim_plant.
          // GetPositions(..., sim_allegro) — the HAND-ONLY 16 dof, not
          // the full hand+cube plant vector — same convention q_contact/
          // q_release_middle already use, which is what finger_start[]
          // indexes into everywhere else.
          std::vector<MatrixXd> ring_spline_pts{x_current.head(n_hand_q),
                                                 q_regrasp_ring_mid,
                                                 q_regrasp_ring};
          ring_regrasp_traj = PiecewisePolynomial<double>::CubicShapePreserving(
              {0.0, 0.5 * FLAGS_regrasp_duration, FLAGS_regrasp_duration},
              ring_spline_pts, true);
          ring_regrasp_start_t = t;
        }
        finger_released = true;
      }

      // --release_finger=ring regrasp: while released and not yet
      // rejoined, ring is PD-driven (see the osc executor below) toward
      // q_regrasp_ring, not solved by C3 — index/middle/thumb keep running
      // under C3 the whole time, unaffected. touching[3] (computed once
      // per tick above, independent of phase) is reused here as the
      // re-contact signal, but force alone isn't enough: if the weaker
      // 3-finger grip lets the cube sag, its face can swing back into
      // contact with ring while ring is still near its OLD point early in
      // the slow-starting spline (CubicShapePreserving eases in from zero
      // velocity at both ends) — that reads as touching[3] too, and would
      // rejoin C3 back at the old spot instead of the new one.
      // ring_near_target (FK vs ring_regrasp_target, --regrasp_touch_tol)
      // requires ring to actually be AT the new point, not just touched
      // somewhere. ring_left_surface separately debounces the just-broken
      // OLD contact still reading true for a tick or two right after the
      // trigger — only a genuine break-then-remake-at-the-right-place
      // latches. Once latched and held for --regrasp_settle_time (same
      // rationale as --handoff_settle_time), ring rejoins C3.
      if (FLAGS_release_middle && FLAGS_release_finger == "ring" &&
          finger_released && !finger_rejoined) {
        if (!ring_left_surface && !touching[3]) {
          ring_left_surface = true;
        }
        const Vector3d ring_tip_pos =
            sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                          sim_plant.get_body(tip_bodies[3])) *
            ring_surface_offset;
        const bool ring_near_target =
            (ring_tip_pos - ring_regrasp_target).norm() <
            FLAGS_regrasp_touch_tol;
        if (ring_left_surface && !ring_touch_latched && touching[3] &&
            ring_near_target) {
          ring_touch_latched = true;
          ring_touch_time = t;
          std::cout << "[t=" << t << "] release_middle: ring re-contacted "
                       "the new point, settling " << FLAGS_regrasp_settle_time
                    << " s before rejoining C3\n";
        }
        if (ring_touch_latched &&
            t >= ring_touch_time + FLAGS_regrasp_settle_time) {
          // Ring's OLD position is baked into x_des_base (C3's own cost
          // target, set once at the original handoff and otherwise left
          // alone across rebuilds) and into q_contact_live/_end (the
          // --track_cube_contact executor target). Left unrefreshed, both
          // pull ring straight back to its old point the instant the osc
          // executor's override below stops overriding it (once
          // finger_rejoined) — update all three to the new point FIRST,
          // before rebuild_c3 reads x_des_base. footprint_C[3] is the
          // fourth: resolve_contact_ik now DOES track ring, so without it
          // the next track-IK solve would overwrite the other three.
          footprint_C[3] = Vector3d(FLAGS_release_middle_tri_spread,
                                    -(h_cube - FLAGS_penetration_index_middle),
                                    middle_z);
          x_des_base.segment(finger_start[3], 4) =
              q_regrasp_ring.segment(finger_start[3], 4);
          q_contact_live.segment(finger_start[3], 4) =
              q_regrasp_ring.segment(finger_start[3], 4);
          q_contact_live_end.segment(finger_start[3], 4) =
              q_regrasp_ring.segment(finger_start[3], 4);
          rebuild_c3({0, 1, 2, 3}, x_current, t);
          finger_rejoined = true;
          std::cout << "[t=" << t << "] release_middle: ring rejoined C3, "
                       "solving 4 contacts (index, middle, thumb, ring)\n";
        }
      }

      // Middle loosens, chained after ring's rejoin: once ring is back in
      // the LCS, middle drops out and arcs DOWN to index's height (same
      // rebuild/spline pattern as ring's own release above, just
      // triggered by finger_rejoined instead of --release_middle_t).
      if (FLAGS_release_middle && FLAGS_release_finger == "ring" &&
          finger_rejoined && !middle_loosened) {
        rebuild_c3({0, 2, 3}, x_current, t);
        std::cout << "[t=" << t << "] release_middle: middle finger "
                     "loosened, C3 now solving 3 contacts "
                     "(index, thumb, ring)\n";
        std::vector<MatrixXd> middle_spline_pts{x_current.head(n_hand_q),
                                                 q_regrasp_middle_mid,
                                                 q_regrasp_middle};
        middle_regrasp_traj = PiecewisePolynomial<double>::CubicShapePreserving(
            {0.0, 0.5 * FLAGS_regrasp_duration, FLAGS_regrasp_duration},
            middle_spline_pts, true);
        middle_regrasp_start_t = t;
        middle_loosened = true;
      }

      // Middle rejoin detection — mirrors the ring-rejoin block above
      // exactly: touching[1] (middle's physical index) + FK-position
      // check against middle_regrasp_target (tip_bodies[1]/tip_surface_pt
      // — middle uses the shared index/middle/thumb surface-offset
      // convention, not ring_surface_offset) + middle_left_surface
      // debounce + --regrasp_settle_time dwell, then rejoin C3 as all 4
      // contacts and refresh x_des_base/q_contact_live/_end's
      // finger_start[1] segment (same staleness fix as ring's, same
      // reason: resolve_contact_ik is 3-point-only and x_des_base is
      // otherwise left alone across rebuilds).
      if (FLAGS_release_middle && FLAGS_release_finger == "ring" &&
          middle_loosened && !middle_rejoined) {
        if (!middle_left_surface && !touching[1]) {
          middle_left_surface = true;
        }
        const Vector3d middle_tip_pos =
            sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                          sim_plant.get_body(tip_bodies[1])) *
            tip_surface_pt;
        const bool middle_near_target =
            (middle_tip_pos - middle_regrasp_target).norm() <
            FLAGS_regrasp_touch_tol;
        if (middle_left_surface && !middle_touch_latched && touching[1] &&
            middle_near_target) {
          middle_touch_latched = true;
          middle_touch_time = t;
          std::cout << "[t=" << t << "] release_middle: middle "
                       "re-contacted the new point, settling "
                    << FLAGS_regrasp_settle_time
                    << " s before rejoining C3\n";
        }
        if (middle_touch_latched &&
            t >= middle_touch_time + FLAGS_regrasp_settle_time) {
          // Replaces the old middle_z_live patch inside resolve_contact_ik.
          footprint_C[1] = Vector3d(middle_x,
                                    -(h_cube - FLAGS_penetration_index_middle),
                                    FLAGS_release_middle_tri_base_z);
          x_des_base.segment(finger_start[1], 4) =
              q_regrasp_middle.segment(finger_start[1], 4);
          q_contact_live.segment(finger_start[1], 4) =
              q_regrasp_middle.segment(finger_start[1], 4);
          q_contact_live_end.segment(finger_start[1], 4) =
              q_regrasp_middle.segment(finger_start[1], 4);
          rebuild_c3({0, 1, 2, 3}, x_current, t);
          middle_rejoined = true;
          std::cout << "[t=" << t << "] release_middle: middle rejoined "
                       "C3, solving 4 contacts (index, middle, thumb, "
                       "ring)\n";
        }
      }

      // Index loosens, chained after middle's rejoin: once middle is back
      // in the LCS, index drops out and arcs UP to the top line (same
      // rebuild/spline pattern as ring's and middle's releases above,
      // just triggered by middle_rejoined).
      if (FLAGS_release_middle && FLAGS_release_finger == "ring" &&
          middle_rejoined && !index_loosened) {
        rebuild_c3({1, 2, 3}, x_current, t);
        std::cout << "[t=" << t << "] release_middle: index finger "
                     "loosened, C3 now solving 3 contacts "
                     "(middle, thumb, ring)\n";
        std::vector<MatrixXd> index_spline_pts{x_current.head(n_hand_q),
                                                q_regrasp_index_mid,
                                                q_regrasp_index};
        index_regrasp_traj = PiecewisePolynomial<double>::CubicShapePreserving(
            {0.0, 0.5 * FLAGS_regrasp_duration, FLAGS_regrasp_duration},
            index_spline_pts, true);
        index_regrasp_start_t = t;
        index_loosened = true;
      }

      // Index rejoin detection — mirrors the ring/middle rejoin blocks
      // above exactly: touching[0] (index's physical index) + FK-position
      // check against index_regrasp_target (tip_bodies[0]/tip_surface_pt)
      // + index_left_surface debounce + --regrasp_settle_time dwell, then
      // rejoin C3 as all 4 contacts and refresh x_des_base/q_contact_live/
      // _end's finger_start[0] segment (same staleness fix as ring's and
      // middle's, same reason).
      if (FLAGS_release_middle && FLAGS_release_finger == "ring" &&
          index_loosened && !index_rejoined) {
        if (!index_left_surface && !touching[0]) {
          index_left_surface = true;
        }
        const Vector3d index_tip_pos =
            sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                          sim_plant.get_body(tip_bodies[0])) *
            tip_surface_pt;
        const bool index_near_target =
            (index_tip_pos - index_regrasp_target).norm() <
            FLAGS_regrasp_touch_tol;
        if (index_left_surface && !index_touch_latched && touching[0] &&
            index_near_target) {
          index_touch_latched = true;
          index_touch_time = t;
          std::cout << "[t=" << t << "] release_middle: index "
                       "re-contacted the new point, settling "
                    << FLAGS_regrasp_settle_time
                    << " s before rejoining C3\n";
        }
        if (index_touch_latched &&
            t >= index_touch_time + FLAGS_regrasp_settle_time) {
          // Replaces the old index_z_live patch inside resolve_contact_ik.
          footprint_C[0] = Vector3d(
              index_x, -(h_cube - FLAGS_penetration_index_middle), middle_z);
          x_des_base.segment(finger_start[0], 4) =
              q_regrasp_index.segment(finger_start[0], 4);
          q_contact_live.segment(finger_start[0], 4) =
              q_regrasp_index.segment(finger_start[0], 4);
          q_contact_live_end.segment(finger_start[0], 4) =
              q_regrasp_index.segment(finger_start[0], 4);
          rebuild_c3({0, 1, 2, 3}, x_current, t);
          index_rejoined = true;
          std::cout << "[t=" << t << "] release_middle: index rejoined "
                       "C3, solving 4 contacts (index, middle, thumb, "
                       "ring)\n";
        }
      }

      // ── --gait: rotate, walk the three -Y fingers back, repeat ──────────
      // One cycle = rotate the cube by --gait_delta with all 4 contacts
      // holding, then regrasp ring, middle and index in turn (thumb is the
      // pivot and never releases). Each regrasp returns its fingertip to the
      // world point it occupied before the rotation, which resets the travel
      // the rotation consumed and leaves the hand in exactly the
      // configuration the next cycle starts from — so the cycle repeats
      // unchanged and the cube's angle is the only thing that accumulates.
      if (FLAGS_gait && !cube_pinned && gait_state != kGaitDone) {
        // relay only: ring takes no part in the reach phase, so on the first
        // gait tick it is still wherever the 3-point grasp IK happened to
        // leave it. Solve its parked pose once, up front, so it is held
        // clear of the cube from the very first rotation rather than
        // floating unconstrained until the first disengage sets it.
        if (FLAGS_gait_scheme == "relay" && q_ring_parked.size() != n_hand_q) {
          bool park_ok = false;
          q_ring_parked = solve_leg_ik(
              CubePoseFromPositions(sim_plant.GetPositions(plant_ctx, sim_cube)),
              3, relay_ring_park_C, &park_ok);
          std::cout << "[t=" << t << "] relay: ring parked "
                    << FLAGS_relay_ring_retract * 1e3 << " mm off the face"
                    << (park_ok ? "" : " (IK INFEASIBLE)") << "\n";
        }
        const int f = leg_plan[gait_leg].first;
        const LegKind leg_kind = leg_plan[gait_leg].second;

        if (gait_state == kGaitRotate) {
          // Wait out the pin-release transient before the first rotation.
          // Releasing the pin injects ~2 deg of off-axis wobble that decays
          // over about a second; starting to turn into it would fold that
          // disturbance into cycle 1. Later cycles don't need the guard —
          // --gait_hold_time already ran after the previous rotation.
          const bool gait_ready =
              gait_cycle > 0 || t_ref_now >= FLAGS_gait_hold_time;
          if (!gait_entered && gait_ready) {
            // Absolute angles, not deltas applied to wherever the cube got
            // to — so the ~1.2deg per-cycle tracking shortfall stays a fixed
            // lag instead of compounding into the next cycle's target.
            gait_theta_start = gait_cycle * FLAGS_gait_delta;
            gait_theta_target = (gait_cycle + 1) * FLAGS_gait_delta;
            gait_rotate_t0 = t_ref_now;
            gait_entered = true;
            // Rotation runs on rotate_fingers — four for triangle, three for
            // relay. Normally the previous leg already left C3 there, so
            // this is a no-op; it matters on the very first cycle and as a
            // guard that a scheme never rotates against the wrong model.
            if (active_fingers != rotate_fingers)
              rebuild_c3(rotate_fingers, x_current, t);
            std::cout << "[t=" << t << "] gait cycle " << (gait_cycle + 1)
                      << "/" << FLAGS_gait_cycles << ": rotating "
                      << gait_theta_start * 180.0 / M_PI << " -> "
                      << gait_theta_target * 180.0 / M_PI << " deg\n";
          }
          // gait_entered, not just the clock: while the guard above is still
          // waiting, gait_rotate_t0 holds the PREVIOUS cycle's start (or 0),
          // so an unguarded test here could satisfy itself and skip straight
          // to the regrasps without ever having commanded a rotation.
          if (gait_entered &&
              t_ref_now >= gait_rotate_t0 + FLAGS_gait_rotate_duration +
                               FLAGS_gait_hold_time) {
            gait_state = kGaitMove;
            gait_leg = 0;
            gait_entered = false;
          }
        } else if (gait_state == kGaitMove) {
          // Settle between legs. A leg ends by rebuilding C3 with its finger
          // added back; starting the next one immediately would rebuild again
          // a millisecond later with a different finger removed, both times
          // linearizing a cube still ringing from the landing. Wait it out
          // with the current grasp intact — nothing needs to move.
          if (!gait_entered && t < gait_leg_done_t + FLAGS_gait_leg_gap) {
            // holding
          } else if (!gait_entered) {
            // A leg that starts from contact must leave the LCS first: the
            // grasp really is one contact lighter while the finger is in the
            // air, so C3 has to solve that problem instead of planning with
            // a contact that no longer exists. An ENGAGE leg is the
            // exception — its finger is already off the cube and absent from
            // the LCS, so there is nothing to remove.
            if (leg_kind != kLegEngage) {
              std::vector<int> remaining;
              for (int i : active_fingers)
                if (i != f) remaining.push_back(i);
              rebuild_c3(remaining, x_current, t);
            }
            // Where this leg is headed, in the CUBE's frame.
            //   REGRASP   — the footprint walks back one --gait_delta, so
            //               the fingertip returns to the world point it
            //               occupied before the rotation (or, in realign, to
            //               the finger's own original vertex).
            //   ENGAGE    — ring's holding point, the face centre.
            //   DISENGAGE — ring's parking point, clear of the face.
            const RigidTransform<double> X_WC_leg = CubePoseFromPositions(
                sim_plant.GetPositions(plant_ctx, sim_cube));
            if (leg_kind == kLegEngage) {
              gait_leg_target_C = relay_ring_hold_C;
            } else if (leg_kind == kLegDisengage) {
              gait_leg_target_C = relay_ring_park_C;
            } else {
              gait_leg_target_C =
                  gait_in_realign ? footprint_C0[f]
                                  : Vector3d(R_gait_back * footprint_C[f]);
            }
            gait_leg_target_pt = X_WC_leg * gait_leg_target_C;
            bool dest_ok = false, mid_ok = false;
            gait_leg_q_dest =
                solve_leg_ik(X_WC_leg, f, gait_leg_target_C, &dest_ok);
            // Waypoint. A REGRASP arcs: halfway between the old and new
            // footprints, pushed out past the -Y face so the tip travels
            // around the face rather than dragging across it. ENGAGE and
            // DISENGAGE instead go straight out to the parking depth above
            // the holding point, so ring approaches and leaves along the
            // face normal rather than sweeping across the face.
            Vector3d mid_C;
            if (leg_kind == kLegRegrasp) {
              mid_C = 0.5 * (footprint_C[f] + gait_leg_target_C);
              mid_C.y() = -(h_cube + FLAGS_regrasp_arc_clearance);
            } else {
              mid_C = relay_ring_park_C;
            }
            const VectorXd q_leg_mid =
                solve_leg_ik(X_WC_leg, f, mid_C, &mid_ok);
            if (!dest_ok || !mid_ok) {
              // Say so loudly: an unreachable destination means the finger
              // flies as close as it can, never contacts, and the leg hangs.
              std::cout << "  [gait] WARNING finger " << f << " leg IK "
                        << (dest_ok ? "" : "destination ")
                        << (mid_ok ? "" : "waypoint ")
                        << "INFEASIBLE — this leg will not latch\n";
            }
            // Gentle touchdown: for a leg that ends IN CONTACT (ENGAGE,
            // REGRASP — not DISENGAGE, which ends in free air and has
            // nothing to be gentle about), insert one more waypoint
            // --gait_touch_offset outside the true target and give the
            // short final hover-to-contact stretch its OWN time budget,
            // --gait_touch_duration, on top of the arc/transit above. Same
            // (x,z) as the target, y pulled outward by the offset — the
            // same "add to h_cube to stand off, subtract to press in"
            // convention mid_C and the park point already use. Distance is
            // small (mm) and gets seconds, not a fraction of one, so the
            // finger visibly slows for the last stretch rather than
            // carrying its transit speed straight into the surface.
            std::vector<double> knot_times{0.0, 0.5 * FLAGS_regrasp_duration,
                                           FLAGS_regrasp_duration};
            std::vector<MatrixXd> pts{x_current.head(n_hand_q), q_leg_mid};
            if (leg_kind != kLegDisengage) {
              Vector3d hover_C = gait_leg_target_C;
              hover_C.y() = -(h_cube + FLAGS_gait_touch_offset);
              bool hover_ok = false;
              pts.push_back(solve_leg_ik(X_WC_leg, f, hover_C, &hover_ok));
              if (!hover_ok)
                std::cout << "  [gait] WARNING finger " << f
                          << " touch-hover waypoint INFEASIBLE\n";
              knot_times.push_back(FLAGS_regrasp_duration +
                                   FLAGS_gait_touch_duration);
            }
            pts.push_back(gait_leg_q_dest);
            gait_traj = PiecewisePolynomial<double>::CubicShapePreserving(
                knot_times, pts, true);
            gait_traj_t0 = t;
            gait_left_surface = false;
            gait_touch_latched = false;
            gait_entered = true;
            const char* kind_word = leg_kind == kLegEngage    ? "engaging"
                                    : leg_kind == kLegDisengage ? "parking"
                                                                : "releasing";
            std::cout << "[t=" << t << "] gait"
                      << (gait_in_realign ? " realign" : "") << ": "
                      << kind_word << " finger " << f << ", C3 now solving "
                      << active_fingers.size() << " contacts\n";
          }
          // ── Seek: re-aim at the cube where it actually is now ───────────
          // The arc spline is open-loop, aimed at the pose measured when the
          // leg started. That is fine for clearing the face but not for
          // touching down, because the cube keeps moving during the leg —
          // and moves FASTEST here, since lifting a finger drops the grasp
          // to three contacts. Measured drift reaches ~18 mm and climbs, so
          // a finger flown to a stale point arrives beside the surface, never
          // contacts, and the leg waits forever.
          //
          // Once the arc has played out, re-solve the destination against the
          // live cube pose so the fingertip closes on the real surface. Also
          // refreshes the touch check's target, which is otherwise comparing
          // against a point the cube has left. Rate is
          // --gait_seek_period_steps; a full 4-point IK per update makes it
          // the gait's dominant per-tick cost.
          // gait_traj.end_time(), not the bare --regrasp_duration: a
          // contact-ending leg's spline now runs --gait_touch_duration
          // PAST --regrasp_duration (the gentle final approach). Using the
          // flag directly here would fire the seek mid-taper and again
          // once the taper actually ends — end_time() is whichever is
          // really the spline's last knot, DISENGAGE included (its spline
          // is unchanged, so this is a no-op there).
          if (gait_entered && t - gait_traj_t0 > gait_traj.end_time() &&
              c3_iter % std::max(1, FLAGS_gait_seek_period_steps) == 0) {
            const RigidTransform<double> X_WC_seek = CubePoseFromPositions(
                sim_plant.GetPositions(plant_ctx, sim_cube));
            bool seek_ok = false;
            const VectorXd q_seek =
                solve_leg_ik(X_WC_seek, f, gait_leg_target_C, &seek_ok);
            if (seek_ok) {
              gait_leg_q_dest = q_seek;
              gait_leg_target_pt = X_WC_seek * gait_leg_target_C;
            }
          }

          // A DISENGAGE leg ends by NOT touching, so there is nothing to
          // detect — it is done once the finger has flown clear and settled.
          // Latching it here lets the shared completion block below run
          // unchanged; the branch there skips the parts that assume contact.
          if (gait_entered && leg_kind == kLegDisengage && !gait_touch_latched &&
              t - gait_traj_t0 >= FLAGS_regrasp_duration) {
            gait_touch_latched = true;
            gait_touch_time = t;
          }

          // Re-contact detection, same two guards the --release_finger
          // chain needs: gait_left_surface debounces the just-broken OLD
          // contact still reading true for a tick or two, and the
          // position check requires f to be AT its return point rather
          // than merely touching the cube somewhere en route.
          if (!gait_left_surface && !touching[f]) gait_left_surface = true;
          const Vector3d tip_pos =
              sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                            sim_plant.get_body(tip_bodies[f])) *
              (f == 3 ? ring_surface_offset : tip_surface_pt);
          const bool near_target =
              (tip_pos - gait_leg_target_pt).norm() < FLAGS_regrasp_touch_tol;
          // A leg that cannot land must not hang the run silently. Report
          // what it was doing and how close it got, then stop the gait —
          // the grasp is left intact at four contacts.
          // gait_traj.end_time(), same reasoning as the seek trigger above:
          // a contact-ending leg's spline now ends --gait_touch_duration
          // later than --regrasp_duration alone. Without this the timeout
          // clock would start ticking mid-taper and could fire on a
          // touchdown that was simply going slowly on purpose.
          if (gait_entered && leg_kind != kLegDisengage && !gait_touch_latched &&
              t - gait_traj_t0 > gait_traj.end_time() + FLAGS_gait_leg_timeout) {
            std::cout << "[t=" << t << "] gait: finger " << f
                      << " FAILED to re-contact within "
                      << FLAGS_gait_leg_timeout << " s of arriving — tip is "
                      << (tip_pos - gait_leg_target_pt).norm() * 1e3
                      << " mm from target, touching=" << touching[f]
                      << ". Stopping the gait.\n";
            // active_fingers is left exactly as it is. The finger really is
            // off the cube, so the reduced contact set is the honest model —
            // adding it back would have C3 planning against a contact that
            // failed to form, which is the mistake this whole gait avoids.
            gait_state = kGaitDone;
          }
          // gait_entered: during the --gait_leg_gap hold this leg has not
          // started, so gait_traj_t0 and gait_leg_target_pt still belong to
          // the PREVIOUS leg. Latching off those would complete a leg that
          // never ran, on stale targets.
          if (gait_entered && gait_left_surface && !gait_touch_latched &&
              touching[f] && near_target) {
            gait_touch_latched = true;
            gait_touch_time = t;
            std::cout << "[t=" << t << "] gait: finger " << f
                      << " re-contacted, settling "
                      << FLAGS_regrasp_settle_time << " s\n";
          }
          // gait_entered: without it, a leg that has finished leaves
          // gait_touch_latched set (it is only cleared in a leg's ENTRY), so
          // during the --gait_leg_gap hold — when entry has not run yet —
          // this fires again on the previous leg's latch and completes the
          // NEXT leg instantly. That cascaded through every remaining leg at
          // one per tick, finishing a whole cycle's regrasps in 4 ms without
          // a finger moving.
          if (gait_entered && gait_touch_latched &&
              t >= gait_touch_time + FLAGS_regrasp_settle_time) {
            // The fingertip is back at its original WORLD point, so in the
            // cube's frame its footprint has rotated backwards by one
            // --gait_delta. Everything that describes where f should be —
            // the tracking IK's footprint, C3's cost target, and the
            // executor's live reference — has to move together, or the
            // next track-IK solve drags f straight back off its new point.
            // A gait leg's footprint walks backwards by one --gait_delta
            // (the fingertip returned to the world point it started at); a
            // realign leg lands on the finger's ORIGINAL cube-frame vertex
            // outright. Either way the tracking IK's footprint, C3's cost
            // target and the executor's live reference must move together,
            // or the next track-IK solve drags f off its new point.
            const VectorXd& q_dest = gait_leg_q_dest;
            footprint_C[f] = gait_leg_target_C;
            x_des_base.segment(finger_start[f], 4) =
                q_dest.segment(finger_start[f], 4);
            q_contact_live.segment(finger_start[f], 4) =
                q_dest.segment(finger_start[f], 4);
            q_contact_live_end.segment(finger_start[f], 4) =
                q_dest.segment(finger_start[f], 4);
            if (leg_kind == kLegDisengage) {
              // Ring is off the cube now. C3 was already rebuilt without it
              // at leg entry, so nothing to add back — but the tracking IK
              // has to stop solving for it, and the executor needs somewhere
              // to hold it while it waits out the next rotation.
              ring_engaged = false;
              q_ring_parked = q_dest;
            } else {
              std::vector<int> joined = active_fingers;
              if (std::find(joined.begin(), joined.end(), f) == joined.end())
                joined.push_back(f);
              std::sort(joined.begin(), joined.end());
              rebuild_c3(joined, x_current, t);
              if (leg_kind == kLegEngage) {
                ring_engaged = true;
                hand_tau_crossfade_from = last_tau_pd;
                hand_tau_crossfade_active = true;
              }
            }
            std::cout << "[t=" << t << "] gait"
                      << (gait_in_realign ? " realign" : "") << ": finger "
                      << f
                      << (leg_kind == kLegDisengage ? " parked, C3 solving "
                                                    : " joined C3, solving ")
                      << active_fingers.size() << " contacts\n";
            gait_entered = false;
            gait_touch_latched = false;   // cleared here as well as at entry
            gait_left_surface = false;
            gait_leg_done_t = t;
            ++gait_leg;
            if (gait_leg >= static_cast<int>(leg_plan.size())) {
              gait_leg = 0;
              if (gait_in_realign) {
                // Realignment is the last thing the gait does.
                gait_state = kGaitDone;
                std::cout << "[t=" << t
                          << "] gait: DONE — triangle realigned, base parallel "
                             "to the cube's base edge\n";
              } else if (++gait_cycle >= FLAGS_gait_cycles) {
                std::cout << "[t=" << t << "] gait: " << gait_cycle
                          << " cycles complete, commanded total "
                          << gait_cycle * FLAGS_gait_delta * 180.0 / M_PI
                          << " deg\n";
                if (FLAGS_gait_realign) {
                  // Straight into the realign legs: same kGaitMove state, no
                  // rotation in between, cube held at its final angle.
                  gait_in_realign = true;
                  std::cout << "[t=" << t
                            << "] gait: realigning fingers to their original "
                               "cube-frame vertices\n";
                } else {
                  gait_state = kGaitDone;
                }
              } else {
                gait_state = kGaitRotate;
              }
            }
          }
        }
      }

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
      // horizon at a single instant. Done at --track_ik_period_steps
      // cadence (an IK solve per control tick would be wasteful; this is
      // deliberately its own, independent, cheaper clock from
      // --relin_period_steps — see that flag's docs). Pinned → cube
      // frozen, skip.
      if (FLAGS_track_cube_contact && !cube_pinned && do_track_ik) {
        const auto ik_t0 = std::chrono::steady_clock::now();
        const RigidTransform<double> X_WC_meas = CubePoseFromPositions(
            sim_plant.GetPositions(plant_ctx, sim_cube));
        const std::pair<RigidTransform<double>, Vector3d> pose_now =
            cube_target_pose(t_ref_now);
        const std::pair<RigidTransform<double>, Vector3d> pose_end =
            cube_target_pose(t_ref_end);
        // Capture the about-to-be-stale value as the ramp's departure point
        // BEFORE overwriting it below — see q_contact_live_ramp_from's doc.
        q_contact_live_ramp_from = q_contact_live;
        track_ik_ramp_t0 = t;
        q_contact_live =
            resolve_contact_ik(clamp_ik_lead(pose_now.first, X_WC_meas), t);
        q_contact_live_end =
            resolve_contact_ik(clamp_ik_lead(pose_end.first, X_WC_meas), t);
        const double ik_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - ik_t0)
                                 .count();
        // Save/restore stream format — a bare setprecision here leaks into
        // every later print (the "dt=0.0" / "|A|=70.3" corruption).
        if (FLAGS_legacy_log) {
          const auto cout_flags = std::cout.flags();
          const auto cout_prec = std::cout.precision();
          std::cout << "  [track IK] " << std::fixed << std::setprecision(1)
                    << ik_ms << " ms\n";
          std::cout.flags(cout_flags);
          std::cout.precision(cout_prec);
        }
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
      // --gait counts as motion regardless of --cube_motion_mode: it drives
      // its own rotation reference through cube_target_pose, and without
      // this C3's cube-pose cost would sit at the static x_des_base while
      // only the fingers knew the cube was supposed to turn.
      const bool motion =
          ((FLAGS_cube_motion_mode != "none" || FLAGS_gait) && !cube_pinned);
      if (track || motion) {
        std::vector<VectorXd> x_des_traj(FLAGS_N + 1, x_des_base);
        if (track) {
          for (int k = 0; k <= FLAGS_N; ++k) {
            const double frac =
                FLAGS_N > 0 ? static_cast<double>(k) / FLAGS_N : 0.0;
            const VectorXd q_k =
                (1.0 - frac) * q_contact_live + frac * q_contact_live_end;
            // n_grasp_fingers, not a hardcoded 3: with the 4-finger triangle
            // ring is a real gripping contact and needs a per-knot reference
            // like everyone else. Leaving it out left ring's segment frozen
            // at x_des_base while the other three tracked a moving cube —
            // the same stale-ring failure resolve_contact_ik had.
            for (int i = 0; i < n_grasp_fingers; ++i)
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
        last_solve_t = t;
        const auto t0 = std::chrono::steady_clock::now();
        c3->Solve(x_current);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        solve_ms_sum += ms;
        solve_ms_max = std::max(solve_ms_max, ms);
        ++solve_calls;

        // Both diagnostics below assume the ORIGINAL {index,middle,thumb}
        // composition specifically (hardcoded "idx"/"mid"/"thu" labels and,
        // for print_cube_lambda_map, literal column indices like Dzn(2) for
        // "thumb") — checking n_contacts==3 alone isn't enough once
        // --release_middle can produce a DIFFERENT 3-contact set
        // ({index,thumb,ring}), so compare the actual composition.
        const bool is_original_3fingers =
            (active_fingers == std::vector<int>{0, 1, 2});

        // Print the map for every completed solve, including after each
        // relinearization, so changes in contact geometry are visible.
        // Gated behind --lambda_map_debug (off by default) — see flag doc.
        // Also requires is_original_3fingers: print_cube_lambda_map
        // hardcodes index/middle/thumb indexing (e.g. Dzn(2) for the
        // "thumb" column) — an out-of-bounds access for 2 contacts, a
        // mislabel for {index,thumb,ring}.
        if (FLAGS_lambda_map_debug && is_original_3fingers) {
          print_cube_lambda_map(c3->GetLCS(), t);
        }

        const std::vector<VectorXd> xplan     = c3->GetStateSolution();
        const std::vector<VectorXd> lam_plan  = c3->GetForceSolution();
        const std::vector<VectorXd> u_plan    = c3->GetInputSolution();

        if (FLAGS_c3_joint_plan_log) {
          const VectorXd q_meas = x_current.head(n_hand_q);
          std::cout << "[C3 QPLAN t=" << t << "] q_meas=["
                    << q_meas.transpose() << "]\n";
          for (size_t k = 0; k < xplan.size(); ++k) {
            const VectorXd qk = xplan[k].head(n_hand_q);
            const VectorXd dq = qk - q_meas;
            std::cout << "  q[" << k << "]=[" << qk.transpose()
                      << "]  |qk-qmeas|=" << dq.norm()
                      << " max=" << dq.cwiseAbs().maxCoeff();
            if (k == 1) {
              std::cout << "  q1err_by=[idx "
                        << dq.segment(finger_start[0], 4).norm() << ", mid "
                        << dq.segment(finger_start[1], 4).norm() << ", thu "
                        << dq.segment(finger_start[2], 4).norm() << ", ring "
                        << dq.segment(finger_start[3], 4).norm() << "]";
            }
            std::cout << "\n";
          }
        }

        if (!FLAGS_legacy_log) {
          // Default: none of the per-solve diagnostics below print at all.
          // They are a wall of tables per C3 solve, and under --gait_scheme=
          // relay the rotate phase's contact set is {index,middle,thumb} —
          // the very set the full-table branch was written for — so relay
          // triggers the most verbose path of all. --gait_log carries what
          // the gait needs; --legacy_log brings these back.
        } else if (FLAGS_contact_force_log) {
          // --contact_force_log suppresses both the C3 PLAN table and the
          // --plan_debug SOLVER DIAG output — see xplan_prev caching below.
        } else if (!is_original_3fingers) {
          // Non-default contact set (--release_middle, either the
          // 2-contact window or the 3-contact {index,thumb,ring} one): the
          // PLAN table below (and --plan_debug's SOLVER DIAG) are built
          // around fixed labels ("idx","mid","thu") — not worth adapting.
          // Simple status line instead; doesn't affect what C3 actually
          // solves or what torque gets applied, only this diagnostic
          // printout.
          std::cout << "\n=== C3 PLAN @ t=" << t << " s  (active fingers=[";
          for (size_t k = 0; k < active_fingers.size(); ++k)
            std::cout << active_fingers[k]
                      << (k + 1 < active_fingers.size() ? "," : "");
          std::cout << "] — table suppressed) ===\n";
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
        const bool use_ik_target =
            FLAGS_osc_target_source == "ik" ||
            (FLAGS_osc_target_source == "auto" &&
             FLAGS_track_cube_contact);
        if (use_ik_target) {
          // Ramp, not snap — q_contact_live itself only ever steps (see its
          // doc); reading it raw here is what fed a --track_ik_period_steps
          // staircase straight into a pure-kp PD law. frac reaches 1 right
          // as the NEXT do_track_ik fires (fixed period), at which point
          // ramp_from/ramp_t0 are refreshed and the ramp restarts from
          // wherever this one ended — continuous by construction, no
          // residual jump at the handoff.
          const double track_ik_period =
              std::max(1, FLAGS_track_ik_period_steps) * control_dt;
          const double ramp_frac = std::clamp(
              (t - track_ik_ramp_t0) / track_ik_period, 0.0, 1.0);
          q_des = (1.0 - ramp_frac) * q_contact_live_ramp_from +
                  ramp_frac * q_contact_live;
        } else if (xplan_now.size() > 1) {
          q_des = xplan_now[1].head(n_hand_q);
        }

        // --release_middle: once triggered and until it rejoins C3 (ring
        // only — see finger_rejoined), --release_finger's 4 joints are
        // pinned to a target instead of whatever q_des picked above — this
        // IS the PD control: no C3 lambda feedforward reaches this finger
        // (excluded from active_fingers below) and q_des feeds straight
        // into the same tau_pd law as everyone else, so a plain position
        // error is all that drives it. Middle just pins to its fixed
        // retracted target forever; ring instead reads a time-indexed
        // point off ring_regrasp_traj — the spline built at release time —
        // so it travels smoothly from its old contact point to the new
        // one instead of q_des jumping straight there. Unconditional
        // otherwise, so it wins regardless of --track_cube_contact. The
        // other fingers need no override here — they've been at their
        // triangle positions since the reach phase (or are back under
        // normal C3 tracking, for ring, once finger_rejoined).
        if (finger_released && !finger_rejoined) {
          if (FLAGS_release_finger == "middle") {
            q_des.segment(finger_start[1], 4) =
                q_release_middle.segment(finger_start[1], 4);
          } else {  // "ring"
            const double tl = std::clamp(t - ring_regrasp_start_t, 0.0,
                                          ring_regrasp_traj.end_time());
            const VectorXd q_ring_spline = ring_regrasp_traj.value(tl).col(0);
            q_des.segment(finger_start[3], 4) =
                q_ring_spline.segment(finger_start[3], 4);
          }
        }
        // Middle's loosen phase, chained after ring's rejoin: same
        // PD-not-C3 mechanism as the block above, just for middle and
        // gated on middle_loosened/middle_rejoined instead of
        // finger_released/finger_rejoined — these never overlap in time
        // (middle_loosened only ever goes true after finger_rejoined
        // already has, at which point the block above is already
        // inactive) and never touch the same finger_start[] slot.
        if (middle_loosened && !middle_rejoined) {
          const double tl = std::clamp(t - middle_regrasp_start_t, 0.0,
                                        middle_regrasp_traj.end_time());
          const VectorXd q_middle_spline =
              middle_regrasp_traj.value(tl).col(0);
          q_des.segment(finger_start[1], 4) =
              q_middle_spline.segment(finger_start[1], 4);
        }
        // Index's loosen phase, chained after middle's rejoin: same
        // PD-not-C3 mechanism as the two blocks above, just for index and
        // gated on index_loosened/index_rejoined.
        if (index_loosened && !index_rejoined) {
          const double tl = std::clamp(t - index_regrasp_start_t, 0.0,
                                        index_regrasp_traj.end_time());
          const VectorXd q_index_spline =
              index_regrasp_traj.value(tl).col(0);
          q_des.segment(finger_start[0], 4) =
              q_index_spline.segment(finger_start[0], 4);
        }
        // --gait: whichever finger is currently walking back to its old
        // world point is PD-driven along gait_traj rather than following C3
        // or the tracking IK — same mechanism as the three blocks above,
        // just re-armed once per finger per cycle instead of firing once.
        // gait_entered gates it: on the tick a leg completes, gait_leg has
        // already advanced but the next leg's trajectory has not been built
        // yet, and without this guard the previous finger's path would be
        // applied to the next finger for one tick.
        // relay: ring spends every rotation off the cube. Nothing else
        // commands it there — q_contact_live only carries the fingers the
        // tracking IK solves for — so hold it at the parked configuration
        // the disengage leg ended on, or it drifts back into the cube.
        if (FLAGS_gait && !ring_engaged && q_ring_parked.size() == n_hand_q &&
            !(gait_state == kGaitMove && gait_entered &&
              leg_plan[gait_leg].first == 3)) {
          q_des.segment(finger_start[3], 4) =
              q_ring_parked.segment(finger_start[3], 4);
        }
        if (FLAGS_gait && gait_state == kGaitMove && gait_entered) {
          const int gf = leg_plan[gait_leg].first;
          const double tl = t - gait_traj_t0;
          // Arc, then seek. While the spline is running the finger follows
          // it, which is what lifts the tip clear of the face. Past its end
          // the spline would just hold its final knot — the destination as
          // it was aimed when the leg began — so switch to gait_leg_q_dest,
          // which the seek block re-solves against the cube's live pose.
          // Holding the stale knot instead is what left a finger parked
          // beside a cube that had drifted out from under it.
          if (tl >= gait_traj.end_time()) {
            q_des.segment(finger_start[gf], 4) =
                gait_leg_q_dest.segment(finger_start[gf], 4);
          } else {
            const VectorXd q_gait_spline =
                gait_traj.value(std::max(tl, 0.0)).col(0);
            q_des.segment(finger_start[gf], 4) =
                q_gait_spline.segment(finger_start[gf], 4);
          }
        }

        // Gravity compensation: tau = -tau_gravity holds the hand static.
        const VectorXd tau_grav = sim_plant.GetVelocitiesFromArray(
            sim_allegro, -sim_plant.CalcGravityGeneralizedForces(plant_ctx));

        // qdot_des for --osc_kd: q_des is now FULLY finalized (every
        // override above has run), so this is the one place per tick that
        // sees its real, final value. Raw finite difference, then
        // low-pass filtered — see --osc_qd_filter_tau's doc for why the
        // raw version is unsafe (it would spike on every discrete q_des
        // update, e.g. --track_ik_period_steps' ~5ms jumps, exactly
        // reintroducing the kind of kick this is meant to remove).
        VectorXd qd_des = VectorXd::Zero(n_hand);
        if (qd_des_filt_valid) {
          const VectorXd qd_des_raw = (q_des - q_des_prev) / control_dt;
          const double alpha =
              std::clamp(control_dt / std::max(1e-6, FLAGS_osc_qd_filter_tau),
                        0.0, 1.0);
          qd_des_filt += alpha * (qd_des_raw - qd_des_filt);
          qd_des = qd_des_filt;
        }
        q_des_prev = q_des;
        qd_des_filt_valid = true;

        // Joint-space PD against a REAL velocity target — NOT via
        // CalcInverseDynamics/M(q), which was the earlier, heavier
        // architecture that injected energy at pin release and kicked the
        // cube loose. See --osc_kd's doc: qdot_des used to be implicitly
        // zero here, which fought the fingers' own legitimate tracking
        // motion during any active rotation; qd_des above fixes that.
        VectorXd tau_pd = FLAGS_osc_kp * (q_des - q_hand);
        if (FLAGS_osc_kd != 0.0) {
          tau_pd -= FLAGS_osc_kd * (v_hand - qd_des);
        }
        if (hand_tau_crossfade_active) {
          const double blend =
              FLAGS_gait_torque_ramp_time <= 0.0
                  ? 1.0
                  : std::clamp((t - finger_joined_t[3]) /
                                   FLAGS_gait_torque_ramp_time,
                               0.0, 1.0);
          tau_pd = (1.0 - blend) * hand_tau_crossfade_from + blend * tau_pd;
          if (blend >= 1.0) hand_tau_crossfade_active = false;
        }
        last_tau_pd = tau_pd;

        // Feedforward contact normal force from C3's own planned lambda_n
        // (physical units: GetForceSolution() / GetLambdaScaling()).
        const std::vector<VectorXd> lam_plan_now = c3->GetForceSolution();
        const VectorXd lam0 = lam_plan_now[0] / c3->GetLambdaScaling();
        const RotationMatrix<double> R_WC =
            CubePoseFromPositions(sim_plant.GetPositions(plant_ctx, sim_cube))
                .rotation();
        // 4th entry (ring) is --release_middle only: ring now touches the
        // -Y face (the triangle's base-right point, same face as
        // index/middle), so its inward push direction is +Y — same as
        // index/middle, not the old +X-face -X direction.
        const std::array<Vector3d, 4> press_C{
            Vector3d(0, 1, 0), Vector3d(0, 1, 0), Vector3d(0, -1, 0),
            Vector3d(0, 1, 0)};
        // Iterates active_fingers (physical finger indices currently in the
        // LCS) rather than 0..3 — this is what excludes middle from the
        // force feedforward once released (structural: it simply isn't in
        // active_fingers). af indexes normal_groups/lam0 (LCS/λ-ordered,
        // position in active_fingers); finger is the physical index for
        // tip_bodies/press_C.
        VectorXd tau_force = VectorXd::Zero(n_hand_v);
        std::array<double, 4> fn_target{0.0, 0.0, 0.0, 0.0};
        std::array<double, 4> fn_applied{0.0, 0.0, 0.0, 0.0};
        for (size_t af = 0; af < active_fingers.size(); ++af) {
          const int finger = active_fingers[af];
          double fn = 0.0;
          for (int idx : normal_groups[af]) fn += lam0(idx);
          fn_target[finger] = fn;
          // Ramp a newly-joined finger's grip in over --gait_force_ramp_time
          // instead of handing it C3's full, freshly-computed value the
          // instant it joins — see the flag doc for why (measured: the step
          // is the correct optimum changing when the contact count changes,
          // not solver noise, so nothing removes it except spreading it out
          // in time). finger_joined_t defaults to -1e9, so a finger that has
          // been in since the very first rebuild reads ramp=1 always.
          const double ramp =
              FLAGS_gait_force_ramp_time <= 0.0
                  ? 1.0
                  : std::clamp((t - finger_joined_t[finger]) /
                                   FLAGS_gait_force_ramp_time,
                               0.0, 1.0);
          fn = (1.0 - ramp) * force_crossfade_from[finger] + ramp * fn;
          fn *= FLAGS_lambda_torque_scale;
          fn_applied[finger] = fn;
          Eigen::MatrixXd J(3, sim_plant.num_velocities());
          sim_plant.CalcJacobianTranslationalVelocity(
              plant_ctx, drake::multibody::JacobianWrtVariable::kV,
              sim_plant.get_body(tip_bodies[finger]).body_frame(),
              Vector3d::Zero(), sim_plant.world_frame(),
              sim_plant.world_frame(), &J);
          tau_force += J.leftCols(n_hand_v).transpose() *
                      (fn * (R_WC * press_C[finger]));
        }

        tau_hand = tau_grav + tau_pd + tau_force;
        const bool handoff_debug_window =
            FLAGS_gait && FLAGS_gait_scheme == "relay" &&
            (gait_touch_latched ||
             (ring_engaged && t <= finger_joined_t[3] + 0.25));
        const bool first_handoff_ticks =
            ring_engaged && t <= finger_joined_t[3] + 0.015;
        if (FLAGS_gait_handoff_debug && handoff_debug_window &&
            (first_handoff_ticks || c3_iter % 5 == 0)) {
          const double ring_qerr =
              (q_des.segment(finger_start[3], 4) -
               q_hand.segment(finger_start[3], 4))
                  .norm();
          const double ring_tau_pd =
              tau_pd.segment(finger_start[3], 4).norm();
          VectorXd delta_tau = VectorXd::Zero(n_hand);
          if (tau_hand_prev_debug_valid)
            delta_tau = tau_hand - tau_hand_prev_debug;
          std::cout << "[HANDOFF t=" << t << "] ring_qerr=" << ring_qerr
                    << " ring_tau_pd=" << ring_tau_pd
                    << " |tau_pd|=" << tau_pd.norm()
                    << " |tau_force|=" << tau_force.norm()
                    << " |dtau|=" << delta_tau.norm() << " pd_by=["
                    << tau_pd.segment(finger_start[0], 4).norm() << ","
                    << tau_pd.segment(finger_start[1], 4).norm() << ","
                    << tau_pd.segment(finger_start[2], 4).norm() << ","
                    << tau_pd.segment(finger_start[3], 4).norm()
                    << "] dtau_by=["
                    << delta_tau.segment(finger_start[0], 4).norm() << ","
                    << delta_tau.segment(finger_start[1], 4).norm() << ","
                    << delta_tau.segment(finger_start[2], 4).norm() << ","
                    << delta_tau.segment(finger_start[3], 4).norm()
                    << "] fn_target=["
                    << fn_target[0] << "," << fn_target[1] << ","
                    << fn_target[2] << "," << fn_target[3]
                    << "] fn_applied=[" << fn_applied[0] << ","
                    << fn_applied[1] << "," << fn_applied[2] << ","
                    << fn_applied[3] << "]\n";
        }
        tau_hand_prev_debug = tau_hand;
        tau_hand_prev_debug_valid = true;
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
        for (int i = 0; i < n_grasp_fingers; ++i)
          q_tgt_c3.segment(finger_start[i], 4) =
              q_contact.segment(finger_start[i], 4);
        tau_hand = FLAGS_kp * (q_tgt_c3 - q_hand) +
                   FLAGS_kd * (-v_hand) + tau_g;

        // Diagnostic: is the commanded torque actually saturating at
        // ±tau_max before the clamp below? If so, kp/kd values stop
        // mattering — the delivered torque is just the bound, every tick,
        // which looks exactly like a gain-independent limit cycle.
        if (do_relin && !FLAGS_contact_force_log && FLAGS_legacy_log) {
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

    // ── --jerk_log: flag a sudden cube-acceleration spike, every tick ────
    // Deliberately every tick, not on --gait_log_period's cadence: the
    // point is catching the EXACT tick something abrupt happens, not the
    // nearest sampled one. See the flag doc for why this is acceleration
    // (one finite difference of the cube's own simulated velocity) rather
    // than literal jerk.
    if (FLAGS_jerk_log && phase == kC3 && !cube_pinned) {
      const Vector3d v_cube_now =
          sim_plant.GetVelocities(plant_ctx, sim_cube).tail<3>();
      if (v_cube_prev_valid) {
        const Vector3d accel = (v_cube_now - v_cube_prev) / control_dt;
        const double accel_norm = accel.norm();
        if (accel_norm > FLAGS_jerk_accel_thresh) {
          if (!jerk_active) {
            jerk_active = true;
            // Full context computed here, not borrowed from --gait_log
            // (which runs on its own slower cadence and may not have fired
            // this exact tick) — so this line means something read alone.
            const double t_ref_log = std::max(0.0, t - (handoff_t + 0.5));
            const RigidTransform<double> X_cmd =
                cube_target_pose(t_ref_log).first;
            const Eigen::AngleAxis<double> aa_cmd =
                (X_WC0.rotation().inverse() * X_cmd.rotation()).ToAngleAxis();
            const Eigen::AngleAxis<double> aa_meas =
                (X_WC0.rotation().inverse() * X_WC_now.rotation())
                    .ToAngleAxis();
            const Vector3d axis_ref =
                aa_cmd.angle() > 1e-6 ? aa_cmd.axis() : Vector3d::UnitY();
            const Vector3d r_meas = aa_meas.angle() * aa_meas.axis();
            const double deg = 180.0 / M_PI;
            const double got_deg = r_meas.dot(axis_ref) * deg;
            const double off_deg =
                (r_meas - r_meas.dot(axis_ref) * axis_ref).norm() * deg;
            const Vector3d drift_mm =
                (X_WC_now.translation() - X_cmd.translation()) * 1e3;
            const char* fname[4] = {"index", "middle", "thumb", "ring"};
            const char* kname[3] = {"REGRASP", "ENGAGE", "DISENGAGE"};

            const auto jl_flags = std::cout.flags();
            const auto jl_prec = std::cout.precision();
            std::cout << std::fixed << std::setprecision(2) << "[JERK t=" << t
                      << "] accel=" << std::setprecision(2) << accel_norm
                      << "m/s^2 (thresh=" << FLAGS_jerk_accel_thresh
                      << ")  v=(" << v_cube_now.x() << "," << v_cube_now.y()
                      << "," << v_cube_now.z() << ")m/s"
                      // How long ago the currently-active LCS/lambda were
                      // last refreshed, in ms — a small solve_age here means
                      // this jerk landed right after a fresh do_solve; a
                      // small relin_age means right after a relinearize.
                      // Consistently small across every jerk line would
                      // point at the ongoing 25Hz/200Hz update cadence
                      // itself, not the contact-count transition — neither
                      // small would point elsewhere entirely (contact
                      // detection, the simulator's own integrator, etc).
                      << "  solve_age=" << (t - last_solve_t) * 1e3
                      << "ms relin_age=" << (t - last_relin_t) * 1e3
                      << "ms  ";
            if (FLAGS_gait) {
              if (gait_in_realign) {
                std::cout << "REALIGN  ";
              } else {
                std::cout << "cyc " << (gait_cycle + 1) << "/"
                          << FLAGS_gait_cycles << "  ";
              }
              if (gait_state == kGaitRotate) {
                std::cout << "ROTATE  ";
              } else if (gait_state == kGaitMove) {
                const int lf = leg_plan[gait_leg].first;
                std::cout
                    << "MOVE " << fname[lf] << " "
                    << kname[static_cast<int>(leg_plan[gait_leg].second)]
                    << "  ";
              } else {
                std::cout << "DONE  ";
              }
            }
            std::cout << std::setprecision(1)
                      << "cmd=" << aa_cmd.angle() * deg << " got=" << got_deg
                      << " off=" << off_deg << " drift=(" << drift_mm.x()
                      << "," << drift_mm.y() << "," << drift_mm.z()
                      << ")mm  grip=[";
            for (size_t k = 0; k < active_fingers.size(); ++k)
              std::cout << fname[active_fingers[k]]
                        << (k + 1 < active_fingers.size() ? "," : "");
            std::cout << "]\n";
            std::cout.flags(jl_flags);
            std::cout.precision(jl_prec);
          }
        } else {
          jerk_active = false;
        }
      }
      v_cube_prev = v_cube_now;
      v_cube_prev_valid = true;
    }

    // ── --gait_log: one line that says what the gait is doing ───────────
    // Everything the legacy per-relin and per-solve prints buried. Reads the
    // same measured cube pose the rotation diagnostic below uses, so the
    // rotation numbers here need no second computation: cmd is what the
    // schedule asked for, got is what the cube actually did about that axis,
    // off is how far it tilted away from it.
    if (FLAGS_gait && FLAGS_gait_log && phase == kC3 && t >= next_gait_log_t) {
      next_gait_log_t = t + std::max(1e-3, FLAGS_gait_log_period);
      const double t_ref_log = std::max(0.0, t - (handoff_t + 0.5));
      const RigidTransform<double> X_cmd = cube_target_pose(t_ref_log).first;
      const Eigen::AngleAxis<double> aa_cmd =
          (X_WC0.rotation().inverse() * X_cmd.rotation()).ToAngleAxis();
      const Eigen::AngleAxis<double> aa_meas =
          (X_WC0.rotation().inverse() * X_WC_now.rotation()).ToAngleAxis();
      const Vector3d axis_ref =
          aa_cmd.angle() > 1e-6 ? aa_cmd.axis() : Vector3d::UnitY();
      const Vector3d r_meas = aa_meas.angle() * aa_meas.axis();
      const double deg = 180.0 / M_PI;
      const double got_deg = r_meas.dot(axis_ref) * deg;
      const double off_deg =
          (r_meas - r_meas.dot(axis_ref) * axis_ref).norm() * deg;
      const Vector3d drift_mm =
          (X_WC_now.translation() - X_cmd.translation()) * 1e3;

      const char* fname[4] = {"index", "middle", "thumb", "ring"};
      const char* kname[3] = {"REGRASP", "ENGAGE", "DISENGAGE"};

      const auto gl_flags = std::cout.flags();
      const auto gl_prec = std::cout.precision();
      std::cout << std::fixed << std::setprecision(2) << "[gait t=" << t
                << "] ";
      // Where we are in the plan.
      if (gait_in_realign) {
        std::cout << "REALIGN     ";
      } else {
        std::cout << "cyc " << (gait_cycle + 1) << "/" << FLAGS_gait_cycles
                  << "  ";
      }
      // What the current state is doing. During a leg, name the finger, the
      // kind of move, how far its tip still has to go, and whether it has
      // found the surface yet — the three things that decide whether the leg
      // latches or times out.
      if (gait_state == kGaitRotate) {
        std::cout << "ROTATE                          ";
      } else if (gait_state == kGaitMove) {
        const int lf = leg_plan[gait_leg].first;
        const Vector3d tip =
            sim_plant.EvalBodyPoseInWorld(plant_ctx,
                                          sim_plant.get_body(tip_bodies[lf])) *
            (lf == 3 ? ring_surface_offset : tip_surface_pt);
        std::cout << "MOVE " << fname[lf] << " "
                  << kname[static_cast<int>(leg_plan[gait_leg].second)]
                  << "  d=" << std::setprecision(1)
                  << (tip - gait_leg_target_pt).norm() * 1e3 << "mm"
                  << " touch=" << (touching[lf] ? "yes" : "no ") << " ";
      } else {
        std::cout << "DONE                            ";
      }
      std::cout << std::setprecision(1) << " cmd=" << aa_cmd.angle() * deg
                << " got=" << got_deg << " off=" << off_deg
                << " drift=(" << drift_mm.x() << "," << drift_mm.y() << ","
                << drift_mm.z() << ")mm  grip=[";
      for (size_t k = 0; k < active_fingers.size(); ++k)
        std::cout << fname[active_fingers[k]]
                  << (k + 1 < active_fingers.size() ? "," : "");
      std::cout << "]\n";
      std::cout.flags(gl_flags);
      std::cout.precision(gl_prec);
    }

    // ── --rot_log: cube orientation tracking ────────────────────────────
    // Everything measured against X_WC0, the pose --cube_move_* offsets are
    // defined from. Both rotations are reduced to angle-axis: for a pure
    // --cube_move_pitch command the axis is the grasp axis (world/cube Y),
    // so "meas" is signed by whether the measured axis points along +Y or
    // -Y. off_axis is the angle between the measured and commanded axes —
    // it separates "the cube turned less than asked" (a tracking/stiffness
    // problem) from "the cube turned somewhere else" (contacts slipping),
    // which a scalar angle error alone cannot distinguish.
    if (FLAGS_rot_log && phase == kC3 && t >= next_rot_log_t) {
      next_rot_log_t = t + std::max(1e-3, FLAGS_rot_log_period);
      const double t_ref_log = std::max(0.0, t - (handoff_t + 0.5));
      const RigidTransform<double> X_cmd = cube_target_pose(t_ref_log).first;

      const Eigen::AngleAxis<double> aa_cmd =
          (X_WC0.rotation().inverse() * X_cmd.rotation()).ToAngleAxis();
      const Eigen::AngleAxis<double> aa_meas =
          (X_WC0.rotation().inverse() * X_WC_now.rotation()).ToAngleAxis();
      // Residual orientation error, commanded → measured. Independent of
      // the axis bookkeeping above: this is the number that has to be
      // small at the end of a maneuver.
      const double err_deg =
          (X_cmd.rotation().inverse() * X_WC_now.rotation())
              .ToAngleAxis().angle() * 180.0 / M_PI;

      // Decompose the measured rotation into its component ALONG the
      // commanded axis (par — the rotation we asked for) and everything
      // perpendicular to it (perp — parasitic tilt). Done on the rotation
      // VECTOR (angle*axis), not by comparing axis directions: the axis of
      // a near-zero rotation is numerically arbitrary, so an axis-vs-axis
      // angle reads ~90deg off pure disturbance whenever the commanded
      // rotation is small — which is every run's first second, and is not
      // slip. par/perp stay interpretable at any magnitude (a 2deg wobble
      // reads as perp=2deg, not as a 90deg "off axis" alarm). Rotation
      // vectors don't compose linearly, so this is exact only for small
      // perp; that is the regime where it matters.
      const Vector3d axis_ref =
          aa_cmd.angle() > 1e-6 ? aa_cmd.axis() : Vector3d::UnitY();
      const Vector3d r_meas = aa_meas.angle() * aa_meas.axis();
      const double par_deg = r_meas.dot(axis_ref) * 180.0 / M_PI;
      const double perp_deg =
          (r_meas - r_meas.dot(axis_ref) * axis_ref).norm() * 180.0 / M_PI;
      const double cmd_deg = aa_cmd.angle() * 180.0 / M_PI;
      // Per-axis, not a norm: the direction is the diagnosis. dz below zero
      // is the cube SAGGING — friction losing to gravity, so the fix is grip
      // force. dy is motion along the grasp axis, i.e. the cube squeezed out
      // from between thumb and fingers by unbalanced normal forces, which is
      // a force-distribution problem instead. A single magnitude cannot tell
      // those apart, and they pull in opposite directions.
      const Vector3d drift_mm =
          (X_WC_now.translation() - X_cmd.translation()) * 1e3;

      const auto rot_flags = std::cout.flags();
      const auto rot_prec = std::cout.precision();
      std::cout << "  [rot t=" << std::fixed << std::setprecision(2) << t
                << "] cmd=" << std::setprecision(1) << cmd_deg
                << "deg par=" << par_deg << "deg perp=" << perp_deg
                << "deg err=" << err_deg << "deg drift=[" << drift_mm.x()
                << "," << drift_mm.y() << "," << drift_mm.z() << "]mm\n";
      std::cout.flags(rot_flags);
      std::cout.precision(rot_prec);
    }
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
