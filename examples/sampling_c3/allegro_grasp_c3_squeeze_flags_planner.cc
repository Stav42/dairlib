#include <limits>
#include <string>

#include <gflags/gflags.h>

namespace dairlib {

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
             "flag also sets the ramp's duration — the actual fix for a "
             "continuous, rotation-only jerk traced to this cadence meeting "
             "a --osc_kd=0 (no velocity term) PD law.");
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

}  // namespace dairlib
