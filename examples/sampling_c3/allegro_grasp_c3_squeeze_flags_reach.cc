#include <limits>
#include <string>

#include <gflags/gflags.h>

namespace dairlib {

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
            "Re-solve the grasp-contact IK periodically after release, so the "
            "C3 hand-q reference (k_hold) and OSC PD anchor update instead "
            "of staying at static t=0 q_contact. The pose used for that IK is "
            "selected by --contact_ik_pose_source. Default false keeps the "
            "static q_contact (A/B toggle).");
DEFINE_string(contact_ik_pose_source, "reference",
              "Pose source for live contact IK when --track_cube_contact=true: "
              "'reference' preserves the existing behavior: use the commanded "
              "cube reference, capped by --cube_ik_lead_{pos,rot}_max relative "
              "to the measured cube. 'measured' uses the current simulator cube "
              "pose directly, so the OSC PD contact targets move with a drifting "
              "cube. Measured mode is contact-following, not a new force or "
              "contact-loss recovery controller.");
DEFINE_bool(contact_force_log, false,
            "Print a throttled per-finger SAP contact table for forces "
            "actually resolved ON THE CUBE. Each row sums every point-pair "
            "contact for that fingertip and reports geometric-normal "
            "compression, friction magnitude, vertical force, and slip. "
            "In osc mode, it also reports C3's raw normal force and the "
            "post-scale/crossfade OSC normal-force command from the prior "
            "1 ms tick. With --osc_full_contact_force=true it additionally "
            "prints C3, OSC-command, and SAP-resolved M_z about the cube "
            "center. Suppresses the C3 PLAN / --plan_debug SOLVER DIAG "
            "tables and other high-rate diagnostics while active.");
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
DEFINE_bool(osc_torque_split_log, false,
            "With --exec_mode=osc, print at each C3 solve the magnitude and "
            "per-finger split between OSC's joint-PD torque and its C3 "
            "normal-force Jacobian-transpose torque. Percentages normalize "
            "those two component norms only; gravity is deliberately excluded.");

}  // namespace dairlib
