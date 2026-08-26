#include <limits>
#include <string>

#include <gflags/gflags.h>

namespace dairlib {

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
              "enough for a slow trend.");
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

}  // namespace dairlib
