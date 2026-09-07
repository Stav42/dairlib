#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <gflags/gflags.h>

#include "allegro_grasp_c3_squeeze_flags.h"

namespace dairlib {

DEFINE_double(cube_start_x, 0.0, "Cube nominal start position, world x (m).");
DEFINE_double(cube_start_y, 0.0, "Cube nominal start position, world y (m).");
DEFINE_double(cube_start_z, 0.58,
              "Cube nominal start position, world z (m). See the note above "
              "on how this interacts with ring's reach during --gait.");
DEFINE_double(cube_size_scale, 1.0,
              "Uniform cube-size scale relative to the nominal 60 mm cube. "
              "1.0 uses the original 60 mm / 50 g cube; 0.6 uses the "
              "36 mm cube with mass and inertia scaled at constant density "
              "(10.8 g). The simulator and C3 linearization use the same "
              "selected model. Currently supported values are 1.0, 0.8, "
              "and 0.6.");

// ── General ──────────────────────────────────────────────────────────────────
DEFINE_bool(preview, true,
            "Pause on the two static Meshcat previews (q_pregrasp, q_contact) "
            "and wait for Enter before starting the sim. Set false for "
            "unattended runs and for any command that pipes stdout: the "
            "prompt goes to stdout, so a pipe hides it and the run looks "
            "hung when it is really blocked on std::cin.get().");
DEFINE_double(sim_time, std::numeric_limits<double>::infinity(),
              "Total simulation time (s).");

void ValidateAndNormalizeSqueezeFlags() {
  constexpr double kScaleTolerance = 1e-12;
  const bool supported_cube_size =
      std::abs(FLAGS_cube_size_scale - 1.0) < kScaleTolerance ||
      std::abs(FLAGS_cube_size_scale - 0.8) < kScaleTolerance ||
      std::abs(FLAGS_cube_size_scale - 0.6) < kScaleTolerance;
  if (!supported_cube_size) {
    throw std::runtime_error(
        "--cube_size_scale must be 1.0 (60 mm), 0.8 (48 mm), or 0.6 "
        "(36 mm).");
  }
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
  if (FLAGS_contact_ik_pose_source != "reference" &&
      FLAGS_contact_ik_pose_source != "measured") {
    throw std::runtime_error(
        "--contact_ik_pose_source must be 'reference' or 'measured'.");
  }
  if (FLAGS_osc_wrench_feedback) {
    if (FLAGS_exec_mode != "osc" || !FLAGS_osc_full_contact_force) {
      throw std::runtime_error(
          "--osc_wrench_feedback requires --exec_mode=osc and "
          "--osc_full_contact_force=true.");
    }
    if (FLAGS_osc_wrench_feedback_period <= 0.0 ||
        FLAGS_osc_wrench_feedback_yaw_kp < 0.0 ||
        FLAGS_osc_wrench_feedback_yaw_ki < 0.0 ||
        FLAGS_osc_wrench_feedback_max_yaw_moment <= 0.0 ||
        FLAGS_osc_wrench_feedback_max_force_per_contact <= 0.0 ||
        FLAGS_osc_wrench_feedback_force_rate_limit <= 0.0 ||
        FLAGS_osc_wrench_feedback_allocation_damping < 0.0 ||
        FLAGS_osc_wrench_feedback_min_commanded_yaw_moment <= 0.0 ||
        FLAGS_osc_wrench_feedback_min_resolved_yaw_moment < 0.0 ||
        FLAGS_osc_wrench_feedback_authority_timeout <= 0.0) {
      throw std::runtime_error(
          "measured-wrench feedback gains, limits, and periods must be "
          "nonnegative, with positive command limits and periods.");
    }
  }
  if (FLAGS_gait && FLAGS_gait_scheme != "triangle" &&
      FLAGS_gait_scheme != "relay" && FLAGS_gait_scheme != "spider") {
    throw std::runtime_error(
        "--gait_scheme must be 'triangle', 'relay', or 'spider'.");
  }
  if (FLAGS_gait) {
    FLAGS_release_middle = (FLAGS_gait_scheme == "triangle");
    if (!FLAGS_track_cube_contact) {
      throw std::runtime_error("--gait requires --track_cube_contact.");
    }
    if (FLAGS_gait_cycles < 1) {
      throw std::runtime_error("--gait_cycles must be >= 1.");
    }
    // The four-finger spider start no longer performs a contact-topology
    // handoff after yaw, so permit the requested 30 deg isolated yaw while
    // retaining a finite guard against accidental multi-turn commands.
    const double spider_yaw_limit = 0.55;  // slightly above 30 deg
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_yaw_delta == 0.0 ||
         std::abs(FLAGS_spider_yaw_delta) > spider_yaw_limit)) {
      throw std::runtime_error(
          "--spider_yaw_delta must be nonzero and no larger than " +
          std::to_string(spider_yaw_limit) +
          " rad for the isolated spider-yaw milestone.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_yaw_duration <= 0.0 ||
         FLAGS_spider_ring_placement_duration <= 0.0)) {
      throw std::runtime_error(
          "--spider_yaw_duration and --spider_ring_placement_duration "
          "must be positive.");
    }
    const double cube_half_width = 0.03 * FLAGS_cube_size_scale;
    const double spider_triangle_half_width =
        FLAGS_spider_triangle_half_width * FLAGS_cube_size_scale;
    const double spider_triangle_base_z =
        FLAGS_spider_triangle_base_z * FLAGS_cube_size_scale;
    const double spider_triangle_apex_z =
        FLAGS_spider_triangle_apex_z * FLAGS_cube_size_scale;
    if (FLAGS_gait_scheme == "spider" &&
        (spider_triangle_half_width <= 0.0 ||
         spider_triangle_half_width >= cube_half_width ||
         std::abs(spider_triangle_base_z) >= cube_half_width ||
         std::abs(spider_triangle_apex_z) >= cube_half_width)) {
      throw std::runtime_error(
          "the spider triangle must lie strictly within the selected yellow "
          "face, with a positive base half-width.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_index_crawl <= 0.0 ||
         FLAGS_spider_index_crawl > 0.006)) {
      throw std::runtime_error(
          "--spider_index_crawl must lie in (0, 0.006] m so index stays "
          "safely inside yellow/4 during the first crawl.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_index_arc_clearance <= 0.0 ||
         FLAGS_spider_index_arc_clearance > 0.02)) {
      throw std::runtime_error(
          "--spider_index_arc_clearance must lie in (0, 0.02] m.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        FLAGS_spider_index_duration <= 0.0) {
      throw std::runtime_error("--spider_index_duration must be positive.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_support_normal_margin <= 0.0 ||
         FLAGS_spider_support_normal_margin > 1.0)) {
      throw std::runtime_error(
          "--spider_support_normal_margin must lie in (0, 1] N.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_support_verify_time <= 0.0 ||
         FLAGS_spider_support_settle_time <= 0.0 ||
         FLAGS_spider_support_settle_yaw_error <= 0.0 ||
         FLAGS_spider_support_settle_yaw_error > M_PI ||
         FLAGS_spider_support_settle_translation_error <= 0.0 ||
         FLAGS_spider_support_settle_linear_speed <= 0.0 ||
         FLAGS_spider_support_settle_angular_speed <= 0.0 ||
         FLAGS_spider_ring_handoff_vertical_force_deficit < 0.0 ||
         FLAGS_spider_ring_handoff_vertical_force_deficit > 1.0 ||
         FLAGS_spider_support_max_translation_error <= 0.0 ||
         FLAGS_spider_support_max_orientation_error <= 0.0 ||
         FLAGS_spider_support_max_linear_speed <= 0.0 ||
         FLAGS_spider_support_max_angular_speed <= 0.0)) {
      throw std::runtime_error(
          "spider support settle/verification durations and all bounds must "
          "be positive; settle yaw error must not exceed pi; ring handoff "
          "vertical-force deficit must lie in [0, 1] N.");
    }
    if (FLAGS_gait_scheme == "spider" && FLAGS_spider_virtual_ring_search &&
        (FLAGS_spider_ring_search_rows <= 0 ||
         FLAGS_spider_ring_search_cols <= 0 ||
         FLAGS_spider_ring_search_face_margin < 0.0 ||
         FLAGS_spider_ring_search_face_margin >= cube_half_width)) {
      throw std::runtime_error(
          "spider virtual-ring search requires positive grid dimensions and "
          "a face margin smaller than the selected cube half-width.");
    }
  }
  if (FLAGS_release_middle && FLAGS_exec_mode != "osc") {
    throw std::runtime_error(
        "--release_middle is only implemented for --exec_mode=osc.");
  }
}

}  // namespace dairlib
