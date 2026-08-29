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
    if (FLAGS_gait_scheme == "spider" &&
        (FLAGS_spider_yaw_delta == 0.0 ||
         std::abs(FLAGS_spider_yaw_delta) > 0.3)) {
      throw std::runtime_error(
          "--spider_yaw_delta must be nonzero and no larger than 0.3 rad "
          "for the isolated small-turn milestone.");
    }
    if (FLAGS_gait_scheme == "spider" && FLAGS_spider_yaw_duration <= 0.0) {
      throw std::runtime_error("--spider_yaw_duration must be positive.");
    }
    if (FLAGS_gait_scheme == "spider" &&
        (std::abs(FLAGS_spider_ring_red_y) > 0.02 ||
         std::abs(FLAGS_spider_ring_hold_z) > 0.02)) {
      throw std::runtime_error(
          "--spider_ring_{red_y,hold_z} must lie within +/-0.02 m of face "
          "centre.");
    }
  }
  if (FLAGS_release_middle && FLAGS_exec_mode != "osc") {
    throw std::runtime_error(
        "--release_middle is only implemented for --exec_mode=osc.");
  }
}

}  // namespace dairlib
