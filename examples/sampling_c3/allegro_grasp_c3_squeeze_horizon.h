#pragma once

#include <vector>

#include <Eigen/Dense>

#include "examples/sampling_c3/allegro_grasp_c3_squeeze_reference.h"

namespace dairlib::allegro_grasp_c3 {

// State-vector indices needed to write a C3 desired trajectory. This keeps
// target construction independent of C3, Drake plants, and gflags.
struct HorizonStateLayout {
  int horizon{0};
  int num_hand_positions{0};
  int num_positions{0};
  int num_hand_velocities{0};
  std::vector<int> finger_start_indices;
};

struct HorizonTargetRequest {
  Eigen::VectorXd base_state;

  // When enabled, each active finger's four joints are interpolated between
  // these two separately solved live-IK postures.
  bool track_contact_posture{false};
  Eigen::VectorXd contact_posture_start;
  Eigen::VectorXd contact_posture_end;
  int num_grasp_fingers{0};

  // Empty means retain the cube components of base_state. Otherwise the
  // vector must contain horizon + 1 raw cube references.
  std::vector<CubeReference> cube_references;
};

// Builds x_des[0..N] from a base state, optional hand-contact endpoint
// targets, and optional raw cube references. This is the exact data assembly
// that precedes C3::UpdateTarget and is intended to be unit tested directly.
std::vector<Eigen::VectorXd> BuildHorizonTargetTrajectory(
    const HorizonStateLayout& layout, const HorizonTargetRequest& request);

}  // namespace dairlib::allegro_grasp_c3
