#include "examples/sampling_c3/allegro_grasp_c3_squeeze_horizon.h"

#include <stdexcept>

namespace dairlib::allegro_grasp_c3 {
namespace {

void ValidateLayout(const HorizonStateLayout& layout,
                    const HorizonTargetRequest& request) {
  if (layout.horizon < 0 || layout.num_hand_positions < 0 ||
      layout.num_positions < layout.num_hand_positions ||
      layout.num_hand_velocities < 0) {
    throw std::invalid_argument("Invalid C3 horizon state layout.");
  }
  const int cube_linear_velocity_start =
      layout.num_positions + layout.num_hand_velocities + 3;
  if (request.base_state.size() < cube_linear_velocity_start + 3) {
    throw std::invalid_argument(
        "Base state is too short for the configured cube state layout.");
  }
  if (request.track_contact_posture) {
    if (request.contact_posture_start.size() < layout.num_hand_positions ||
        request.contact_posture_end.size() < layout.num_hand_positions) {
      throw std::invalid_argument(
          "Live contact postures must contain all hand positions.");
    }
    if (request.num_grasp_fingers < 0 ||
        request.num_grasp_fingers >
            static_cast<int>(layout.finger_start_indices.size())) {
      throw std::invalid_argument(
          "Number of grasp fingers is incompatible with the layout.");
    }
    for (int finger = 0; finger < request.num_grasp_fingers; ++finger) {
      const int start = layout.finger_start_indices[finger];
      if (start < 0 || start + 4 > layout.num_hand_positions) {
        throw std::invalid_argument("Invalid four-joint finger block.");
      }
    }
  }
  if (!request.cube_references.empty() &&
      request.cube_references.size() !=
          static_cast<size_t>(layout.horizon + 1)) {
    throw std::invalid_argument(
        "Cube-reference trajectory must have horizon + 1 knots.");
  }
}

void WriteCubeReference(const CubeReference& cube_reference,
                        const HorizonStateLayout& layout,
                        Eigen::VectorXd* state) {
  Eigen::Quaterniond quaternion =
      cube_reference.pose.rotation().ToQuaternion();
  // The main program uses a positive-w quaternion convention for this free
  // body; choosing the equivalent sign here keeps the tracking error smooth.
  if (quaternion.w() < 0.0) quaternion.coeffs() *= -1.0;
  (*state)(layout.num_hand_positions + 0) = quaternion.w();
  (*state)(layout.num_hand_positions + 1) = quaternion.x();
  (*state)(layout.num_hand_positions + 2) = quaternion.y();
  (*state)(layout.num_hand_positions + 3) = quaternion.z();
  state->segment(layout.num_hand_positions + 4, 3) =
      cube_reference.pose.translation();
  const int cube_linear_velocity_start =
      layout.num_positions + layout.num_hand_velocities + 3;
  state->segment(cube_linear_velocity_start, 3) =
      cube_reference.translational_velocity;
}

}  // namespace

std::vector<Eigen::VectorXd> BuildHorizonTargetTrajectory(
    const HorizonStateLayout& layout, const HorizonTargetRequest& request) {
  ValidateLayout(layout, request);
  std::vector<Eigen::VectorXd> trajectory(layout.horizon + 1,
                                           request.base_state);

  for (int knot = 0; knot <= layout.horizon; ++knot) {
    Eigen::VectorXd& target = trajectory[knot];
    if (request.track_contact_posture) {
      const double fraction =
          layout.horizon > 0 ? static_cast<double>(knot) / layout.horizon
                             : 0.0;
      const Eigen::VectorXd posture =
          (1.0 - fraction) * request.contact_posture_start +
          fraction * request.contact_posture_end;
      for (int finger = 0; finger < request.num_grasp_fingers; ++finger) {
        const int start = layout.finger_start_indices[finger];
        target.segment(start, 4) = posture.segment(start, 4);
      }
    }
    if (!request.cube_references.empty()) {
      WriteCubeReference(request.cube_references[knot], layout, &target);
    }
  }
  return trajectory;
}

}  // namespace dairlib::allegro_grasp_c3
