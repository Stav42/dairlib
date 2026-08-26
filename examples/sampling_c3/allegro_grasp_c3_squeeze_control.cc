#include "examples/sampling_c3/allegro_grasp_c3_squeeze_control.h"

#include <algorithm>
#include <stdexcept>

namespace dairlib::allegro_grasp_c3 {
namespace {

void CheckSameSize(const Eigen::VectorXd& first, const Eigen::VectorXd& second,
                   const char* message) {
  if (first.size() != second.size()) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

Eigen::VectorXd UpdateDesiredVelocityFilter(
    const Eigen::VectorXd& desired_position, double dt, double time_constant,
    DesiredVelocityFilterState* state) {
  if (state == nullptr || dt <= 0.0 || time_constant < 0.0) {
    throw std::invalid_argument(
        "Filter state, positive dt, and nonnegative time constant are required.");
  }
  if (!state->valid) {
    state->valid = true;
    state->previous_target = desired_position;
    state->filtered_velocity =
        Eigen::VectorXd::Zero(desired_position.size());
    return state->filtered_velocity;
  }

  CheckSameSize(desired_position, state->previous_target,
                "Desired-position dimension changed.");
  CheckSameSize(desired_position, state->filtered_velocity,
                "Filtered-velocity dimension changed.");
  const Eigen::VectorXd raw_velocity =
      (desired_position - state->previous_target) / dt;
  const double alpha = std::clamp(dt / std::max(1e-6, time_constant), 0.0,
                                  1.0);
  state->filtered_velocity +=
      alpha * (raw_velocity - state->filtered_velocity);
  state->previous_target = desired_position;
  return state->filtered_velocity;
}

Eigen::VectorXd ComputeJointPdTorque(
    const Eigen::VectorXd& desired_position,
    const Eigen::VectorXd& measured_position,
    const Eigen::VectorXd& measured_velocity,
    const Eigen::VectorXd& desired_velocity, double kp, double kd) {
  CheckSameSize(desired_position, measured_position,
                "Desired and measured positions must have equal size.");
  CheckSameSize(desired_position, measured_velocity,
                "Position and measured-velocity dimensions must match.");
  CheckSameSize(desired_position, desired_velocity,
                "Position and desired-velocity dimensions must match.");
  return kp * (desired_position - measured_position) -
         kd * (measured_velocity - desired_velocity);
}

double BlendJoinedNormalForce(double previous_force, double target_force,
                              double elapsed, double ramp_time,
                              double torque_scale) {
  const double fraction =
      ramp_time <= 0.0 ? 1.0 : std::clamp(elapsed / ramp_time, 0.0, 1.0);
  return torque_scale *
         ((1.0 - fraction) * previous_force + fraction * target_force);
}

Eigen::VectorXd ClampTorque(const Eigen::VectorXd& torque,
                            double torque_limit) {
  if (torque_limit < 0.0) {
    throw std::invalid_argument("Torque limit must be nonnegative.");
  }
  return torque.cwiseMin(torque_limit).cwiseMax(-torque_limit);
}

}  // namespace dairlib::allegro_grasp_c3
