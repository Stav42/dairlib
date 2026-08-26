#pragma once

#include <Eigen/Dense>

namespace dairlib::allegro_grasp_c3 {

// Stateful low-pass filter for finite-differenced joint targets. It is kept
// separate from Drake so target-velocity behavior can be tested as a sequence
// of ordinary Eigen vectors.
struct DesiredVelocityFilterState {
  bool valid{false};
  Eigen::VectorXd previous_target;
  Eigen::VectorXd filtered_velocity;
};

Eigen::VectorXd UpdateDesiredVelocityFilter(
    const Eigen::VectorXd& desired_position, double dt, double time_constant,
    DesiredVelocityFilterState* state);

// Joint-space PD law used by the osc executor.
Eigen::VectorXd ComputeJointPdTorque(
    const Eigen::VectorXd& desired_position,
    const Eigen::VectorXd& measured_position,
    const Eigen::VectorXd& measured_velocity,
    const Eigen::VectorXd& desired_velocity, double kp, double kd);

// Cross-fades a normal-force request after a new contact joins the active set.
double BlendJoinedNormalForce(double previous_force, double target_force,
                              double elapsed, double ramp_time,
                              double torque_scale);

// Per-element saturation used just before actuator input is applied.
Eigen::VectorXd ClampTorque(const Eigen::VectorXd& torque,
                            double torque_limit);

}  // namespace dairlib::allegro_grasp_c3
