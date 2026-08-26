#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

#include <drake/multibody/tree/multibody_tree_indexes.h>

#include "allegro_grasp_c3_squeeze_control.h"

namespace drake {
namespace systems {
template <typename T>
class Context;
}
namespace multibody {
template <typename T>
class MultibodyPlant;
}
}  // namespace drake

namespace dairlib::allegro_grasp_c3 {

// Data needed to realize C3's first planned contact force as the OSC branch's
// Jacobian-transposed normal-force proxy.  It deliberately contains no gflags
// or C3 object: callers extract the physical lambda_0 first, which makes this
// computation a direct unit-test seam.
struct OscExecutorRequest {
  const drake::multibody::MultibodyPlant<double>& plant;
  const drake::systems::Context<double>& context;
  drake::multibody::ModelInstanceIndex hand_model;
  drake::multibody::ModelInstanceIndex cube_model;
  std::array<drake::multibody::BodyIndex, 4> tip_bodies;
  std::array<int, 4> finger_start;
  std::vector<int> active_fingers;
  std::vector<std::vector<int>> normal_groups;
  std::array<double, 4> force_crossfade_from;
  std::array<double, 4> finger_joined_time;
  Eigen::VectorXd q_desired;
  Eigen::VectorXd q_measured;
  Eigen::VectorXd v_measured;
  Eigen::VectorXd lambda0_physical;
  double time{};
  double dt{};
  double kp{};
  double kd{};
  double desired_velocity_filter_tau{};
  double force_ramp_time{};
  double lambda_torque_scale{};
};

struct OscExecutorResult {
  Eigen::VectorXd torque;
  Eigen::VectorXd pd_torque;
  Eigen::VectorXd force_torque;
  std::array<double, 4> normal_force_target{};
  std::array<double, 4> normal_force_applied{};
};

// Computes tau = tau_gravity + tau_PD + sum_i J_i^T f_{n,i} n_i.  This is
// intentionally not an operational-space inverse-dynamics controller.
OscExecutorResult ComputeOscExecutorTorque(
    const OscExecutorRequest& request,
    DesiredVelocityFilterState* desired_velocity_filter);

// Data for task_space after the caller has selected its position target and
// extracted the C3 dual-delta lambda.  Keeping the force projection and the
// torque realization here gives direct tests a single observable result.
struct TaskSpaceExecutorRequest {
  const drake::multibody::MultibodyPlant<double>& plant;
  const drake::systems::Context<double>& context;
  drake::multibody::ModelInstanceIndex hand_model;
  drake::multibody::ModelInstanceIndex cube_model;
  std::array<drake::multibody::BodyIndex, 3> tip_bodies;
  std::vector<std::vector<int>> normal_groups;
  Eigen::VectorXd fingertip_position_desired;
  Eigen::VectorXd hand_velocity;
  Eigen::VectorXd lambda_delta0;
  Eigen::VectorXd c3_input0_scaled;
  double task_kp{};
  double task_kd{};
  double force_floor{};
  bool add_gravity_compensation{};
};

Eigen::VectorXd ComputeTaskSpaceExecutorTorque(
    const TaskSpaceExecutorRequest& request);

}  // namespace dairlib::allegro_grasp_c3
