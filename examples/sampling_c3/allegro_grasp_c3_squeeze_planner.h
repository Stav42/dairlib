#pragma once

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "allegro_grasp_c3_squeeze_config.h"
#include "allegro_grasp_c3_squeeze_lcs_model.h"
#include "c3/core/c3.h"

namespace dairlib::allegro_grasp_c3 {

struct PlannerDimensions {
  int state{};
  int input{};
  int lambda{};
  int auxiliary{};
  int contacts{};
  int positions{};
  int hand_positions{};
  int hand_velocities{};
};

// Owns C3 and its mutable contact topology. Rebuild, relinearization, target
// updates, and solve cadence are exposed as separate operations so each can be
// exercised without running a simulator.
class C3Planner {
 public:
  C3Planner(const SqueezeConfig& config, LcsModel* model);
  ~C3Planner();

  C3Planner(const C3Planner&) = delete;
  C3Planner& operator=(const C3Planner&) = delete;

  void SetBaseTarget(const Eigen::VectorXd& target);
  void SetFingerTarget(int finger, const Eigen::Vector4d& joints);
  void Rebuild(const std::vector<int>& active_fingers,
               const Eigen::VectorXd& state, double time);
  void Relinearize(const Eigen::VectorXd& state);
  void UpdateTarget(const std::vector<Eigen::VectorXd>& target);
  void Solve(const Eigen::VectorXd& state);

  bool initialized() const;
  double input_scale() const;
  Eigen::VectorXd FirstPhysicalInput() const;
  Eigen::VectorXd FirstPhysicalForce() const;
  std::array<double, 4> DesiredNormalForces(double time) const;

  const PlannerDimensions& dimensions() const;
  const std::vector<int>& active_fingers() const;
  const std::vector<std::vector<int>>& normal_groups() const;
  const std::array<double, 4>& finger_joined_times() const;
  const std::array<double, 4>& force_crossfade_origins() const;
  const Eigen::VectorXd& base_target() const;

  std::vector<Eigen::VectorXd> state_solution() const;
  std::vector<Eigen::VectorXd> input_solution() const;
  std::vector<Eigen::VectorXd> force_solution() const;
  std::vector<Eigen::VectorXd> dual_delta_solution();
  const c3::C3& implementation() const;

 private:
  void ConfigureForceTracking();
  c3::LCS MakeScaledLinearization(const Eigen::VectorXd& state);

  const SqueezeConfig& config_;
  LcsModel* model_{};
  c3::LCSFactoryOptions lcs_options_;
  c3::C3Options c3_options_;
  PlannerDimensions dimensions_;
  Eigen::MatrixXd state_cost_;
  Eigen::VectorXd base_target_;
  std::vector<int> active_fingers_{0, 1, 2};
  std::vector<std::vector<int>> normal_groups_;
  std::array<double, 4> finger_joined_times_{-1e9, -1e9, -1e9, -1e9};
  std::array<double, 4> force_crossfade_origins_{0.0, 0.0, 0.0, 0.0};
  std::unique_ptr<c3::C3> c3_;
  double input_scale_{1.0};
};

}  // namespace dairlib::allegro_grasp_c3
