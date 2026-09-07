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

// The physical first-knot contact-force plan reconstructed in the exact
// world-frame force bases used by the current C3 linearization.  Forces are
// ON the cube, not on the fingertips.  The entries are indexed by Allegro
// finger number; inactive fingers remain zero.
struct C3ContactForcePlan {
  bool valid{};
  std::array<Eigen::Vector3d, 4> force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> contact_point_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
};

// A model-based check of whether the current C3 horizon can safely hold the
// cube with its presently active contacts.  These are observed properties of
// C3's solution, not desired per-finger force values.
struct C3HoldPlanMetrics {
  bool has_solution{};
  double minimum_normal_force{};
  double maximum_translation_error{};
  Eigen::Vector3d translation_error_at_maximum{Eigen::Vector3d::Zero()};
  int maximum_translation_error_knot{-1};
  double maximum_orientation_error{};
  double maximum_linear_speed{};
  double maximum_angular_speed{};
  // Contact-only vertical force on the cube at each force knot.  Compare
  // this with cube_weight to see whether the C3 plan counters gravity.
  std::vector<double> planned_cube_vertical_contact_force;
  // World-frame vertical-force breakdown of the same C3 contact plan. Each
  // row is in active_fingers() order.  For Stewart--Trinkle, beta is the
  // tangential/friction contribution and normal is lambda_n's contribution.
  // All values are forces ON the cube, so positive means upward.
  std::vector<std::vector<double>> planned_beta_vertical_force;
  std::vector<std::vector<double>> planned_normal_vertical_force;
  std::vector<double> planned_total_beta_vertical_force;
  std::vector<double> planned_total_normal_vertical_force;
  std::vector<double> planned_total_vertical_contact_force;
  double cube_weight{};
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
               const Eigen::VectorXd& state, double time,
               double normal_force_margin = 0.0,
               bool log_rebuild = true);
  void Relinearize(const Eigen::VectorXd& state);
  void UpdateTarget(const std::vector<Eigen::VectorXd>& target);
  void Solve(const Eigen::VectorXd& state);

  bool initialized() const;
  double input_scale() const;
  Eigen::VectorXd FirstPhysicalInput() const;
  Eigen::VectorXd FirstPhysicalForce() const;
  C3ContactForcePlan FirstPhysicalContactForcePlan() const;
  std::array<double, 4> DesiredNormalForces(double time) const;
  C3HoldPlanMetrics EvaluateHoldPlan() const;

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
  void ConfigureNormalForceMargin();
  c3::LCS MakeScaledLinearization(const Eigen::VectorXd& state);

  const SqueezeConfig& config_;
  LcsModel* model_{};
  c3::LCSFactoryOptions lcs_options_;
  c3::C3Options c3_options_;
  PlannerDimensions dimensions_;
  Eigen::MatrixXd state_cost_;
  Eigen::VectorXd base_target_;
  std::vector<Eigen::VectorXd> horizon_target_;
  std::vector<int> active_fingers_{0, 1, 2};
  std::vector<std::vector<int>> normal_groups_;
  std::array<double, 4> finger_joined_times_{-1e9, -1e9, -1e9, -1e9};
  std::array<double, 4> force_crossfade_origins_{0.0, 0.0, 0.0, 0.0};
  std::unique_ptr<c3::C3> c3_;
  double input_scale_{1.0};
  double normal_force_margin_{0.0};
};

}  // namespace dairlib::allegro_grasp_c3
