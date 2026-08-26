#pragma once

#include <array>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <drake/common/trajectories/piecewise_polynomial.h>
#include <drake/math/rigid_transform.h>

#include "allegro_grasp_c3_squeeze_config.h"
#include "allegro_grasp_c3_squeeze_grasp.h"
#include "allegro_grasp_c3_squeeze_planner.h"
#include "allegro_grasp_c3_squeeze_reference.h"

namespace dairlib::allegro_grasp_c3 {

enum class GaitPhase { kRotate, kMove, kDone };
enum class GaitLegKind { kRegrasp, kEngage };

// Owns the release/recontact and gait state machines. It emits planner
// topology changes and joint-reference overrides through named operations;
// it does not own the simulator or run a control loop.
class ManeuverController {
 public:
  ManeuverController(const SqueezeConfig& config, GraspSetup* grasp);

  void Update(double time, double reference_time,
              const std::array<bool, 4>& touching,
              const Eigen::VectorXd& state,
              const drake::math::RigidTransform<double>& cube_pose,
              C3Planner* planner, Eigen::VectorXd* contact_start,
              Eigen::VectorXd* contact_end);
  void OverrideJointTarget(double time, Eigen::VectorXd* desired) const;
  void ConfigureCubeReference(CubeMotionReferenceConfig* reference) const;
  bool ring_engaged() const;

 private:
  struct RecontactState {
    bool started{};
    bool left_surface{};
    bool latched{};
    bool completed{};
    double start_time{};
    double latch_time{-1.0};
    drake::trajectories::PiecewisePolynomial<double> trajectory;
  };

  void UpdateRelease(double time, double reference_time,
                     const std::array<bool, 4>& touching,
                     const Eigen::VectorXd& state, C3Planner* planner,
                     Eigen::VectorXd* contact_start,
                     Eigen::VectorXd* contact_end);
  void UpdateGait(double time, double reference_time,
                  const std::array<bool, 4>& touching,
                  const Eigen::VectorXd& state,
                  const drake::math::RigidTransform<double>& cube_pose,
                  C3Planner* planner, Eigen::VectorXd* contact_start,
                  Eigen::VectorXd* contact_end);
  void CompleteFinger(int finger, const Eigen::VectorXd& destination,
                      const Eigen::Vector3d& footprint,
                      const Eigen::VectorXd& state, double time,
                      C3Planner* planner, Eigen::VectorXd* contact_start,
                      Eigen::VectorXd* contact_end);

  const SqueezeConfig& config_;
  GraspSetup* grasp_{};
  std::array<RecontactState, 3> release_steps_;
  bool middle_release_only_{};

  GaitPhase gait_phase_{GaitPhase::kRotate};
  int gait_cycle_{};
  int gait_leg_{};
  bool gait_realign_mode_{};
  bool gait_entered_{};
  bool gait_left_surface_{};
  bool gait_touch_latched_{};
  double gait_touch_time_{-1.0};
  double gait_leg_done_time_{-1e9};
  double gait_trajectory_start_{};
  double gait_theta_start_{};
  double gait_theta_target_{};
  double gait_rotate_start_{};
  Eigen::VectorXd gait_destination_;
  Eigen::VectorXd ring_parked_positions_;
  Eigen::Vector3d gait_target_C_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gait_target_W_{Eigen::Vector3d::Zero()};
  drake::trajectories::PiecewisePolynomial<double> gait_trajectory_;
  std::vector<std::pair<int, GaitLegKind>> gait_plan_;
  bool ring_engaged_{};
};

}  // namespace dairlib::allegro_grasp_c3
