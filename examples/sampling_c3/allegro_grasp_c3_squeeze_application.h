#pragma once

#include <array>
#include <memory>

#include <Eigen/Core>

#include "allegro_grasp_c3_squeeze_config.h"
#include "allegro_grasp_c3_squeeze_control.h"
#include "allegro_grasp_c3_squeeze_environment.h"
#include "allegro_grasp_c3_squeeze_grasp.h"
#include "allegro_grasp_c3_squeeze_lcs_model.h"
#include "allegro_grasp_c3_squeeze_maneuver_controller.h"
#include "allegro_grasp_c3_squeeze_planner.h"
#include "allegro_grasp_c3_squeeze_reference.h"

namespace dairlib::allegro_grasp_c3 {

enum class ControlPhase { kReach, kC3 };

struct ReachState {
  std::array<bool, 4> arrived{false, false, false, false};
  std::array<double, 4> arrival_time{-1.0, -1.0, -1.0, -1.0};
};

struct TrackingState {
  Eigen::VectorXd contact_start;
  Eigen::VectorXd contact_end;
  Eigen::VectorXd ramp_origin;
  double ramp_start_time{};
  long ik_failures{};
};

struct PlannerSchedule {
  long control_steps{};
  long relinearizations{};
  long solves{};
  bool solved_this_tick{};
  double last_relinearization_time{-1e9};
  double last_solve_time{-1e9};
};

// The force command that generated the contact state measured on the following
// 1 ms control tick. Kept separately from the executor result so the
// simulator-contact diagnostic has a small, explicit data dependency.
struct OscContactForceCommand {
  bool valid{};
  double time{};
  bool used_full_contact_force{};
  std::array<double, 4> c3_normal_force_target{};
  std::array<double, 4> c3_normal_force_applied{};
  std::array<Eigen::Vector3d, 4> c3_force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> osc_force_command_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> c3_contact_point_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
};

// State of the optional measured-SAP-wrench feedback layer.  This first
// implementation intentionally corrects world-Z yaw only, while preserving
// C3's nominal full contact-force plan as the holding wrench.
struct YawWrenchFeedbackState {
  bool authority_lost{};
  double last_update_time{-1.0};
  double last_log_time{-1.0};
  double yaw_integral{};
  double yaw_error{};
  double c3_yaw_moment{};
  double sap_yaw_moment{};
  double requested_yaw_moment{};
  double allocated_yaw_moment{};
  double authority_test_start_time{-1.0};
  std::array<Eigen::Vector3d, 4> force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
};

class SqueezeApplication {
 public:
  explicit SqueezeApplication(SqueezeConfig config);
  int Run();

 private:
  std::array<bool, 4> DetectContacts() const;
  void LogContacts(double time) const;
  void LogRotation(double time) const;
  Eigen::VectorXd ComputeReachTorque(double time,
                                     const std::array<bool, 4>& touching);
  void StartPlanner(double time);
  void UpdatePlanner(double time);
  void UpdateManeuver(double time, const std::array<bool, 4>& touching);
  void UpdateTrackingReference(double time, double reference_time);
  void UpdateHorizonTarget(double reference_time);
  std::vector<double> ComputePlannedCubeVerticalContactForces();
  std::array<Eigen::Vector3d, 4> UpdateYawWrenchFeedback(
      double time, const C3ContactForcePlan& c3_contact_force_plan);
  Eigen::VectorXd ComputeExecutionTorque(double time);
  Eigen::VectorXd ComputeOscTorque(double time);
  Eigen::VectorXd ComputeTaskSpaceTorque();
  void ApplyAndAdvance(double time, const Eigen::VectorXd& torque);
  void PublishTrackingTelemetry();

  CubeReference CubeTarget(double reference_time) const;
  Eigen::VectorXd InterpolatedContactTarget(double time) const;

  SqueezeConfig config_;
  SimulationEnvironment environment_;
  GraspSetup grasp_;
  LcsModel lcs_model_;
  C3Planner planner_;
  ManeuverController maneuver_;
  ControlPhase phase_{ControlPhase::kReach};
  ReachState reach_;
  TrackingState tracking_;
  PlannerSchedule schedule_;
  OscContactForceCommand last_osc_contact_force_command_;
  YawWrenchFeedbackState yaw_wrench_feedback_;
  DesiredVelocityFilterState desired_velocity_filter_;
  Eigen::VectorXd last_osc_pd_torque_;
  Eigen::VectorXd osc_pd_crossfade_from_;
  int maneuver_handoff_generation_last_control_{};
  bool osc_pd_crossfade_active_{};
  double osc_pd_crossfade_start_time_{};
  bool cube_pinned_{true};
  double handoff_time_{-1.0};
  static constexpr double kControlDt = 0.001;
};

}  // namespace dairlib::allegro_grasp_c3
