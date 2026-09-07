#include "allegro_grasp_c3_squeeze_maneuver_controller.h"

#include <algorithm>

namespace dairlib::allegro_grasp_c3 {

ManeuverController::ManeuverController(const SqueezeConfig& config,
                                       GraspSetup* grasp)
    : config_(config), grasp_(grasp),
      ring_engaged_(grasp->grasp_finger_count == 4) {
  DRAKE_DEMAND(grasp_ != nullptr);
  if (config_.gait_scheme == "relay") {
    gait_plan_ = {{3, GaitLegKind::kEngage}};
  } else if (config_.gait_scheme == "spider") {
    // Establish ring on yellow a little above face centre, then stop in the
    // four-contact hold. Index remains on yellow until a later maneuver
    // explicitly asks to move it.
    // With the four-finger spider start, ring is already on the cube and no
    // placement trajectory is needed after the yaw.
    gait_plan_ = config_.gait && grasp->grasp_finger_count == 4
                     ? std::vector<std::pair<int, GaitLegKind>>{}
                     : std::vector<std::pair<int, GaitLegKind>>{
                           {3, GaitLegKind::kEngage}};
    if (config_.spider_index_crawl_after_ring)
      gait_plan_.push_back({0, GaitLegKind::kSpiderCrawl});
  } else if (config_.gait_scheme == "triangle") {
    gait_plan_ = {{3, GaitLegKind::kRegrasp},
                  {1, GaitLegKind::kRegrasp},
                  {0, GaitLegKind::kRegrasp}};
  }
}

bool ManeuverController::UsesParkedRing() const {
  return config_.gait_scheme == "relay" ||
         (IsSpider() && grasp_->grasp_finger_count < 4);
}

bool ManeuverController::IsSpider() const {
  return config_.gait_scheme == "spider";
}

void ManeuverController::Update(
    double time, double reference_time,
    const std::array<bool, 4>& touching, const Eigen::VectorXd& state,
    const drake::math::RigidTransform<double>& cube_pose,
    double sap_vertical_contact_force, double cube_weight, C3Planner* planner,
    Eigen::VectorXd* contact_start, Eigen::VectorXd* contact_end) {
  if (config_.gait) {
    UpdateGait(time, reference_time, touching, state, cube_pose,
               sap_vertical_contact_force, cube_weight, planner, contact_start,
               contact_end);
  } else if (config_.release_middle) {
    UpdateRelease(time, reference_time, touching, state, planner,
                  contact_start, contact_end);
  }
}

void ManeuverController::CompleteFinger(
    int finger, const Eigen::VectorXd& destination,
    const Eigen::Vector3d& footprint, const Eigen::VectorXd& state,
    double time, C3Planner* planner, Eigen::VectorXd* contact_start,
    Eigen::VectorXd* contact_end) {
  grasp_->footprints[finger] = footprint;
  const int start = GraspSetup::kFingerStarts[finger];
  planner->SetFingerTarget(finger, destination.segment<4>(start));
  contact_start->segment<4>(start) = destination.segment<4>(start);
  contact_end->segment<4>(start) = destination.segment<4>(start);
  std::vector<int> contacts = planner->active_fingers();
  if (std::find(contacts.begin(), contacts.end(), finger) == contacts.end())
    contacts.push_back(finger);
  std::sort(contacts.begin(), contacts.end());
  planner->Rebuild(contacts, state, time);
  ++handoff_generation_;
}

void ManeuverController::ConfigureCubeReference(
    CubeMotionReferenceConfig* reference) const {
  if (!config_.gait) return;
  reference->gait_enabled = true;
  reference->gait_rotate_start_time = gait_rotate_start_;
  reference->gait_rotate_duration = IsSpider()
      ? config_.spider_yaw_duration
      : config_.gait_rotate_duration;
  reference->gait_theta_start = gait_theta_start_;
  reference->gait_theta_target = gait_theta_target_;
  reference->gait_rotation_frame = IsSpider()
      ? GaitRotationFrame::kWorldZ
      : GaitRotationFrame::kCubeY;
}

bool ManeuverController::ring_engaged() const { return ring_engaged_; }

int ManeuverController::handoff_generation() const {
  return handoff_generation_;
}

std::array<Eigen::Vector3d, 4>
ManeuverController::OscPressDirectionsInCube() const {
  std::array<Eigen::Vector3d, 4> result{
      Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(0, -1, 0), Eigen::Vector3d(0, 1, 0)};
  // The spider ring now returns to yellow / -Y, so its existing +Y press
  // direction is also its inward normal direction after it joins C3.
  return result;
}

void ManeuverController::OverrideJointTarget(
    double time, Eigen::VectorXd* desired) const {
  const bool ring_is_moving =
      gait_phase_ == GaitPhase::kMove && (gait_entered_ || gait_retreating_) &&
      !gait_plan_.empty() && gait_plan_[gait_leg_].first == 3;
  if (UsesParkedRing() && !ring_engaged_ &&
      ring_parked_positions_.size() == desired->size() &&
      !ring_is_moving) {
    const int start = GraspSetup::kFingerStarts[3];
    desired->segment<4>(start) = ring_parked_positions_.segment<4>(start);
  }
  if (middle_release_only_) {
    const int start = GraspSetup::kFingerStarts[1];
    desired->segment<4>(start) =
        grasp_->release_middle_positions.segment<4>(start);
  }
  const std::array<int, 3> release_fingers{3, 1, 0};
  for (int step = 0; step < 3; ++step) {
    const RecontactState& state = release_steps_[step];
    if (!state.started || state.completed) continue;
    const double sample = std::clamp(time - state.start_time, 0.0,
                                     state.trajectory.end_time());
    const Eigen::VectorXd q = state.trajectory.value(sample).col(0);
    const int finger = release_fingers[step];
    const int start = GraspSetup::kFingerStarts[finger];
    desired->segment<4>(start) = q.segment<4>(start);
  }
  if (config_.gait && gait_phase_ == GaitPhase::kMove && gait_entered_) {
    const int finger = gait_plan_[gait_leg_].first;
    Eigen::VectorXd q;
    if (gait_retreating_) {
      const double elapsed = std::clamp(
          time - gait_retreat_start_time_, 0.0,
          gait_retreat_trajectory_.end_time());
      q = gait_retreat_trajectory_.value(elapsed).col(0);
    } else if (gait_touch_hold_active_ &&
               gait_touch_hold_positions_.size() == desired->size()) {
      q = gait_touch_hold_positions_;
    } else {
      const double elapsed = time - gait_trajectory_start_;
      q = elapsed >= gait_trajectory_.end_time()
              ? gait_destination_
              : gait_trajectory_.value(std::max(0.0, elapsed)).col(0);
    }
    const int start = GraspSetup::kFingerStarts[finger];
    desired->segment<4>(start) = q.segment<4>(start);
  }
}

}  // namespace dairlib::allegro_grasp_c3
