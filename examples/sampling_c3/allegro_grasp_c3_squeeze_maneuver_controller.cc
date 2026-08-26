#include "allegro_grasp_c3_squeeze_maneuver_controller.h"

#include <algorithm>

namespace dairlib::allegro_grasp_c3 {

ManeuverController::ManeuverController(const SqueezeConfig& config,
                                       GraspSetup* grasp)
    : config_(config), grasp_(grasp),
      ring_engaged_(grasp->grasp_finger_count == 4) {
  DRAKE_DEMAND(grasp_ != nullptr);
  gait_plan_ = config_.gait_scheme == "relay"
                   ? std::vector<std::pair<int, GaitLegKind>>{
                         {3, GaitLegKind::kEngage}}
                   : std::vector<std::pair<int, GaitLegKind>>{
                         {3, GaitLegKind::kRegrasp},
                         {1, GaitLegKind::kRegrasp},
                         {0, GaitLegKind::kRegrasp}};
}

void ManeuverController::Update(
    double time, double reference_time,
    const std::array<bool, 4>& touching, const Eigen::VectorXd& state,
    const drake::math::RigidTransform<double>& cube_pose, C3Planner* planner,
    Eigen::VectorXd* contact_start, Eigen::VectorXd* contact_end) {
  if (config_.gait) {
    UpdateGait(time, reference_time, touching, state, cube_pose, planner,
               contact_start, contact_end);
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
}

void ManeuverController::ConfigureCubeReference(
    CubeMotionReferenceConfig* reference) const {
  if (!config_.gait) return;
  reference->gait_enabled = true;
  reference->gait_rotate_start_time = gait_rotate_start_;
  reference->gait_rotate_duration = config_.gait_rotate_duration;
  reference->gait_theta_start = gait_theta_start_;
  reference->gait_theta_target = gait_theta_target_;
}

bool ManeuverController::ring_engaged() const { return ring_engaged_; }

void ManeuverController::OverrideJointTarget(
    double time, Eigen::VectorXd* desired) const {
  if (config_.gait_scheme == "relay" && !ring_engaged_ &&
      ring_parked_positions_.size() == desired->size() &&
      !(gait_phase_ == GaitPhase::kMove && gait_entered_ &&
        gait_plan_[gait_leg_].first == 3)) {
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
    const double elapsed = time - gait_trajectory_start_;
    const Eigen::VectorXd q = elapsed >= gait_trajectory_.end_time()
                                  ? gait_destination_
                                  : gait_trajectory_.value(
                                        std::max(0.0, elapsed)).col(0);
    const int start = GraspSetup::kFingerStarts[finger];
    desired->segment<4>(start) = q.segment<4>(start);
  }
}

}  // namespace dairlib::allegro_grasp_c3
