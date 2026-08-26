#include "allegro_grasp_c3_squeeze_maneuver_controller.h"

#include <iostream>

#include "allegro_grasp_c3_squeeze_maneuver.h"

namespace dairlib::allegro_grasp_c3 {

void ManeuverController::UpdateRelease(
    double time, double reference_time,
    const std::array<bool, 4>& touching, const Eigen::VectorXd& state,
    C3Planner* planner, Eigen::VectorXd* contact_start,
    Eigen::VectorXd* contact_end) {
  RecontactState& ring = release_steps_[0];
  if (!ring.started && !middle_release_only_ &&
      reference_time >= config_.release_middle_t) {
    if (config_.release_finger == "middle") {
      planner->Rebuild({0, 2, 3}, state, time);
      middle_release_only_ = true;
      std::cout << "[t=" << time
                << "] release: middle retracted; contacts=index,thumb,ring\n";
      return;
    }
    planner->Rebuild({0, 1, 2}, state, time);
    ring.trajectory = MakeThreeKnotJointTrajectory(
        state.head(planner->dimensions().hand_positions),
        grasp_->ring_regrasp_mid_positions, grasp_->ring_regrasp_positions,
        config_.regrasp_duration);
    ring.start_time = time;
    ring.started = true;
    std::cout << "[t=" << time
              << "] release: ring moving; contacts=index,middle,thumb\n";
  }
  if (middle_release_only_ || !ring.started) return;

  const std::array<int, 3> fingers{3, 1, 0};
  const std::array<const Eigen::VectorXd*, 3> destinations{
      &grasp_->ring_regrasp_positions, &grasp_->middle_regrasp_positions,
      &grasp_->index_regrasp_positions};
  const std::array<const Eigen::VectorXd*, 3> midpoints{
      &grasp_->ring_regrasp_mid_positions,
      &grasp_->middle_regrasp_mid_positions,
      &grasp_->index_regrasp_mid_positions};
  const std::array<std::vector<int>, 3> reduced_contacts{
      std::vector<int>{0, 1, 2}, std::vector<int>{0, 2, 3},
      std::vector<int>{1, 2, 3}};

  for (int step = 0; step < 3; ++step) {
    RecontactState& current = release_steps_[step];
    if (step > 0 && release_steps_[step - 1].completed && !current.started) {
      planner->Rebuild(reduced_contacts[step], state, time);
      current.trajectory = MakeThreeKnotJointTrajectory(
          state.head(planner->dimensions().hand_positions), *midpoints[step],
          *destinations[step], config_.regrasp_duration);
      current.start_time = time;
      current.started = true;
      std::cout << "[t=" << time << "] release: finger " << fingers[step]
                << " moving under three-contact support\n";
    }
    if (!current.started || current.completed) continue;

    const Eigen::Vector3d target_world =
        grasp_->initial_cube_pose * grasp_->relay_regrasp_targets[step];
    const bool near_target =
        (grasp_->FingertipPosition(fingers[step]) - target_world).norm() <
        config_.regrasp_touch_tol;
    const RecontactEvent event = UpdateRecontactState(
        touching[fingers[step]], near_target, time,
        config_.regrasp_settle_time, &current.left_surface, &current.latched,
        &current.latch_time);
    if (event == RecontactEvent::kContactLatched) {
      std::cout << "[t=" << time << "] release: finger " << fingers[step]
                << " re-contacted; settling\n";
    }
    if (event != RecontactEvent::kSettled) return;

    CompleteFinger(fingers[step], *destinations[step],
                   grasp_->relay_regrasp_targets[step], state, time, planner,
                   contact_start, contact_end);
    current.completed = true;
    ring_engaged_ = true;
    std::cout << "[t=" << time << "] release: finger " << fingers[step]
              << " rejoined C3\n";
    return;
  }
}

}  // namespace dairlib::allegro_grasp_c3
