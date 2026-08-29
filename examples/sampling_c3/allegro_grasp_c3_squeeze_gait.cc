#include "allegro_grasp_c3_squeeze_maneuver_controller.h"

#include <algorithm>
#include <iostream>

#include <drake/math/roll_pitch_yaw.h>

#include "allegro_grasp_c3_squeeze_maneuver.h"

namespace dairlib::allegro_grasp_c3 {

void ManeuverController::UpdateGait(
    double time, double reference_time,
    const std::array<bool, 4>& touching, const Eigen::VectorXd& state,
    const drake::math::RigidTransform<double>& cube_pose, C3Planner* planner,
    Eigen::VectorXd* contact_start, Eigen::VectorXd* contact_end) {
  if (gait_phase_ == GaitPhase::kDone) return;
  if (IsSpider() && ring_parked_positions_.size() == 0) {
    // The first spider-walk primitive begins from an established three-finger
    // grasp.  Keep the already-clear ring finger exactly where it is; there is
    // no reason to solve a new IK target until the later support-post step.
    ring_parked_positions_ =
        state.head(planner->dimensions().hand_positions);
  } else if (UsesParkedRing() && ring_parked_positions_.size() == 0) {
    bool parked_ok = false;
    ring_parked_positions_ = grasp_->SolveLegIk(
        cube_pose, 3, grasp_->relay_ring_park,
        state.head(planner->dimensions().hand_positions), &parked_ok);
    if (!parked_ok)
      std::cout << "[t=" << time << "] gait: ring park IK infeasible\n";
  }
  const std::vector<int> rotation_contacts =
      UsesParkedRing() ? std::vector<int>{0, 1, 2}
                       : std::vector<int>{0, 1, 2, 3};

  if (gait_phase_ == GaitPhase::kRotate) {
    const bool ready = gait_cycle_ > 0 || reference_time >= config_.gait_hold_time;
    if (!gait_entered_ && ready) {
      const double delta = IsSpider()
          ? config_.spider_yaw_delta
          : config_.gait_delta;
      gait_theta_start_ = gait_cycle_ * delta;
      gait_theta_target_ = (gait_cycle_ + 1) * delta;
      gait_rotate_start_ = reference_time;
      gait_entered_ = true;
      if (planner->active_fingers() != rotation_contacts)
        planner->Rebuild(rotation_contacts, state, time);
      ring_engaged_ = !UsesParkedRing();
      std::cout << "[t=" << time << "] "
                << (IsSpider() ? "spider small turn" : "gait cycle ")
                << (IsSpider() ? "" : std::to_string(gait_cycle_ + 1))
                << ": rotating about "
                << (IsSpider() ? "world +Z" : "cube +Y") << " to "
                << gait_theta_target_ * 180.0 / M_PI << " deg\n";
    }
    const double rotate_duration = IsSpider()
        ? config_.spider_yaw_duration
        : config_.gait_rotate_duration;
    if (gait_entered_ &&
        reference_time >= gait_rotate_start_ + rotate_duration +
                              config_.gait_hold_time) {
      if (IsSpider()) {
        gait_phase_ = GaitPhase::kMove;
        gait_leg_ = 0;
        gait_entered_ = false;
        std::cout << "[t=" << time
                  << "] spider small turn complete at "
                  << gait_theta_target_ * 180.0 / M_PI
                  << " deg yaw; placing ring support\n";
        return;
      }
      gait_phase_ = GaitPhase::kMove;
      gait_leg_ = 0;
      gait_entered_ = false;
    }
    return;
  }

  const int finger = gait_plan_[gait_leg_].first;
  const GaitLegKind kind = gait_plan_[gait_leg_].second;
  if (!gait_entered_ && time < gait_leg_done_time_ + config_.gait_leg_gap)
    return;

  if (!gait_entered_) {
    if (kind != GaitLegKind::kEngage) {
      std::vector<int> remaining;
      for (const int active : planner->active_fingers())
        if (active != finger) remaining.push_back(active);
      planner->Rebuild(remaining, state, time);
    }

    if (kind == GaitLegKind::kEngage) {
      gait_target_C_ = IsSpider() ? grasp_->spider_ring_hold
                                   : grasp_->relay_ring_hold;
    } else if (gait_realign_mode_) {
      gait_target_C_ = grasp_->initial_footprints[finger];
    } else {
      const drake::math::RotationMatrix<double> walk_back(
          drake::math::RollPitchYaw<double>(0.0, -config_.gait_delta, 0.0));
      gait_target_C_ = walk_back * grasp_->footprints[finger];
    }
    Eigen::Vector3d midpoint = kind == GaitLegKind::kEngage
                                   ? (IsSpider() ? grasp_->spider_ring_park
                                                 : grasp_->relay_ring_park)
                                   : 0.5 * (grasp_->footprints[finger] +
                                            gait_target_C_);
    if (kind != GaitLegKind::kEngage)
      midpoint.y() = -(GraspSetup::kCubeSize / 2.0 +
                       config_.regrasp_arc_clearance);
    bool destination_ok = false;
    bool midpoint_ok = false;
    Eigen::VectorXd q_mid;
    const auto solve_engage_target = [&]() {
      q_mid = grasp_->SolveLegIk(cube_pose, finger, midpoint,
                                  *contact_start, &midpoint_ok);
      // For the spider's ring-engagement leg, the outside waypoint is also a
      // better initial guess for the final near-face IK than the old, parked
      // posture. This follows the physical approach path before considering
      // a more central fallback point.
      const Eigen::VectorXd& destination_seed =
          kind == GaitLegKind::kEngage && IsSpider() ? q_mid
                                                      : *contact_start;
      gait_destination_ = grasp_->SolveLegIk(
          cube_pose, finger, gait_target_C_, destination_seed,
          &destination_ok);
    };
    solve_engage_target();

    if (kind == GaitLegKind::kEngage && IsSpider() &&
        (!destination_ok || !midpoint_ok)) {
      // The preferred point is safely inside incoming red but close to its
      // yellow edge. If it is just beyond the joint limits, walk it toward
      // red's face centre in 1 mm increments and use the first fully feasible
      // point. This is a safety policy: never execute a handoff from an IK
      // result that the solver itself rejected.
      const Eigen::Vector3d requested_target = gait_target_C_;
      const double toward_centre = gait_target_C_.y() < 0.0
          ? 1.0 : (gait_target_C_.y() > 0.0 ? -1.0 : 0.0);
      for (int step = 1; toward_centre != 0.0 && step <= 15 &&
                         (!destination_ok || !midpoint_ok); ++step) {
        gait_target_C_.y() = requested_target.y() +
            toward_centre * 0.001 * step;
        if ((toward_centre > 0.0 && gait_target_C_.y() > 0.0) ||
            (toward_centre < 0.0 && gait_target_C_.y() < 0.0)) {
          gait_target_C_.y() = 0.0;
        }
        midpoint = gait_target_C_;
        midpoint.x() = GraspSetup::kCubeSize / 2.0 +
                       config_.relay_ring_retract;
        destination_ok = false;
        midpoint_ok = false;
        solve_engage_target();
        if (gait_target_C_.x() == 0.0) break;
      }
      if (destination_ok && midpoint_ok) {
        std::cout << "[t=" << time
                  << "] spider: ring target adjusted toward face centre "
                  << "for reachable IK: red-y=" << gait_target_C_.y()
                  << " m\n";
      } else {
        gait_phase_ = GaitPhase::kDone;
        std::cout << "[t=" << time
                  << "] spider: no reachable ring-support target; "
                  << "stopping before handoff\n";
        return;
      }
    }
    gait_target_W_ = cube_pose * gait_target_C_;
    gait_trajectory_ = MakeThreeKnotJointTrajectory(
        state.head(planner->dimensions().hand_positions), q_mid,
        gait_destination_, config_.regrasp_duration);
    gait_trajectory_start_ = time;
    gait_left_surface_ = kind == GaitLegKind::kEngage;
    gait_touch_latched_ = false;
    gait_entered_ = true;
    if (!destination_ok || !midpoint_ok) {
      std::cout << "[t=" << time << "] gait: warning, finger " << finger
                << " IK infeasible (target=" << destination_ok
                << ", outside_waypoint=" << midpoint_ok << ")\n";
    }
    return;
  }

  const double elapsed = time - gait_trajectory_start_;
  if (elapsed > gait_trajectory_.end_time() &&
      static_cast<long>(time / 0.001) %
              std::max(1, config_.gait_seek_period_steps) ==
          0) {
    bool seek_ok = false;
    const Eigen::VectorXd updated = grasp_->SolveLegIk(
        cube_pose, finger, gait_target_C_, *contact_start, &seek_ok);
    if (seek_ok) {
      gait_destination_ = updated;
      gait_target_W_ = cube_pose * gait_target_C_;
    }
  }
  if (!gait_left_surface_ && !touching[finger]) gait_left_surface_ = true;
  const bool near_target =
      (grasp_->FingertipPosition(finger) - gait_target_W_).norm() <
      config_.regrasp_touch_tol;
  if (!gait_touch_latched_ && gait_left_surface_ && touching[finger] &&
      near_target) {
    gait_touch_latched_ = true;
    gait_touch_time_ = time;
    std::cout << "[t=" << time << "] gait: finger " << finger
              << " re-contacted; settling\n";
  }
  if (!gait_touch_latched_ &&
      elapsed > gait_trajectory_.end_time() + config_.gait_leg_timeout) {
    gait_phase_ = GaitPhase::kDone;
    std::cout << "[t=" << time << "] gait: finger " << finger
              << " failed to re-contact; stopping gait\n";
    return;
  }
  if (!gait_touch_latched_ ||
      time < gait_touch_time_ + config_.regrasp_settle_time)
    return;

  CompleteFinger(finger, gait_destination_, gait_target_C_, state, time,
                 planner, contact_start, contact_end);
  if (kind == GaitLegKind::kEngage) ring_engaged_ = true;
  gait_entered_ = false;
  gait_left_surface_ = false;
  gait_touch_latched_ = false;
  gait_leg_done_time_ = time;
  std::cout << "[t=" << time << "] gait: finger " << finger
            << " joined C3\n";

  if (++gait_leg_ < static_cast<int>(gait_plan_.size())) return;
  gait_leg_ = 0;
  if (IsSpider()) {
    gait_phase_ = GaitPhase::kDone;
    std::cout << "[t=" << time
              << "] spider ring support established; holding four contacts\n";
    return;
  }
  if (gait_realign_mode_) {
    gait_phase_ = GaitPhase::kDone;
    std::cout << "[t=" << time << "] gait: realignment complete\n";
  } else if (++gait_cycle_ >= config_.gait_cycles) {
    if (config_.gait_realign && config_.gait_scheme != "relay") {
      gait_realign_mode_ = true;
    } else {
      gait_phase_ = GaitPhase::kDone;
    }
  } else {
    gait_phase_ = GaitPhase::kRotate;
  }
}

}  // namespace dairlib::allegro_grasp_c3
