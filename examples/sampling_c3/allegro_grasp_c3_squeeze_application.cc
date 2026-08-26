#include "allegro_grasp_c3_squeeze_application.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "allegro_grasp_c3_squeeze_execution.h"
#include "allegro_grasp_c3_squeeze_diagnostics.h"
#include "allegro_grasp_c3_squeeze_horizon.h"
#include "allegro_grasp_c3_squeeze_reference.h"
#include "cube_kinematics.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;

}  // namespace

SqueezeApplication::SqueezeApplication(SqueezeConfig config)
    : config_(std::move(config)),
      environment_(config_),
      grasp_(config_, &environment_),
      lcs_model_(),
      planner_(config_, &lcs_model_),
      maneuver_(config_, &grasp_) {
  tracking_.contact_start = grasp_.contact_positions;
  tracking_.contact_end = grasp_.contact_positions;
  tracking_.ramp_origin = grasp_.contact_positions;
}

int SqueezeApplication::Run() {
  grasp_.PreviewInitialPoses(config_.preview);
  environment_.SetInitialState(grasp_.pregrasp_positions,
                               grasp_.initial_cube_positions);
  environment_.Initialize();

  for (double time = kControlDt; time < config_.sim_time;
       time += kControlDt) {
    environment_.PublishState();
    const std::array<bool, 4> touching = DetectContacts();
    LogContacts(time);
    VectorXd torque;
    if (phase_ == ControlPhase::kReach) {
      torque = ComputeReachTorque(time, touching);
    } else {
      UpdateManeuver(time, touching);
      UpdatePlanner(time);
      torque = ComputeExecutionTorque(time);
    }
    ApplyAndAdvance(time, torque);
  }
  return 0;
}

std::array<bool, 4> SqueezeApplication::DetectContacts() const {
  std::array<bool, 4> touching{false, false, false, false};
  const auto& contacts = environment_.contacts();
  for (int k = 0; k < contacts.num_point_pair_contacts(); ++k) {
    const auto& info = contacts.point_pair_contact_info(k);
    if (info.contact_force().norm() < config_.contact_force_thresh) continue;
    for (int finger = 0; finger < 4; ++finger) {
      const bool matches =
          (info.bodyA_index() == grasp_.tip_bodies[finger] &&
           info.bodyB_index() == grasp_.cube_body) ||
          (info.bodyB_index() == grasp_.tip_bodies[finger] &&
           info.bodyA_index() == grasp_.cube_body);
      touching[finger] = touching[finger] || matches;
    }
  }
  return touching;
}

void SqueezeApplication::LogContacts(double time) const {
  if (!config_.contact_force_log) return;
  const int period = std::max(
      1, static_cast<int>(std::lround(
             1.0 / (config_.contact_force_log_hz * kControlDt))));
  if (static_cast<long>(std::lround(time / kControlDt)) % period != 0) return;
  const auto& contacts = environment_.contacts();
  std::cout << "[t=" << time << "] contacts="
            << contacts.num_point_pair_contacts();
  for (int k = 0; k < contacts.num_point_pair_contacts(); ++k) {
    const auto& info = contacts.point_pair_contact_info(k);
    std::cout << " [" << environment_.plant().get_body(info.bodyA_index()).name()
              << "-" << environment_.plant().get_body(info.bodyB_index()).name()
              << " |f|=" << info.contact_force().norm() << "]";
  }
  std::cout << "\n";
}

VectorXd SqueezeApplication::ComputeReachTorque(
    double time, const std::array<bool, 4>& touching) {
  const VectorXd q = environment_.hand_positions();
  const VectorXd v = environment_.hand_velocities();
  for (int finger = 0; finger < grasp_.grasp_finger_count; ++finger) {
    if (!reach_.arrived[finger] && touching[finger] &&
        time > config_.contact_enable_t) {
      reach_.arrived[finger] = true;
      reach_.arrival_time[finger] = time;
      std::cout << "[t=" << time << "] finger " << finger << " arrived\n";
    }
  }

  const double sample_time = std::clamp(
      time, 0.0, grasp_.reach_trajectory.end_time());
  VectorXd q_target = grasp_.reach_trajectory.value(sample_time).col(0);
  VectorXd v_target = grasp_.reach_velocity.value(sample_time).col(0);
  for (int finger = 0; finger < grasp_.grasp_finger_count; ++finger) {
    if (!reach_.arrived[finger]) continue;
    const int start = GraspSetup::kFingerStarts[finger];
    q_target.segment(start, 4) = grasp_.contact_positions.segment(start, 4);
    v_target.segment(start, 4).setZero();
  }

  auto& plant = environment_.plant();
  const VectorXd gravity = plant.GetVelocitiesFromArray(
      environment_.hand_model(),
      -plant.CalcGravityGeneralizedForces(environment_.plant_context()));
  const bool all_three = reach_.arrived[0] && reach_.arrived[1] &&
                         reach_.arrived[2];
  const bool ring_ready = !config_.release_middle || reach_.arrived[3];
  const double last_arrival = *std::max_element(
      reach_.arrival_time.begin(), reach_.arrival_time.end());
  if (all_three && ring_ready &&
      time >= last_arrival + config_.handoff_settle_time) {
    StartPlanner(time);
  }
  return config_.kp * (q_target - q) + config_.kd * (v_target - v) + gravity;
}

void SqueezeApplication::StartPlanner(double time) {
  VectorXd target = VectorXd::Zero(planner_.dimensions().state);
  VectorXd desired_hand = environment_.hand_positions();
  for (int finger = 0; finger < grasp_.grasp_finger_count; ++finger) {
    const int start = GraspSetup::kFingerStarts[finger];
    desired_hand.segment(start, 4) =
        grasp_.contact_positions.segment(start, 4);
  }
  target.head(planner_.dimensions().hand_positions) = desired_hand;
  target.segment(planner_.dimensions().hand_positions, 7) =
      grasp_.initial_cube_positions;
  planner_.SetBaseTarget(target);
  planner_.Rebuild(config_.release_middle
                       ? std::vector<int>{0, 1, 2, 3}
                       : std::vector<int>{0, 1, 2},
                   environment_.state(), time);
  phase_ = ControlPhase::kC3;
  handoff_time_ = time;
  std::cout << "[t=" << time << "] all fingers settled -> C3 handoff\n";
}

CubeReference SqueezeApplication::CubeTarget(double reference_time) const {
  CubeMotionReferenceConfig request;
  request.motion_mode = config_.cube_motion_mode;
  request.translation_offset = Vector3d(
      config_.cube_move_dx, config_.cube_move_dy, config_.cube_move_dz);
  request.rpy_offset = Vector3d(
      config_.cube_move_roll, config_.cube_move_pitch, config_.cube_move_yaw);
  request.move_duration = config_.cube_move_duration;
  request.sine_period = config_.cube_move_period;
  maneuver_.ConfigureCubeReference(&request);
  return MakeCubeReference(grasp_.initial_cube_pose, request, reference_time);
}

void SqueezeApplication::UpdateManeuver(
    double time, const std::array<bool, 4>& touching) {
  if (cube_pinned_) return;
  const double reference_time = std::max(0.0, time - (handoff_time_ + 0.5));
  maneuver_.Update(
      time, reference_time, touching, environment_.state(),
      CubePoseFromPositions(environment_.cube_positions()), &planner_,
      &tracking_.contact_start, &tracking_.contact_end);
}

void SqueezeApplication::UpdatePlanner(double time) {
  const bool relinearize = config_.relinearize &&
      schedule_.control_steps % std::max(1, config_.relin_period_steps) == 0;
  const bool solve =
      schedule_.control_steps % std::max(1, config_.c3_period_steps) == 0;
  const bool track =
      schedule_.control_steps % std::max(1, config_.track_ik_period_steps) == 0;
  ++schedule_.control_steps;
  const double reference_time = std::max(0.0, time - (handoff_time_ + 0.5));

  if (relinearize) {
    planner_.Relinearize(environment_.state());
    schedule_.last_relinearization_time = time;
    ++schedule_.relinearizations;
  }
  if (config_.track_cube_contact && !cube_pinned_ && track)
    UpdateTrackingReference(time, reference_time);
  UpdateHorizonTarget(reference_time);
  if (solve) {
    planner_.Solve(environment_.state());
    schedule_.last_solve_time = time;
    ++schedule_.solves;
    if (config_.lambda_map_debug &&
        planner_.active_fingers() == std::vector<int>({0, 1, 2})) {
      const PlannerDimensions& d = planner_.dimensions();
      PrintCubeLambdaMap(
          planner_.implementation().GetLCS(), time,
          {d.contacts, d.hand_positions, d.hand_velocities, d.positions,
           d.state, d.lambda, config_.num_friction_directions,
           planner_.implementation().GetLambdaScaling(), config_.alpha_m,
           config_.c3_dt, config_.contact_model});
    }
  }
}

void SqueezeApplication::UpdateTrackingReference(double time,
                                                  double reference_time) {
  const auto measured = CubePoseFromPositions(environment_.cube_positions());
  const auto now = CubeTarget(reference_time).pose;
  const auto end = CubeTarget(reference_time + config_.N * config_.c3_dt).pose;
  const auto safe_now = ClampIkLead(now, measured,
                                    config_.cube_ik_lead_pos_max,
                                    config_.cube_ik_lead_rot_max);
  const auto safe_end = ClampIkLead(end, measured,
                                    config_.cube_ik_lead_pos_max,
                                    config_.cube_ik_lead_rot_max);
  tracking_.ramp_origin = tracking_.contact_start;
  tracking_.ramp_start_time = time;
  bool start_ok = false;
  bool end_ok = false;
  const VectorXd start = grasp_.SolveContactIk(
      safe_now, tracking_.contact_start, maneuver_.ring_engaged(),
      &start_ok);
  const VectorXd finish = grasp_.SolveContactIk(
      safe_end, start_ok ? start : tracking_.contact_start,
      maneuver_.ring_engaged(), &end_ok);
  if (start_ok) tracking_.contact_start = start;
  if (end_ok) tracking_.contact_end = finish;
  if (!start_ok || !end_ok) ++tracking_.ik_failures;
}

void SqueezeApplication::UpdateHorizonTarget(double reference_time) {
  const bool track = config_.track_cube_contact && !cube_pinned_;
  const bool motion = config_.cube_motion_mode != "none" && !cube_pinned_;
  if (!track && !motion) return;

  HorizonStateLayout layout{
      config_.N, planner_.dimensions().hand_positions,
      planner_.dimensions().positions,
      planner_.dimensions().hand_velocities,
      std::vector<int>(GraspSetup::kFingerStarts.begin(),
                       GraspSetup::kFingerStarts.end())};
  HorizonTargetRequest request;
  request.base_state = planner_.base_target();
  request.track_contact_posture = track;
  request.contact_posture_start = tracking_.contact_start;
  request.contact_posture_end = tracking_.contact_end;
  request.num_grasp_fingers = grasp_.grasp_finger_count;
  if (motion) {
    request.cube_references.reserve(config_.N + 1);
    for (int knot = 0; knot <= config_.N; ++knot)
      request.cube_references.push_back(
          CubeTarget(reference_time + knot * config_.c3_dt));
  }
  planner_.UpdateTarget(BuildHorizonTargetTrajectory(layout, request));
  if (config_.show_cube_target)
    environment_.meshcat()->SetTransform("/cube_target",
                                          CubeTarget(reference_time).pose);
}

VectorXd SqueezeApplication::InterpolatedContactTarget(double time) const {
  const double period = std::max(1, config_.track_ik_period_steps) * kControlDt;
  const double fraction = std::clamp(
      (time - tracking_.ramp_start_time) / period, 0.0, 1.0);
  return (1.0 - fraction) * tracking_.ramp_origin +
         fraction * tracking_.contact_start;
}

VectorXd SqueezeApplication::ComputeExecutionTorque(double time) {
  if (config_.exec_mode == "direct") return planner_.FirstPhysicalInput();
  if (config_.exec_mode == "osc") return ComputeOscTorque(time);
  if (cube_pinned_) {
    auto& plant = environment_.plant();
    const VectorXd gravity = plant.GetVelocitiesFromArray(
        environment_.hand_model(),
        -plant.CalcGravityGeneralizedForces(environment_.plant_context()));
    return config_.kp * (grasp_.contact_positions -
                         environment_.hand_positions()) -
           config_.kd * environment_.hand_velocities() + gravity;
  }
  return ComputeTaskSpaceTorque();
}

VectorXd SqueezeApplication::ComputeOscTorque(double time) {
  VectorXd desired = environment_.hand_positions();
  const bool use_ik = config_.osc_target_source == "ik" ||
      (config_.osc_target_source == "auto" && config_.track_cube_contact);
  const std::vector<VectorXd> plan = planner_.state_solution();
  if (use_ik) desired = InterpolatedContactTarget(time);
  else if (plan.size() > 1)
    desired = plan[1].head(planner_.dimensions().hand_positions);
  maneuver_.OverrideJointTarget(time, &desired);

  const OscExecutorResult result = ComputeOscExecutorTorque(
      OscExecutorRequest{
          environment_.plant(), environment_.plant_context(),
          environment_.hand_model(), environment_.cube_model(),
          grasp_.tip_bodies, GraspSetup::kFingerStarts,
          planner_.active_fingers(), planner_.normal_groups(),
          planner_.force_crossfade_origins(), planner_.finger_joined_times(),
          desired, environment_.hand_positions(),
          environment_.hand_velocities(), planner_.FirstPhysicalForce(), time,
          kControlDt, config_.osc_kp, config_.osc_kd,
          config_.osc_qd_filter_tau, config_.gait_force_ramp_time,
          config_.lambda_torque_scale},
      &desired_velocity_filter_);
  return result.torque;
}

VectorXd SqueezeApplication::ComputeTaskSpaceTorque() {
  VectorXd targets(9);
  const std::vector<VectorXd> plan = planner_.state_solution();
  if (config_.fk_target && plan.size() > 1) {
    targets = grasp_.FingertipPositionsForHand(
        plan[1].head(planner_.dimensions().hand_positions));
  } else {
    for (int i = 0; i < 3; ++i)
      targets.segment<3>(3 * i) =
          grasp_.initial_cube_pose * grasp_.footprints[i];
  }
  const std::vector<VectorXd> dual = planner_.dual_delta_solution();
  return ComputeTaskSpaceExecutorTorque(TaskSpaceExecutorRequest{
      environment_.plant(), environment_.plant_context(),
      environment_.hand_model(), environment_.cube_model(),
      {grasp_.tip_bodies[0], grasp_.tip_bodies[1], grasp_.tip_bodies[2]},
      planner_.normal_groups(), targets, environment_.hand_velocities(),
      dual[0].segment(
          planner_.dimensions().state, planner_.dimensions().lambda),
      planner_.FirstPhysicalInput(), config_.task_kp, config_.task_kd,
      config_.force_floor, config_.exec_grav_comp});
}

void SqueezeApplication::PublishTrackingTelemetry() {
  const VectorXd actual = environment_.hand_positions().segment<4>(0);
  const std::vector<VectorXd> plan = planner_.initialized()
                                          ? planner_.state_solution()
                                          : std::vector<VectorXd>{};
  VectorXd plan0;
  VectorXd plan1;
  if (!plan.empty()) plan0 = plan[0].segment<4>(0);
  if (plan.size() > 1) plan1 = plan[1].segment<4>(0);
  environment_.PublishIndexTracking(
      actual, plan0.size() == 4 ? &plan0 : nullptr,
      plan1.size() == 4 ? &plan1 : nullptr);
}

void SqueezeApplication::ApplyAndAdvance(double time, const VectorXd& torque) {
  const VectorXd applied = phase_ == ControlPhase::kC3
                               ? ClampTorque(torque, config_.tau_max)
                               : torque;
  environment_.ApplyHandTorque(applied);
  PublishTrackingTelemetry();
  environment_.AdvanceTo(time);
  if (cube_pinned_ && phase_ == ControlPhase::kC3 &&
      time >= handoff_time_ + 0.5) {
    cube_pinned_ = false;
    std::cout << "[t=" << time << "] cube pin released\n";
  }
  if (cube_pinned_) environment_.PinCube(grasp_.initial_cube_positions);
}

}  // namespace dairlib::allegro_grasp_c3
