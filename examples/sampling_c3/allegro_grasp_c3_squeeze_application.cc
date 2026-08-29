#include "allegro_grasp_c3_squeeze_application.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <Eigen/Geometry>

#include <drake/math/roll_pitch_yaw.h>

#include "allegro_grasp_c3_squeeze_execution.h"
#include "allegro_grasp_c3_squeeze_diagnostics.h"
#include "allegro_grasp_c3_squeeze_horizon.h"
#include "allegro_grasp_c3_squeeze_reference.h"
#include "cube_kinematics.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;

constexpr double kRotationAxisEpsilon = 1e-6;

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
  ring_engaged_last_control_ = maneuver_.ring_engaged();
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
    LogRotation(time);
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
  PrintSapContactForceDiagnostic({
      time,
      last_osc_normal_command_.time,
      last_osc_normal_command_.valid,
      config_.lambda_torque_scale,
      SummarizeSapFingertipCubeContacts(
          environment_.contacts(), grasp_.cube_body, grasp_.tip_bodies),
      last_osc_normal_command_.c3_normal_force_target,
      last_osc_normal_command_.c3_normal_force_applied});
}

void SqueezeApplication::LogRotation(double time) const {
  if (!config_.rot_log || phase_ != ControlPhase::kC3) return;
  const int period = std::max(
      1, static_cast<int>(std::lround(
             std::max(kControlDt, config_.rot_log_period) / kControlDt)));
  if (static_cast<long>(std::lround(time / kControlDt)) % period != 0) return;

  // The raw reference, rather than the IK-lead-clamped pose, is what C3 and
  // the target ghost receive. Express command and measurement in the initial
  // cube frame so their rotation angles and axes have the same reference.
  const double reference_time = std::max(0.0, time - (handoff_time_ + 0.5));
  const auto commanded = CubeTarget(reference_time).pose;
  const auto measured = CubePoseFromPositions(environment_.cube_positions());
  const Eigen::Matrix3d R_WC0 = grasp_.initial_cube_pose.rotation().matrix();
  const Eigen::Matrix3d R_C0C_cmd =
      R_WC0.transpose() * commanded.rotation().matrix();
  const Eigen::Matrix3d R_C0C_meas =
      R_WC0.transpose() * measured.rotation().matrix();
  const Eigen::Matrix3d R_cmd_meas =
      commanded.rotation().matrix().transpose() * measured.rotation().matrix();
  // Left-relative rotations expose rotation about fixed world axes.  In
  // particular, yaw() below is the signed progress of the spider primitive
  // about world Z even when the cube's initial orientation is not identity.
  const drake::math::RollPitchYaw<double> command_world_delta(
      commanded.rotation() * grasp_.initial_cube_pose.rotation().inverse());
  const drake::math::RollPitchYaw<double> measured_world_delta(
      measured.rotation() * grasp_.initial_cube_pose.rotation().inverse());
  const double yaw_error = std::remainder(
      measured_world_delta.yaw_angle() - command_world_delta.yaw_angle(),
      2.0 * M_PI);

  const Eigen::AngleAxisd command_rotation(R_C0C_cmd);
  const Eigen::AngleAxisd measured_rotation(R_C0C_meas);
  const Eigen::AngleAxisd residual_rotation(R_cmd_meas);
  const bool have_axes =
      command_rotation.angle() > kRotationAxisEpsilon &&
      measured_rotation.angle() > kRotationAxisEpsilon;
  const double axis_drift = have_axes
      ? std::acos(std::clamp(command_rotation.axis().dot(
                                  measured_rotation.axis()),
                              -1.0, 1.0))
      : 0.0;
  const Vector3d position_drift_C0 = R_WC0.transpose() *
      (measured.translation() - grasp_.initial_cube_pose.translation());
  const double position_error =
      (measured.translation() - commanded.translation()).norm();

  std::ostringstream line;
  line << std::fixed << std::setprecision(3)
       << "[ROTATION t=" << time << " ref=" << reference_time
       << " pinned=" << cube_pinned_ << "] "
       << std::setprecision(6)
       << "cmd_angle=" << command_rotation.angle() << " rad "
       << "meas_angle=" << measured_rotation.angle() << " rad "
       << "residual_angle=" << residual_rotation.angle() << " rad "
       << "cmd_world_yaw=" << command_world_delta.yaw_angle() << " rad "
       << "meas_world_yaw=" << measured_world_delta.yaw_angle() << " rad "
       << "yaw_error=" << yaw_error << " rad "
       << "meas_world_roll=" << measured_world_delta.roll_angle() << " rad "
       << "meas_world_pitch=" << measured_world_delta.pitch_angle() << " rad "
       << "axis_drift=";
  if (have_axes) {
    line << axis_drift << " rad ";
  } else {
    line << "n/a ";
  }
  line << "pos_drift_C0=[" << position_drift_C0.transpose() << "] m "
       << "|pos_drift|=" << position_drift_C0.norm() << " m "
       << "pos_error=" << position_error << " m";
  std::cout << line.str() << "\n";
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
  schedule_.solved_this_tick = false;
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
    schedule_.solved_this_tick = true;
    schedule_.last_solve_time = time;
    ++schedule_.solves;
    if (config_.legacy_log && !config_.contact_force_log) {
      const PlannerDimensions& d = planner_.dimensions();
      PrintLegacyC3Plan({
          time, config_.c3_dt, d.hand_positions + 6,
          planner_.implementation().GetLambdaScaling(), planner_.input_scale(),
          planner_.active_fingers(), planner_.normal_groups(),
          planner_.state_solution(), planner_.input_solution(),
          planner_.force_solution(), ComputePlannedCubeVerticalContactForces()});
    }
    if (config_.c3_joint_plan_log) {
      PrintC3JointPlan({time, planner_.dimensions().hand_positions,
                        environment_.hand_positions(),
                        planner_.state_solution()});
    }
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

std::vector<double> SqueezeApplication::ComputePlannedCubeVerticalContactForces() {
  const PlannerDimensions& d = planner_.dimensions();
  const std::vector<VectorXd> force_plan = planner_.force_solution();
  const auto& lcs = planner_.implementation().GetLCS();
  const std::vector<Eigen::MatrixXd>& D = lcs.D();
  std::vector<VectorXd> cube_z_rows;
  cube_z_rows.reserve(force_plan.size());
  const int cube_z_velocity_row = d.positions + d.hand_velocities + 5;
  const double lambda_scaling = planner_.implementation().GetLambdaScaling();
  for (size_t knot = 0; knot < force_plan.size() && !D.empty(); ++knot) {
    const Eigen::MatrixXd& D_k = D.at(std::min(knot, D.size() - 1));
    if (cube_z_velocity_row >= D_k.rows()) break;
    cube_z_rows.push_back(D_k.row(cube_z_velocity_row).transpose() /
                          lambda_scaling);
  }
  auto& plant = lcs_model_.plant();
  const auto cube_body = plant.GetBodyIndices(lcs_model_.cube_model()).at(0);
  const double cube_mass = plant.get_body(cube_body).get_mass(lcs_model_.context());
  const int first_contact_force =
      config_.contact_model == "stewart_and_trinkle" ? d.contacts : 0;
  return ComputePlannedVerticalContactForces({
      cube_mass, config_.c3_dt, first_contact_force, std::move(cube_z_rows),
      force_plan});
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
  const bool motion =
      (config_.cube_motion_mode != "none" || config_.gait) && !cube_pinned_;
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

  OscExecutorResult result = ComputeOscExecutorTorque(
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
  const bool ring_engaged = maneuver_.ring_engaged();
  if (ring_engaged && !ring_engaged_last_control_ &&
      last_osc_pd_torque_.size() == result.pd_torque.size()) {
    // Four-contact IK changes the desired posture of the whole hand, not just
    // ring.  Preserve the applied pre-handoff PD torque and blend toward the
    // new four-contact value so that a valid ring touchdown does not kick the
    // cube into a new tilted static-friction equilibrium.
    osc_pd_crossfade_from_ = last_osc_pd_torque_;
    osc_pd_crossfade_start_time_ = time;
    osc_pd_crossfade_active_ = true;
  }
  ring_engaged_last_control_ = ring_engaged;
  if (osc_pd_crossfade_active_) {
    const double blend = config_.gait_torque_ramp_time <= 0.0
        ? 1.0
        : std::clamp((time - osc_pd_crossfade_start_time_) /
                         config_.gait_torque_ramp_time,
                     0.0, 1.0);
    const VectorXd blended_pd =
        (1.0 - blend) * osc_pd_crossfade_from_ + blend * result.pd_torque;
    result.torque += blended_pd - result.pd_torque;
    result.pd_torque = blended_pd;
    if (blend >= 1.0) osc_pd_crossfade_active_ = false;
  }
  last_osc_pd_torque_ = result.pd_torque;
  last_osc_normal_command_ = {
      true, time, result.normal_force_target, result.normal_force_applied};
  if (config_.osc_torque_split_log && schedule_.solved_this_tick) {
    const VectorXd commanded = ClampTorque(result.torque, config_.tau_max);
    const auto command_projection = ProjectHandTorqueToFingertipForces({
        environment_.plant(), environment_.plant_context(),
        environment_.hand_model(), grasp_.tip_bodies, GraspSetup::kFingerStarts,
        commanded});
    const auto pd_projection = ProjectHandTorqueToFingertipForces({
        environment_.plant(), environment_.plant_context(),
        environment_.hand_model(), grasp_.tip_bodies, GraspSetup::kFingerStarts,
        result.pd_torque});
    PrintOscTorqueSplit(
        {time,
         GraspSetup::kFingerStarts,
         commanded,
         result.pd_torque,
         result.force_torque,
         command_projection.force_world,
         pd_projection.force_world,
         result.normal_direction_world,
         command_projection.relative_torque_residual,
         pd_projection.relative_torque_residual,
         result.normal_force_target,
         result.normal_force_applied});
  }
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
