#include "allegro_grasp_c3_squeeze_application.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <Eigen/Geometry>

#include <drake/geometry/shape_specification.h>
#include <drake/math/rigid_transform.h>
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
      lcs_model_(config_.cube_size_scale),
      planner_(config_, &lcs_model_),
      maneuver_(config_, &grasp_) {
  tracking_.contact_start = grasp_.contact_positions;
  tracking_.contact_end = grasp_.contact_positions;
  tracking_.ramp_origin = grasp_.contact_positions;
  maneuver_handoff_generation_last_control_ = maneuver_.handoff_generation();
}

int SqueezeApplication::Run() {
  grasp_.PreviewInitialPoses(config_.preview);
  environment_.SetInitialState(grasp_.pregrasp_positions,
                               grasp_.initial_cube_positions);
  environment_.Initialize();

  const bool show_spider_best_ring_target =
      config_.show_cube_target && config_.gait &&
      config_.gait_scheme == "spider";
  const auto update_spider_best_ring_target = [this]() {
    const auto cube_pose =
        CubePoseFromPositions(environment_.cube_positions());
    const Vector3d target_W = cube_pose * grasp_.spider_ring_hold;
    environment_.meshcat()->SetTransform(
        "/spider/virtual_ring_search_best_target",
        drake::math::RigidTransform<double>(target_W));
  };
  if (show_spider_best_ring_target) {
    environment_.meshcat()->SetObject(
        "/spider/virtual_ring_search_best_target",
        drake::geometry::Sphere(0.004),
        drake::geometry::Rgba(1.0, 0.0, 1.0, 1.0));
    update_spider_best_ring_target();
    std::cout << "[visual] spider: magenta sphere marks the selected ring "
                 "target_C=["
              << grasp_.spider_ring_hold.transpose()
              << "] m (sphere is visual only)\n";
  }

  for (double time = kControlDt; time < config_.sim_time;
       time += kControlDt) {
    environment_.PublishState();
    if (show_spider_best_ring_target) update_spider_best_ring_target();
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
  const Eigen::Vector3d cube_center =
      CubePoseFromPositions(environment_.cube_positions()).translation();
  PrintSapContactForceDiagnostic(
      {time,
       last_osc_contact_force_command_.time,
       last_osc_contact_force_command_.valid,
       last_osc_contact_force_command_.used_full_contact_force,
       config_.lambda_torque_scale,
       SummarizeSapFingertipCubeContacts(
           environment_.contacts(), grasp_.cube_body, grasp_.tip_bodies,
           cube_center),
       last_osc_contact_force_command_.c3_normal_force_target,
       last_osc_contact_force_command_.c3_normal_force_applied,
       last_osc_contact_force_command_.c3_force_on_cube_world,
       last_osc_contact_force_command_.osc_force_command_on_cube_world,
       last_osc_contact_force_command_.c3_contact_point_world,
       cube_center});
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
  // Spider starts with ring as an ordinary required initial contact, just as
  // the middle-release experiment does.  Do not enter C3 while its initial
  // four-finger IK target is still in transit.
  const bool ring_ready = grasp_.grasp_finger_count < 4 || reach_.arrived[3];
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
  planner_.Rebuild(grasp_.grasp_finger_count == 4
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
  const auto cube_pose = CubePoseFromPositions(environment_.cube_positions());
  const SapFingertipCubeContactSummary sap =
      SummarizeSapFingertipCubeContacts(
          environment_.contacts(), grasp_.cube_body, grasp_.tip_bodies,
          cube_pose.translation());
  const double cube_weight = environment_.plant()
                                 .get_body(grasp_.cube_body)
                                 .get_mass(environment_.plant_context()) *
      9.81;
  maneuver_.Update(
      time, reference_time, touching, environment_.state(),
      cube_pose, sap.net_force_on_cube_world.z(), cube_weight, &planner_,
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

std::array<Vector3d, 4> SqueezeApplication::UpdateYawWrenchFeedback(
    double time, const C3ContactForcePlan& c3_contact_force_plan) {
  const std::array<Vector3d, 4> zero{
      Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(),
      Vector3d::Zero()};
  if (!config_.osc_wrench_feedback || cube_pinned_ ||
      !c3_contact_force_plan.valid) {
    yaw_wrench_feedback_.force_on_cube_world = zero;
    return zero;
  }

  const auto cube_pose = CubePoseFromPositions(environment_.cube_positions());
  const Vector3d cube_center = cube_pose.translation();
  const SapFingertipCubeContactSummary sap =
      SummarizeSapFingertipCubeContacts(
          environment_.contacts(), grasp_.cube_body, grasp_.tip_bodies,
          cube_center);
  const auto yaw_moment = [&cube_center](
                              const std::array<Vector3d, 4>& forces,
                              const std::array<Vector3d, 4>& points) {
    double result = 0.0;
    for (int finger = 0; finger < 4; ++finger)
      result += (points.at(finger) - cube_center)
                    .cross(forces.at(finger))
                    .z();
    return result;
  };
  yaw_wrench_feedback_.c3_yaw_moment = yaw_moment(
      c3_contact_force_plan.force_on_cube_world,
      c3_contact_force_plan.contact_point_world);
  yaw_wrench_feedback_.sap_yaw_moment =
      sap.moment_about_cube_center_world.z();

  const double reference_time =
      std::max(0.0, time - (handoff_time_ + 0.5));
  const auto desired_pose = CubeTarget(reference_time).pose;
  const drake::math::RollPitchYaw<double> desired_delta(
      desired_pose.rotation() * grasp_.initial_cube_pose.rotation().inverse());
  const drake::math::RollPitchYaw<double> measured_delta(
      cube_pose.rotation() * grasp_.initial_cube_pose.rotation().inverse());
  yaw_wrench_feedback_.yaw_error = std::remainder(
      desired_delta.yaw_angle() - measured_delta.yaw_angle(), 2.0 * M_PI);

  // This experiment is a yaw-maneuver corrector, not a general zero-yaw
  // attitude servo.  Do not consume the authority-test budget while the
  // gait is still in its pre-yaw settling hold; start only once its reference
  // has actually departed from zero.
  if (std::abs(desired_delta.yaw_angle()) < 1e-3) {
    yaw_wrench_feedback_.yaw_integral = 0.0;
    yaw_wrench_feedback_.requested_yaw_moment = 0.0;
    yaw_wrench_feedback_.allocated_yaw_moment = 0.0;
    yaw_wrench_feedback_.authority_test_start_time = -1.0;
    yaw_wrench_feedback_.force_on_cube_world = zero;
    return zero;
  }

  if (yaw_wrench_feedback_.last_update_time >= 0.0 &&
      time < yaw_wrench_feedback_.last_update_time +
                 config_.osc_wrench_feedback_period) {
    return yaw_wrench_feedback_.authority_lost
               ? zero
               : yaw_wrench_feedback_.force_on_cube_world;
  }
  const double dt = yaw_wrench_feedback_.last_update_time < 0.0
                        ? config_.osc_wrench_feedback_period
                        : time - yaw_wrench_feedback_.last_update_time;
  yaw_wrench_feedback_.last_update_time = time;
  if (yaw_wrench_feedback_.authority_lost) {
    yaw_wrench_feedback_.force_on_cube_world = zero;
    return zero;
  }

  // Integrate only while the requested correction has headroom. This is
  // ordinary PI anti-windup; the separate authority gate below prevents a
  // feasible-but-ineffective force map from accumulating indefinitely.
  const double previous_integral = yaw_wrench_feedback_.yaw_integral;
  const double candidate_integral = previous_integral +
      yaw_wrench_feedback_.yaw_error * std::max(0.0, dt);
  const double raw_candidate = config_.osc_wrench_feedback_yaw_kp *
          yaw_wrench_feedback_.yaw_error +
      config_.osc_wrench_feedback_yaw_ki * candidate_integral;
  const double requested = std::clamp(
      raw_candidate, -config_.osc_wrench_feedback_max_yaw_moment,
      config_.osc_wrench_feedback_max_yaw_moment);
  const bool saturated = std::abs(raw_candidate - requested) > 1e-12;
  if (!saturated ||
      requested * yaw_wrench_feedback_.yaw_error < 0.0) {
    yaw_wrench_feedback_.yaw_integral = candidate_integral;
  }
  yaw_wrench_feedback_.requested_yaw_moment = requested;

  std::array<Vector3d, 4> directions = zero;
  std::array<double, 4> leverage{};
  double denominator = config_.osc_wrench_feedback_allocation_damping;
  for (int finger : planner_.active_fingers()) {
    const SapFingerContactForce& contact = sap.fingers.at(finger);
    if (contact.point_contact_count == 0) continue;
    const Vector3d r = contact.contact_point_world - cube_center;
    Vector3d tangent = Vector3d::UnitZ().cross(r);
    const Vector3d normal = contact.normal_into_cube_world;
    if (normal.squaredNorm() > 1e-12)
      tangent -= normal * normal.dot(tangent);
    const double tangent_norm = tangent.norm();
    if (tangent_norm < 1e-9) continue;
    tangent /= tangent_norm;
    const double moment_per_newton = r.cross(tangent).z();
    if (std::abs(moment_per_newton) < 1e-9) continue;
    directions.at(finger) = tangent;
    leverage.at(finger) = moment_per_newton;
    denominator += moment_per_newton * moment_per_newton;
  }

  std::array<Vector3d, 4> correction = zero;
  if (denominator > config_.osc_wrench_feedback_allocation_damping + 1e-12) {
    for (int finger : planner_.active_fingers()) {
      if (std::abs(leverage.at(finger)) < 1e-9) continue;
      const double scalar = std::clamp(
          requested * leverage.at(finger) / denominator,
          -config_.osc_wrench_feedback_max_force_per_contact,
          config_.osc_wrench_feedback_max_force_per_contact);
      const Vector3d target_force = scalar * directions.at(finger);
      const Vector3d delta =
          target_force - yaw_wrench_feedback_.force_on_cube_world.at(finger);
      const double max_delta =
          config_.osc_wrench_feedback_force_rate_limit * std::max(0.0, dt);
      correction.at(finger) = delta.norm() <= max_delta || max_delta <= 0.0
                                  ? target_force
                                  : yaw_wrench_feedback_.force_on_cube_world.at(finger) +
                                        (max_delta / delta.norm()) * delta;
    }
  }
  yaw_wrench_feedback_.force_on_cube_world = correction;
  yaw_wrench_feedback_.allocated_yaw_moment = 0.0;
  for (int finger = 0; finger < 4; ++finger) {
    const SapFingerContactForce& contact = sap.fingers.at(finger);
    if (contact.point_contact_count == 0) continue;
    yaw_wrench_feedback_.allocated_yaw_moment +=
        (contact.contact_point_world - cube_center)
            .cross(correction.at(finger))
            .z();
  }

  const bool authority_test_active =
      std::abs(yaw_wrench_feedback_.allocated_yaw_moment) >=
      config_.osc_wrench_feedback_min_commanded_yaw_moment;
  const bool insufficient_response =
      std::abs(yaw_wrench_feedback_.sap_yaw_moment) <
      config_.osc_wrench_feedback_min_resolved_yaw_moment;
  if (authority_test_active && insufficient_response) {
    if (yaw_wrench_feedback_.authority_test_start_time < 0.0)
      yaw_wrench_feedback_.authority_test_start_time = time;
    if (time >= yaw_wrench_feedback_.authority_test_start_time +
                    config_.osc_wrench_feedback_authority_timeout) {
      yaw_wrench_feedback_.authority_lost = true;
      yaw_wrench_feedback_.force_on_cube_world = zero;
      correction = zero;
      std::cout << "[t=" << time
                << "] wrench feedback: insufficient measured yaw authority; "
                   "freezing integral and returning to nominal C3 hold\n";
    }
  } else {
    yaw_wrench_feedback_.authority_test_start_time = -1.0;
  }

  if (config_.osc_wrench_feedback_log &&
      (yaw_wrench_feedback_.last_log_time < 0.0 ||
       time >= yaw_wrench_feedback_.last_log_time + 0.1)) {
    yaw_wrench_feedback_.last_log_time = time;
    std::cout << std::fixed << std::setprecision(6)
              << "[WRENCH-FB t=" << time << "] yaw_err="
              << yaw_wrench_feedback_.yaw_error << " Mz_C3="
              << yaw_wrench_feedback_.c3_yaw_moment << " Mz_SAP="
              << yaw_wrench_feedback_.sap_yaw_moment << " Mz_req="
              << yaw_wrench_feedback_.requested_yaw_moment << " Mz_alloc="
              << yaw_wrench_feedback_.allocated_yaw_moment << " integral="
              << yaw_wrench_feedback_.yaw_integral << " authority="
              << (yaw_wrench_feedback_.authority_lost ? "lost" : "active")
              << "\n";
  }
  return correction;
}

void SqueezeApplication::UpdateTrackingReference(double time,
                                                  double reference_time) {
  const auto measured = CubePoseFromPositions(environment_.cube_positions());
  const auto now = CubeTarget(reference_time).pose;
  const auto end = CubeTarget(reference_time + config_.N * config_.c3_dt).pose;
  // The historical/reference mode follows the commanded cube motion while
  // limiting how far the hand IK can lead the real cube. Measured mode is a
  // separate feedback option: its contact points are attached to the actual
  // simulator cube pose, so a small cube displacement moves the PD target too.
  const bool use_measured_pose =
      config_.contact_ik_pose_source == "measured";
  const auto ik_now = use_measured_pose
                          ? measured
                          : ClampIkLead(now, measured,
                                        config_.cube_ik_lead_pos_max,
                                        config_.cube_ik_lead_rot_max);
  const auto ik_end = use_measured_pose
                          ? measured
                          : ClampIkLead(end, measured,
                                        config_.cube_ik_lead_pos_max,
                                        config_.cube_ik_lead_rot_max);
  tracking_.ramp_origin = tracking_.contact_start;
  tracking_.ramp_start_time = time;
  bool start_ok = false;
  bool end_ok = false;
  const VectorXd start = grasp_.SolveContactIk(
      ik_now, tracking_.contact_start, maneuver_.ring_engaged(),
      &start_ok);
  const VectorXd finish = grasp_.SolveContactIk(
      ik_end, start_ok ? start : tracking_.contact_start,
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

  const C3ContactForcePlan c3_contact_force_plan =
      planner_.FirstPhysicalContactForcePlan();
  const std::array<Vector3d, 4> wrench_feedback_force =
      UpdateYawWrenchFeedback(time, c3_contact_force_plan);
  OscExecutorResult result = ComputeOscExecutorTorque(
      OscExecutorRequest{
          environment_.plant(), environment_.plant_context(),
          environment_.hand_model(), environment_.cube_model(),
          grasp_.tip_bodies, GraspSetup::kFingerStarts,
          planner_.active_fingers(), planner_.normal_groups(),
          planner_.force_crossfade_origins(), planner_.finger_joined_times(),
          maneuver_.OscPressDirectionsInCube(),
          desired, environment_.hand_positions(),
          environment_.hand_velocities(), planner_.FirstPhysicalForce(),
          c3_contact_force_plan.force_on_cube_world,
          wrench_feedback_force,
          c3_contact_force_plan.valid,
          config_.osc_full_contact_force,
          time,
          kControlDt, config_.osc_kp, config_.osc_kd,
          config_.osc_qd_filter_tau, config_.gait_force_ramp_time,
          config_.lambda_torque_scale},
      &desired_velocity_filter_);
  const int handoff_generation = maneuver_.handoff_generation();
  if (handoff_generation != maneuver_handoff_generation_last_control_ &&
      last_osc_pd_torque_.size() == result.pd_torque.size()) {
    // Every touchdown rebuilds C3 and can change the whole-hand IK posture.
    // Preserve the applied pre-handoff PD torque and blend toward the new
    // value so a valid ring or spider-index touchdown does not kick the cube
    // into another static-friction equilibrium.
    osc_pd_crossfade_from_ = last_osc_pd_torque_;
    osc_pd_crossfade_start_time_ = time;
    osc_pd_crossfade_active_ = true;
  }
  maneuver_handoff_generation_last_control_ = handoff_generation;
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
  last_osc_contact_force_command_.valid = true;
  last_osc_contact_force_command_.time = time;
  last_osc_contact_force_command_.used_full_contact_force =
      result.used_full_contact_force;
  last_osc_contact_force_command_.c3_normal_force_target =
      result.normal_force_target;
  last_osc_contact_force_command_.c3_normal_force_applied =
      result.normal_force_applied;
  last_osc_contact_force_command_.c3_force_on_cube_world =
      result.c3_force_on_cube_world;
  last_osc_contact_force_command_.osc_force_command_on_cube_world =
      result.force_command_on_cube_world;
  last_osc_contact_force_command_.c3_contact_point_world =
      c3_contact_force_plan.contact_point_world;
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
