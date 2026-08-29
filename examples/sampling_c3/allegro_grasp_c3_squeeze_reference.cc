#include "examples/sampling_c3/allegro_grasp_c3_squeeze_reference.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <drake/math/roll_pitch_yaw.h>
#include <drake/math/rotation_matrix.h>

namespace dairlib::allegro_grasp_c3 {
namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

std::pair<double, double> ComputeCubeMotionProfile(
    const CubeMotionReferenceConfig& config, double reference_time) {
  const double duration = std::max(1e-6, config.move_duration);
  const double s = std::clamp(reference_time / duration, 0.0, 1.0);
  const double ramp =
      10.0 * s * s * s - 15.0 * s * s * s * s + 6.0 * s * s * s * s * s;
  const double ramp_dot =
      (reference_time <= 0.0 || reference_time >= duration)
          ? 0.0
          : (30.0 * s * s - 60.0 * s * s * s + 30.0 * s * s * s * s) /
                duration;
  if (config.motion_mode == "step") {
    return {ramp, ramp_dot};
  }
  if (config.motion_mode == "sine") {
    const double frequency = kPi * 2.0 / std::max(1e-6, config.sine_period);
    const double sin_value = std::sin(frequency * reference_time);
    const double cos_value = std::cos(frequency * reference_time);
    return {ramp * sin_value,
            ramp_dot * sin_value + ramp * frequency * cos_value};
  }
  if (config.motion_mode == "none") {
    return {0.0, 0.0};
  }
  throw std::invalid_argument(
      "motion_mode must be 'none', 'step', or 'sine'.");
}

CubeReference MakeCubeReference(
    const drake::math::RigidTransform<double>& initial_cube_pose,
    const CubeMotionReferenceConfig& config, double reference_time) {
  using drake::math::RigidTransform;
  using drake::math::RollPitchYaw;
  using drake::math::RotationMatrix;

  if (config.gait_enabled) {
    const double duration = std::max(1e-6, config.gait_rotate_duration);
    const double s = std::clamp(
        (reference_time - config.gait_rotate_start_time) / duration, 0.0,
        1.0);
    const double ramp =
        10.0 * s * s * s - 15.0 * s * s * s * s + 6.0 * s * s * s * s * s;
    const double theta = config.gait_theta_start +
                         ramp * (config.gait_theta_target -
                                 config.gait_theta_start);
    if (config.gait_rotation_frame == GaitRotationFrame::kWorldZ) {
      const RotationMatrix<double> world_yaw(
          RollPitchYaw<double>(0.0, 0.0, theta));
      return {RigidTransform<double>(
                  world_yaw * initial_cube_pose.rotation(),
                  initial_cube_pose.translation()),
              Eigen::Vector3d::Zero()};
    }
    return {initial_cube_pose *
                RigidTransform<double>(
                    RotationMatrix<double>(RollPitchYaw<double>(0.0, theta, 0.0))),
            Eigen::Vector3d::Zero()};
  }

  const auto [alpha, alpha_dot] =
      ComputeCubeMotionProfile(config, reference_time);
  const RotationMatrix<double> rotation_delta(RollPitchYaw<double>(
      alpha * config.rpy_offset.x(), alpha * config.rpy_offset.y(),
      alpha * config.rpy_offset.z()));
  const RigidTransform<double> desired_pose =
      RigidTransform<double>(alpha * config.translation_offset) *
      initial_cube_pose * RigidTransform<double>(rotation_delta);
  return {desired_pose, alpha_dot * config.translation_offset};
}

drake::math::RigidTransform<double> ClampIkLead(
    const drake::math::RigidTransform<double>& desired_pose,
    const drake::math::RigidTransform<double>& measured_pose,
    double max_translation_lead, double max_rotation_lead) {
  if (max_translation_lead < 0.0 || max_rotation_lead < 0.0) {
    throw std::invalid_argument("IK lead limits must be nonnegative.");
  }

  const Eigen::Vector3d delta_translation =
      desired_pose.translation() - measured_pose.translation();
  const double translation_norm = delta_translation.norm();
  const Eigen::Vector3d clamped_translation =
      translation_norm > max_translation_lead && translation_norm > 0.0
          ? measured_pose.translation() +
                delta_translation * (max_translation_lead / translation_norm)
          : desired_pose.translation();

  const Eigen::Quaterniond measured_quaternion =
      measured_pose.rotation().ToQuaternion();
  const Eigen::Quaterniond desired_quaternion =
      desired_pose.rotation().ToQuaternion();
  const Eigen::AngleAxisd relative_rotation(
      desired_quaternion * measured_quaternion.inverse());
  drake::math::RotationMatrix<double> clamped_rotation =
      desired_pose.rotation();
  if (relative_rotation.angle() > max_rotation_lead) {
    const Eigen::AngleAxisd capped_rotation(max_rotation_lead,
                                            relative_rotation.axis());
    clamped_rotation = drake::math::RotationMatrix<double>(
        Eigen::Quaterniond(capped_rotation) * measured_quaternion);
  }
  return drake::math::RigidTransform<double>(clamped_rotation,
                                               clamped_translation);
}

}  // namespace dairlib::allegro_grasp_c3
