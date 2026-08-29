#pragma once

#include <string>
#include <utility>

#include <Eigen/Dense>

#include <drake/math/rigid_transform.h>

namespace dairlib::allegro_grasp_c3 {

enum class GaitRotationFrame {
  // Historical gait: post-multiply by a rotation about the cube's own +Y.
  kCubeY,
  // Lateral spiderwalk: pre-multiply the cube orientation by world +Z while
  // leaving the cube center fixed.
  kWorldZ,
};

// Parameters of the cube-reference generator. These are data rather than
// gflags so the profile can be evaluated deterministically in a unit test.
struct CubeMotionReferenceConfig {
  std::string motion_mode{"none"};
  Eigen::Vector3d translation_offset{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rpy_offset{Eigen::Vector3d::Zero()};
  double move_duration{0.5};
  double sine_period{1.0};

  // When enabled, gait rotation replaces the ordinary step/sine profile.
  bool gait_enabled{false};
  double gait_rotate_start_time{0.0};
  double gait_rotate_duration{4.0};
  double gait_theta_start{0.0};
  double gait_theta_target{0.0};
  GaitRotationFrame gait_rotation_frame{GaitRotationFrame::kCubeY};
};

struct CubeReference {
  drake::math::RigidTransform<double> pose;
  Eigen::Vector3d translational_velocity{Eigen::Vector3d::Zero()};
};

// Returns a quintic minimum-jerk position scale and its time derivative.
// "none" produces zero; "step" produces a ramp then hold; "sine" applies the
// same ramp as an amplitude envelope around the sine wave.
std::pair<double, double> ComputeCubeMotionProfile(
    const CubeMotionReferenceConfig& config, double reference_time);

// Evaluates the raw cube task reference at reference_time. The return value is
// a desired pose plus the translational velocity that belongs in the state
// reference. Desired angular velocity is intentionally not represented here.
CubeReference MakeCubeReference(
    const drake::math::RigidTransform<double>& initial_cube_pose,
    const CubeMotionReferenceConfig& config, double reference_time);

// Caps desired translation and relative rotation against the currently
// measured pose. This is used only to make a safe fingertip IK target; it does
// not alter the raw cube reference sent to C3.
drake::math::RigidTransform<double> ClampIkLead(
    const drake::math::RigidTransform<double>& desired_pose,
    const drake::math::RigidTransform<double>& measured_pose,
    double max_translation_lead, double max_rotation_lead);

}  // namespace dairlib::allegro_grasp_c3
