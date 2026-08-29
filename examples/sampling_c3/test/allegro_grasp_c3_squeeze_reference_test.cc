#include <cmath>

#include <gtest/gtest.h>

#include <drake/math/rigid_transform.h>
#include <drake/math/roll_pitch_yaw.h>
#include <drake/math/rotation_matrix.h>

#include "examples/sampling_c3/allegro_grasp_c3_squeeze_reference.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::math::RigidTransform;
using drake::math::RollPitchYaw;
using drake::math::RotationMatrix;

constexpr double kPi = 3.14159265358979323846;
constexpr double kTolerance = 1e-13;

TEST(CubeReferenceTest, SpiderYawUsesWorldZAndKeepsCubeCenterFixed) {
  const RigidTransform<double> initial_pose(
      RotationMatrix<double>(RollPitchYaw<double>(0.2, -0.3, 0.4)),
      Eigen::Vector3d(0.5, -0.25, 0.8));
  CubeMotionReferenceConfig config;
  config.gait_enabled = true;
  config.gait_rotate_start_time = 1.0;
  config.gait_rotate_duration = 2.0;
  config.gait_theta_start = 0.0;
  config.gait_theta_target = kPi / 2.0;
  config.gait_rotation_frame = GaitRotationFrame::kWorldZ;

  // At the midpoint of a minimum-jerk ramp, the scale is exactly one half.
  const CubeReference result = MakeCubeReference(initial_pose, config, 2.0);
  const RotationMatrix<double> expected_world_yaw(
      RollPitchYaw<double>(0.0, 0.0, kPi / 4.0));
  const Eigen::Matrix3d expected_rotation =
      (expected_world_yaw * initial_pose.rotation()).matrix();

  EXPECT_TRUE(result.pose.rotation().matrix().isApprox(expected_rotation,
                                                       kTolerance));
  EXPECT_TRUE(result.pose.translation().isApprox(initial_pose.translation(),
                                                  kTolerance));
}

TEST(CubeReferenceTest, HistoricalGaitStillUsesCubeY) {
  const RigidTransform<double> initial_pose(
      RotationMatrix<double>(RollPitchYaw<double>(0.2, -0.3, 0.4)),
      Eigen::Vector3d(0.5, -0.25, 0.8));
  CubeMotionReferenceConfig config;
  config.gait_enabled = true;
  config.gait_rotate_duration = 1.0;
  config.gait_theta_target = 0.25;
  config.gait_rotation_frame = GaitRotationFrame::kCubeY;

  const CubeReference result = MakeCubeReference(initial_pose, config, 1.0);
  const RotationMatrix<double> expected_cube_y(
      RollPitchYaw<double>(0.0, 0.25, 0.0));
  const Eigen::Matrix3d expected_rotation =
      (initial_pose.rotation() * expected_cube_y).matrix();

  EXPECT_TRUE(result.pose.rotation().matrix().isApprox(expected_rotation,
                                                       kTolerance));
  EXPECT_TRUE(result.pose.translation().isApprox(initial_pose.translation(),
                                                  kTolerance));
}

}  // namespace
}  // namespace dairlib::allegro_grasp_c3
