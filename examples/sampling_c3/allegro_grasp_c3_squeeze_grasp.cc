#include "allegro_grasp_c3_squeeze_grasp.h"

#include <iostream>
#include <string>
#include <vector>

#include <drake/geometry/rgba.h>
#include <drake/geometry/shape_specification.h>
#include <drake/math/roll_pitch_yaw.h>

#include "allegro_hand_utils.h"
#include "allegro_grasp_c3_squeeze_ik.h"
#include "cube_kinematics.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::math::RigidTransform;
using drake::math::RotationMatrix;
using Eigen::Vector3d;
using Eigen::VectorXd;

VectorXd MakeCubePositions(const RigidTransform<double>& pose) {
  VectorXd result(7);
  result << pose.rotation().ToQuaternion().w(),
      pose.rotation().ToQuaternion().x(), pose.rotation().ToQuaternion().y(),
      pose.rotation().ToQuaternion().z(), pose.translation();
  return result;
}

}  // namespace

GraspSetup::GraspSetup(const SqueezeConfig& config,
                       SimulationEnvironment* environment)
    : config_(config), environment_(environment) {
  DRAKE_DEMAND(environment_ != nullptr);
  auto& plant = environment_->plant();
  auto& context = environment_->plant_context();
  const auto hand = environment_->hand_model();
  const auto cube = environment_->cube_model();

  const double half = kCubeSize / 2.0;
  const double index_x = config.release_middle
                             ? -config.release_middle_tri_spread
                             : -0.02;
  const double index_z = config.release_middle
                             ? config.release_middle_tri_base_z
                             : 0.0;
  const double middle_x = config.release_middle ? 0.0 : 0.02;
  const double middle_z = config.release_middle
                              ? config.release_middle_tri_apex_z
                              : 0.0;
  grasp_finger_count = config.release_middle ? 4 : 3;
  hand_positions = plant.num_positions(hand);
  initial_cube_pose = RigidTransform<double>(
      RotationMatrix<double>(),
      Vector3d(config.cube_start_x, config.cube_start_y,
               config.cube_start_z));
  initial_cube_positions = MakeCubePositions(initial_cube_pose);
  tip_surface_offset = Vector3d(0, 0, config.tip_surface_offset_z);
  ring_surface_offset =
      Vector3d(0, config.ring_tip_surface_offset_y,
               config.ring_tip_surface_offset_z);

  const std::array<std::string, 4> names{
      "link_3_tip", "link_7_tip", "link_15_tip", "link_11_tip"};
  for (int i = 0; i < 4; ++i)
    tip_bodies[i] = plant.GetBodyByName(names[i], hand).index();
  cube_body = plant.GetBodyIndices(cube)[0];

  footprints = {
      Vector3d(index_x,
               -(half - config.penetration_index_middle), index_z),
      Vector3d(middle_x,
               -(half - config.penetration_index_middle), middle_z),
      Vector3d(0.0, half - config.penetration_thumb, 0.0),
      Vector3d(config.release_middle_tri_spread,
               -(half - config.penetration_index_middle),
               config.release_middle_tri_base_z)};
  initial_footprints = footprints;
  relay_ring_hold = Vector3d(
      0.0, -(half - config.relay_ring_press), config.relay_ring_hold_z);
  relay_ring_park = Vector3d(
      0.0, -(half + config.relay_ring_retract), config.relay_ring_hold_z);
  // For the lateral spider-walk, ring is the temporary bridge onto the
  // incoming face 1 / red. Red is body +X, while its two in-face coordinates
  // are body Y and Z. The park point is outside that same face along +X.
  spider_ring_hold = Vector3d(
      half - config.relay_ring_press, config.spider_ring_red_y,
      config.spider_ring_hold_z);
  spider_ring_park = Vector3d(
      half + config.relay_ring_retract, config.spider_ring_red_y,
      config.spider_ring_hold_z);

  VectorXd targets(9);
  for (int i = 0; i < 3; ++i)
    targets.segment<3>(3 * i) = initial_cube_pose * footprints[i];
  const Vector3d ring_target = initial_cube_pose * footprints[3];
  GraspIkSolver solver(plant, &context, hand, cube, initial_cube_positions,
                       tip_bodies, tip_surface_offset, ring_surface_offset);
  contact_positions = config.release_middle
                          ? solver.SolveWithRing("contact", targets, ring_target)
                          : solver.SolveThreeFinger("contact", targets);

  if (config.release_middle && config.release_finger == "middle") {
    VectorXd release_targets = targets;
    release_targets.segment<3>(3) = initial_cube_pose * Vector3d(
        middle_x, -(half + config.release_middle_offset), middle_z);
    release_middle_positions = solver.SolveWithRing(
        "release_middle", release_targets, ring_target);
  } else if (config.release_middle && config.release_finger == "ring") {
    const Vector3d ring_end_C(config.release_middle_tri_spread,
                              -(half - config.penetration_index_middle),
                              middle_z);
    const Vector3d ring_mid_C(
        config.release_middle_tri_spread,
        -(half + config.regrasp_arc_clearance),
        0.5 * (config.release_middle_tri_base_z + middle_z));
    ring_regrasp_mid_positions = solver.SolveWithRing(
        "regrasp_ring_mid", targets, initial_cube_pose * ring_mid_C);
    ring_regrasp_positions = solver.SolveWithRing(
        "regrasp_ring", targets, initial_cube_pose * ring_end_C);

    const Vector3d middle_end_C(
        middle_x, -(half - config.penetration_index_middle),
        config.release_middle_tri_base_z);
    const Vector3d middle_mid_C(
        middle_x, -(half + config.regrasp_arc_clearance),
        0.5 * (middle_z + config.release_middle_tri_base_z));
    VectorXd middle_targets = targets;
    middle_targets.segment<3>(3) = initial_cube_pose * middle_mid_C;
    middle_regrasp_mid_positions = solver.SolveWithRing(
        "regrasp_middle_mid", middle_targets, initial_cube_pose * ring_end_C);
    middle_targets.segment<3>(3) = initial_cube_pose * middle_end_C;
    middle_regrasp_positions = solver.SolveWithRing(
        "regrasp_middle", middle_targets, initial_cube_pose * ring_end_C);

    const Vector3d index_end_C(
        index_x, -(half - config.penetration_index_middle), middle_z);
    const Vector3d index_mid_C(
        index_x, -(half + config.regrasp_arc_clearance),
        0.5 * (index_z + middle_z));
    VectorXd index_targets = targets;
    index_targets.segment<3>(0) = initial_cube_pose * index_mid_C;
    index_targets.segment<3>(3) = initial_cube_pose * middle_end_C;
    index_regrasp_mid_positions = solver.SolveWithRing(
        "regrasp_index_mid", index_targets, initial_cube_pose * ring_end_C);
    index_targets.segment<3>(0) = initial_cube_pose * index_end_C;
    index_regrasp_positions = solver.SolveWithRing(
        "regrasp_index", index_targets, initial_cube_pose * ring_end_C);
    relay_regrasp_targets = {ring_end_C, middle_end_C, index_end_C};
  }

  plant.SetPositions(&context, hand, contact_positions);
  VectorXd pregrasp_targets(9);
  pregrasp_targets <<
      initial_cube_pose * Vector3d(index_x, -(half + 0.01), index_z),
      initial_cube_pose * Vector3d(middle_x, -(half + 0.01), middle_z),
      initial_cube_pose * Vector3d(0.0, half + 0.01, 0.0);
  const Vector3d ring_pregrasp = initial_cube_pose * Vector3d(
      config.release_middle_tri_spread, -(half + 0.01),
      config.release_middle_tri_base_z);
  pregrasp_positions = config.release_middle
                           ? solver.SolveWithRing("pregrasp", pregrasp_targets,
                                                  ring_pregrasp)
                           : solver.SolveThreeFinger("pregrasp",
                                                     pregrasp_targets);

  std::vector<Eigen::MatrixXd> knots{pregrasp_positions, contact_positions};
  reach_trajectory =
      drake::trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
          {0.0, config.t_contact}, knots, true);
  reach_velocity = reach_trajectory.derivative(1);
  ik_context_ = plant.CreateDefaultContext();

  environment_->PublishContactTarget(contact_positions);
  if (config.show_cube_target) {
    environment_->meshcat()->SetObject(
        "/cube_target",
        drake::geometry::Box(kCubeSize + 0.002, kCubeSize + 0.002,
                             kCubeSize + 0.002),
        drake::geometry::Rgba(0.0, 0.9, 1.0, 0.35));
    environment_->meshcat()->SetTransform("/cube_target", initial_cube_pose);
  }
  if (config.show_cube_start) {
    environment_->meshcat()->SetObject(
        "/cube_start",
        drake::geometry::Box(kCubeSize + 0.004, kCubeSize + 0.004,
                             kCubeSize + 0.004),
        drake::geometry::Rgba(1.0, 0.0, 0.0, 0.35));
    environment_->meshcat()->SetTransform("/cube_start", initial_cube_pose);
  }
}

void GraspSetup::PreviewInitialPoses(bool wait_for_user) {
  auto publish = [&](const VectorXd& hand, const char* label) {
    environment_->SetInitialState(hand, initial_cube_positions);
    environment_->ForcedPublish();
    std::cout << "\n=== STATIC PREVIEW: " << label << " ===\n";
    if (wait_for_user) {
      std::cout << "Press Enter to continue...\n";
      std::cin.get();
    }
  };
  publish(pregrasp_positions, "q_pregrasp (1 cm outside cube faces)");
  publish(contact_positions, "q_contact (C3 target configuration)");
}

VectorXd GraspSetup::SolveContactIk(const RigidTransform<double>& pose,
                                    const VectorXd& seed, bool include_ring,
                                    bool* success) {
  auto& plant = environment_->plant();
  plant.SetPositions(ik_context_.get(), environment_->hand_model(), seed);
  VectorXd targets(9);
  for (int i = 0; i < 3; ++i)
    targets.segment<3>(3 * i) = pose * footprints[i];
  const VectorXd full = include_ring
                            ? SolveGraspIKWithRing(
                                  plant, ik_context_.get(), targets,
                                  pose * footprints[3], tip_surface_offset,
                                  ring_surface_offset, success)
                            : SolveGraspIK(plant, ik_context_.get(), targets,
                                           tip_surface_offset, success);
  plant.SetPositions(ik_context_.get(), full);
  return plant.GetPositions(*ik_context_, environment_->hand_model());
}

VectorXd GraspSetup::SolveLegIk(const RigidTransform<double>& pose,
                                int moving_finger,
                                const Vector3d& target_in_cube,
                                const VectorXd& seed, bool* success) {
  auto& plant = environment_->plant();
  plant.SetPositions(ik_context_.get(), environment_->hand_model(), seed);
  VectorXd targets(9);
  for (int i = 0; i < 3; ++i)
    targets.segment<3>(3 * i) =
        pose * (i == moving_finger ? target_in_cube : footprints[i]);
  const Vector3d ring =
      pose * (moving_finger == 3 ? target_in_cube : footprints[3]);
  const VectorXd full = SolveGraspIKWithRing(
      plant, ik_context_.get(), targets, ring, tip_surface_offset,
      ring_surface_offset, success);
  plant.SetPositions(ik_context_.get(), full);
  return plant.GetPositions(*ik_context_, environment_->hand_model());
}

Vector3d GraspSetup::FingertipPosition(int finger) const {
  const auto& plant = environment_->plant();
  const auto& context = environment_->plant_context();
  const Vector3d offset = finger == 3 ? ring_surface_offset
                                      : tip_surface_offset;
  return plant.EvalBodyPoseInWorld(context, plant.get_body(tip_bodies[finger])) *
         offset;
}

VectorXd GraspSetup::FingertipPositionsForHand(const VectorXd& joints) {
  auto& plant = environment_->plant();
  plant.SetPositions(ik_context_.get(), environment_->hand_model(), joints);
  VectorXd result(9);
  for (int finger = 0; finger < 3; ++finger) {
    result.segment<3>(3 * finger) = plant.EvalBodyPoseInWorld(
        *ik_context_, plant.get_body(tip_bodies[finger])).translation();
  }
  return result;
}

}  // namespace dairlib::allegro_grasp_c3
