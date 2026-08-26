#include "allegro_grasp_c3_squeeze_ik.h"

#include <iostream>

#include "cube_kinematics.h"

namespace dairlib::allegro_grasp_c3 {

using Eigen::Vector3d;
using Eigen::VectorXd;

GraspIkSolver::GraspIkSolver(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context,
    drake::multibody::ModelInstanceIndex hand_model,
    drake::multibody::ModelInstanceIndex cube_model,
    VectorXd cube_positions,
    std::array<drake::multibody::BodyIndex, 4> tip_bodies,
    Vector3d fingertip_surface_offset, Vector3d ring_surface_offset)
    : plant_(plant),
      context_(context),
      hand_model_(hand_model),
      cube_model_(cube_model),
      cube_positions_(std::move(cube_positions)),
      tip_bodies_(tip_bodies),
      fingertip_surface_offset_(std::move(fingertip_surface_offset)),
      ring_surface_offset_(std::move(ring_surface_offset)) {}

void GraspIkSolver::ResetSeed() {
  VectorXd q = VectorXd::Zero(plant_.num_positions(hand_model_));
  q.segment<4>(12) << 1.0, 0.5, 0.5, 0.3;
  plant_.SetPositions(context_, hand_model_, q);
  plant_.SetPositions(context_, cube_model_, cube_positions_);
}

VectorXd GraspIkSolver::SolveThreeFinger(
    const std::string& label, const VectorXd& fingertip_targets) {
  ResetSeed();
  plant_.SetPositions(
      context_, SolveGraspIK(plant_, context_, fingertip_targets,
                             fingertip_surface_offset_));
  const VectorXd q = plant_.GetPositions(*context_, hand_model_);
  double total_error = 0.0;
  for (int finger = 0; finger < 3; ++finger) {
    const Vector3d actual =
        plant_.EvalBodyPoseInWorld(*context_,
                                   plant_.get_body(tip_bodies_[finger])) *
        fingertip_surface_offset_;
    total_error +=
        (actual - fingertip_targets.segment<3>(3 * finger)).norm();
  }
  std::cout << "IK " << label << " total FK error = " << total_error
            << " m\n";
  return q;
}

VectorXd GraspIkSolver::SolveWithRing(
    const std::string& label, const VectorXd& fingertip_targets,
    const Vector3d& ring_target) {
  ResetSeed();
  plant_.SetPositions(
      context_, SolveGraspIKWithRing(
                    plant_, context_, fingertip_targets, ring_target,
                    fingertip_surface_offset_, ring_surface_offset_));
  const VectorXd q = plant_.GetPositions(*context_, hand_model_);
  double total_error =
      (plant_.EvalBodyPoseInWorld(*context_,
                                  plant_.get_body(tip_bodies_[3])) *
           ring_surface_offset_ -
       ring_target)
          .norm();
  for (int finger = 0; finger < 3; ++finger) {
    total_error +=
        (plant_.EvalBodyPoseInWorld(*context_,
                                    plant_.get_body(tip_bodies_[finger])) *
             fingertip_surface_offset_ -
         fingertip_targets.segment<3>(3 * finger))
            .norm();
  }
  std::cout << "IK " << label << " total FK error = " << total_error
            << " m\n";
  return q;
}

}  // namespace dairlib::allegro_grasp_c3
