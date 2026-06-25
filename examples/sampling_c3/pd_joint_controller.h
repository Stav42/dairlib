#pragma once

#include <Eigen/Dense>
#include <drake/systems/framework/leaf_system.h>

namespace dairlib {

class PdJointController : public drake::systems::LeafSystem<double> {
 public:
  PdJointController(int num_joints, double kp, double kd)
      : num_joints_(num_joints), kp_(kp), kd_(kd) {
    state_port_ =
        DeclareVectorInputPort("state", 2 * num_joints).get_index();
    desired_port_ =
        DeclareVectorInputPort("q_desired", num_joints).get_index();
    qdot_desired_port_ =
        DeclareVectorInputPort("qdot_desired", num_joints).get_index();
    DeclareVectorOutputPort("torques", num_joints,
                            &PdJointController::CalcTorques);
  }

 private:
  void CalcTorques(const drake::systems::Context<double>& context,
                   drake::systems::BasicVector<double>* output) const {
    const Eigen::VectorXd state = get_input_port(state_port_).Eval(context);
    const Eigen::VectorXd q = state.head(num_joints_);
    const Eigen::VectorXd v = state.tail(num_joints_);
    const Eigen::VectorXd q_des = get_input_port(desired_port_).Eval(context);
    const Eigen::VectorXd qdot_des =
        get_input_port(qdot_desired_port_).Eval(context);
    output->SetFromVector(kp_ * (q_des - q) + kd_ * (qdot_des - v));
  }

  const int num_joints_;
  const double kp_;
  const double kd_;
  int state_port_;
  int desired_port_;
  int qdot_desired_port_;
};

}  // namespace dairlib
