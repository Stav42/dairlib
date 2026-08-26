#include "allegro_grasp_c3_squeeze_planner.h"

#include <algorithm>
#include <iostream>

#include <drake/solvers/osqp_solver.h>
#include <drake/solvers/solver_options.h>

#include "allegro_grasp_c3_squeeze_contact.h"
#include "c3/core/c3_plus.h"
#include "c3/multibody/lcs_factory.h"

namespace dairlib::allegro_grasp_c3 {
namespace {

using drake::solvers::OsqpSolver;
using drake::solvers::SolverOptions;
using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr std::array<int, 4> kFingerStarts{0, 4, 12, 8};

}  // namespace

C3Planner::C3Planner(const SqueezeConfig& config, LcsModel* model)
    : config_(config), model_(model) {
  DRAKE_DEMAND(model_ != nullptr);
  lcs_options_.contact_model = config_.contact_model;
  lcs_options_.N = config_.N;
  lcs_options_.dt = config_.c3_dt;
  lcs_options_.num_contacts = 3;
  lcs_options_.num_friction_directions = config_.num_friction_directions;
  lcs_options_.mu = config_.mu;

  dimensions_.state = model_->state_size();
  dimensions_.input = model_->input_size();
  dimensions_.positions = model_->position_size();
  dimensions_.hand_positions = model_->hand_position_size();
  dimensions_.hand_velocities = model_->hand_velocity_size();
  dimensions_.contacts = 3;
  dimensions_.lambda = c3::multibody::LCSFactory::GetNumContactVariables(
      c3::multibody::GetContactModelMap().at(config_.contact_model), 3,
      config_.num_friction_directions);
  dimensions_.auxiliary = dimensions_.state + dimensions_.input +
                          2 * dimensions_.lambda;
  normal_groups_ = MakeNormalForceGroups(
      config_.contact_model, dimensions_.contacts,
      config_.num_friction_directions);

  state_cost_ = MatrixXd::Zero(dimensions_.state, dimensions_.state);
  for (int i = dimensions_.hand_positions; i < dimensions_.positions; ++i)
    state_cost_(i, i) = config_.w_cube;
  for (int i = dimensions_.positions;
       i < dimensions_.positions + dimensions_.hand_velocities; ++i)
    state_cost_(i, i) = config_.w_vel;
  for (int i = dimensions_.positions + dimensions_.hand_velocities;
       i < dimensions_.state; ++i)
    state_cost_(i, i) = config_.w_cube_vel;

  c3_options_.admm_iter = config_.admm_iter;
  c3_options_.rho_scale = config_.rho_scale;
  c3_options_.warm_start = config_.warm_start;
  c3_options_.scale_lcs = true;
  c3_options_.gamma = 1.0;
}

C3Planner::~C3Planner() = default;

void C3Planner::SetBaseTarget(const VectorXd& target) {
  DRAKE_DEMAND(target.size() == dimensions_.state);
  base_target_ = target;
}

void C3Planner::SetFingerTarget(int finger, const Eigen::Vector4d& joints) {
  DRAKE_DEMAND(finger >= 0 && finger < 4);
  DRAKE_DEMAND(base_target_.size() == dimensions_.state);
  base_target_.segment<4>(kFingerStarts[finger]) = joints;
}

c3::LCS C3Planner::MakeScaledLinearization(const VectorXd& state) {
  c3::LCS result = model_->Linearize(
      state, VectorXd::Zero(dimensions_.input), active_fingers_, lcs_options_);
  if (!c3_) {
    input_scale_ = config_.input_scale > 0.0
                       ? config_.input_scale
                       : result.A()[0].norm() / result.B()[0].norm();
  }
  std::vector<MatrixXd> scaled_input = result.B();
  for (MatrixXd& matrix : scaled_input) matrix *= input_scale_;
  result.set_B(scaled_input);
  return result;
}

void C3Planner::Rebuild(const std::vector<int>& active_fingers,
                        const VectorXd& state, double time) {
  DRAKE_DEMAND(!active_fingers.empty());
  DRAKE_DEMAND(base_target_.size() == dimensions_.state);

  std::array<double, 4> applied_before{0.0, 0.0, 0.0, 0.0};
  if (c3_ && !c3_->GetForceSolution().empty()) {
    const VectorXd old_force = FirstPhysicalForce();
    for (size_t i = 0; i < active_fingers_.size(); ++i) {
      const int finger = active_fingers_[i];
      const double target = SumNormalForce(old_force, normal_groups_[i]);
      const double blend = config_.gait_force_ramp_time <= 0.0
                               ? 1.0
                               : std::clamp(
                                     (time - finger_joined_times_[finger]) /
                                         config_.gait_force_ramp_time,
                                     0.0, 1.0);
      applied_before[finger] =
          (1.0 - blend) * force_crossfade_origins_[finger] + blend * target;
    }
  }
  force_crossfade_origins_ = applied_before;
  for (const int finger : active_fingers) finger_joined_times_[finger] = time;
  active_fingers_ = active_fingers;

  for (int i = 0; i < dimensions_.hand_positions; ++i)
    state_cost_(i, i) = 0.0;
  for (const int finger : active_fingers_) {
    for (int joint = 0; joint < 4; ++joint)
      state_cost_(kFingerStarts[finger] + joint,
                  kFingerStarts[finger] + joint) = config_.k_hold;
  }

  dimensions_.contacts = static_cast<int>(active_fingers_.size());
  lcs_options_.num_contacts = dimensions_.contacts;
  dimensions_.lambda = c3::multibody::LCSFactory::GetNumContactVariables(
      c3::multibody::GetContactModelMap().at(config_.contact_model),
      dimensions_.contacts, config_.num_friction_directions);
  dimensions_.auxiliary = dimensions_.state + dimensions_.input +
                          2 * dimensions_.lambda;
  normal_groups_ = MakeNormalForceGroups(
      config_.contact_model, dimensions_.contacts,
      config_.num_friction_directions);

  c3::LCS linearization = MakeScaledLinearization(state);
  const std::vector<VectorXd> desired(config_.N + 1, base_target_);
  const std::vector<MatrixXd> Q(config_.N + 1, state_cost_);
  const std::vector<MatrixXd> R(
      config_.N, input_scale_ * input_scale_ * config_.w_R *
                     MatrixXd::Identity(dimensions_.input, dimensions_.input));
  const std::vector<MatrixXd> G(
      config_.N, config_.w_G * MatrixXd::Identity(
                                     dimensions_.auxiliary,
                                     dimensions_.auxiliary));
  const std::vector<MatrixXd> U(
      config_.N, config_.w_U * MatrixXd::Identity(
                                     dimensions_.auxiliary,
                                     dimensions_.auxiliary));
  c3_ = std::make_unique<c3::C3Plus>(
      linearization, c3::C3::CostMatrices(Q, R, G, U), desired, c3_options_);
  c3_->SetAdmmWarmStartAcrossSolves(config_.warm_start_admm);
  ConfigureForceTracking();

  const double scaled_bound = config_.tau_max / input_scale_;
  for (int i = 0; i < dimensions_.input; ++i) {
    Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(dimensions_.input);
    row(i) = 1.0;
    c3_->AddLinearConstraint(row, -scaled_bound, scaled_bound,
                             c3::ConstraintVariable::INPUT);
  }
  SolverOptions solver_options;
  const auto solver = OsqpSolver::id();
  solver_options.SetOption(solver, "max_iter", 4000);
  solver_options.SetOption(solver, "verbose", 0);
  solver_options.SetOption(solver, "warm_starting", 1);
  solver_options.SetOption(solver, "polishing", 1);
  solver_options.SetOption(solver, "polish_refine_iter", 3);
  solver_options.SetOption(solver, "scaled_termination", 1);
  solver_options.SetOption(solver, "check_termination", 25);
  solver_options.SetOption(solver, "scaling", 15);
  solver_options.SetOption(solver, "adaptive_rho", 1);
  solver_options.SetOption(solver, "rho", 1e-4);
  solver_options.SetOption(solver, "sigma", 1e-6);
  solver_options.SetOption(solver, "alpha", 1.6);
  solver_options.SetOption(solver, "eps_abs", config_.osqp_eps);
  solver_options.SetOption(solver, "eps_rel", config_.osqp_eps);
  c3_->SetSolverOptions(solver_options);

  c3_->SetAdmmWarmStartAcrossSolves(true);
  for (int pass = 0; pass < std::max(1, config_.gait_rebuild_solve_passes);
       ++pass)
    c3_->Solve(state);
  c3_->SetAdmmWarmStartAcrossSolves(config_.warm_start_admm);

  std::cout << "  C3 rebuild: contacts=" << dimensions_.contacts
            << "  n_lambda=" << dimensions_.lambda
            << "  n_z=" << dimensions_.auxiliary
            << "  s_u=" << input_scale_ << "\n";
}

void C3Planner::ConfigureForceTracking() {
  if (!c3_ || config_.w_lambda <= 0.0) return;
  const std::array<double, 4> target_force{
      config_.alpha_m, config_.alpha_m, 2.0 * config_.alpha_m, 0.0};
  const double lambda_scale = c3_->GetLambdaScaling();
  std::vector<MatrixXd> weights(
      config_.N, MatrixXd::Zero(dimensions_.lambda, dimensions_.lambda));
  std::vector<VectorXd> targets(
      config_.N, VectorXd::Zero(dimensions_.lambda));
  for (int knot = 0; knot < config_.N; ++knot) {
    for (size_t i = 0; i < active_fingers_.size(); ++i) {
      const auto& group = normal_groups_[i];
      for (const int row : group) {
        for (const int column : group)
          weights[knot](row, column) =
              config_.w_lambda * lambda_scale * lambda_scale;
        targets[knot](row) = target_force[active_fingers_[i]] /
                             (group.size() * lambda_scale);
      }
    }
  }
  c3_->SetForceTrackingWeight(weights);
  c3_->UpdateForceTarget(targets);
}

void C3Planner::Relinearize(const VectorXd& state) {
  DRAKE_DEMAND(c3_ != nullptr);
  c3_->UpdateLCS(MakeScaledLinearization(state));
  ConfigureForceTracking();
}

void C3Planner::UpdateTarget(const std::vector<VectorXd>& target) {
  DRAKE_DEMAND(c3_ != nullptr);
  c3_->UpdateTarget(target);
}
void C3Planner::Solve(const VectorXd& state) {
  DRAKE_DEMAND(c3_ != nullptr);
  c3_->Solve(state);
}

bool C3Planner::initialized() const { return c3_ != nullptr; }
double C3Planner::input_scale() const { return input_scale_; }
VectorXd C3Planner::FirstPhysicalInput() const {
  return input_scale_ * c3_->GetInputSolution().at(0);
}
VectorXd C3Planner::FirstPhysicalForce() const {
  return c3_->GetForceSolution().at(0) / c3_->GetLambdaScaling();
}
std::array<double, 4> C3Planner::DesiredNormalForces(double time) const {
  std::array<double, 4> result{0.0, 0.0, 0.0, 0.0};
  if (!c3_ || c3_->GetForceSolution().empty()) return result;
  const VectorXd force = FirstPhysicalForce();
  for (size_t i = 0; i < active_fingers_.size(); ++i) {
    const int finger = active_fingers_[i];
    const double target = SumNormalForce(force, normal_groups_[i]);
    const double blend = config_.gait_force_ramp_time <= 0.0
                             ? 1.0
                             : std::clamp(
                                   (time - finger_joined_times_[finger]) /
                                       config_.gait_force_ramp_time,
                                   0.0, 1.0);
    result[finger] =
        (1.0 - blend) * force_crossfade_origins_[finger] + blend * target;
  }
  return result;
}

const PlannerDimensions& C3Planner::dimensions() const { return dimensions_; }
const std::vector<int>& C3Planner::active_fingers() const {
  return active_fingers_;
}
const std::vector<std::vector<int>>& C3Planner::normal_groups() const {
  return normal_groups_;
}
const std::array<double, 4>& C3Planner::finger_joined_times() const {
  return finger_joined_times_;
}
const std::array<double, 4>& C3Planner::force_crossfade_origins() const {
  return force_crossfade_origins_;
}
const VectorXd& C3Planner::base_target() const { return base_target_; }
std::vector<VectorXd> C3Planner::state_solution() const {
  return c3_->GetStateSolution();
}
std::vector<VectorXd> C3Planner::input_solution() const {
  return c3_->GetInputSolution();
}
std::vector<VectorXd> C3Planner::force_solution() const {
  return c3_->GetForceSolution();
}
std::vector<VectorXd> C3Planner::dual_delta_solution() {
  return c3_->GetDualDeltaSolution();
}
const c3::C3& C3Planner::implementation() const { return *c3_; }

}  // namespace dairlib::allegro_grasp_c3
