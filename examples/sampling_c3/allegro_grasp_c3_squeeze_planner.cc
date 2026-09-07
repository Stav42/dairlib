#include "allegro_grasp_c3_squeeze_planner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

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

std::vector<double> ComputeCandidateVerticalContactForces(
    double cube_mass, double knot_dt, int first_contact_force,
    const std::vector<VectorXd>& cube_z_velocity_rows,
    const std::vector<VectorXd>& force_plan) {
  const size_t knot_count = std::min(cube_z_velocity_rows.size(),
                                     force_plan.size());
  std::vector<double> result;
  result.reserve(knot_count);
  for (size_t knot = 0; knot < knot_count; ++knot) {
    const VectorXd& velocity_row = cube_z_velocity_rows[knot];
    const VectorXd& lambda = force_plan[knot];
    if (knot_dt <= 0.0 || first_contact_force < 0 ||
        first_contact_force >= velocity_row.size() ||
        first_contact_force >= lambda.size() ||
        velocity_row.size() != lambda.size()) {
      result.push_back(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    const double delta_vz =
        velocity_row.tail(velocity_row.size() - first_contact_force)
            .dot(lambda.tail(lambda.size() - first_contact_force));
    result.push_back(cube_mass * delta_vz / knot_dt);
  }
  return result;
}

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
                        const VectorXd& state, double time,
                        double normal_force_margin, bool log_rebuild) {
  DRAKE_DEMAND(!active_fingers.empty());
  DRAKE_DEMAND(base_target_.size() == dimensions_.state);
  DRAKE_DEMAND(normal_force_margin >= 0.0);

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
  normal_force_margin_ = normal_force_margin;

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
  const bool have_live_horizon =
      static_cast<int>(horizon_target_.size()) == config_.N + 1 &&
      std::all_of(horizon_target_.begin(), horizon_target_.end(),
                  [this](const VectorXd& target) {
                    return target.size() == dimensions_.state;
                  });
  const std::vector<VectorXd> desired = have_live_horizon
      ? horizon_target_
      : std::vector<VectorXd>(config_.N + 1, base_target_);
  horizon_target_ = desired;
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
  ConfigureNormalForceMargin();

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

  if (log_rebuild) {
    std::cout << "  C3 rebuild: contacts=" << dimensions_.contacts
              << "  n_lambda=" << dimensions_.lambda
              << "  n_z=" << dimensions_.auxiliary
              << "  s_u=" << input_scale_ << "\n";
  }
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

void C3Planner::ConfigureNormalForceMargin() {
  if (!c3_ || normal_force_margin_ <= 0.0) return;

  // This is deliberately a shared feasibility constraint, not a reference
  // force or a requested force split. C3 is free to choose any distribution
  // that keeps every currently active contact compressive by this margin.
  for (const std::vector<int>& group : normal_groups_) {
    Eigen::RowVectorXd selector =
        Eigen::RowVectorXd::Zero(dimensions_.lambda);
    for (const int row : group) selector(row) = 1.0;
    // C3's force constraints are expressed in the same scaled-LCS lambda
    // coordinates returned by FirstPhysicalForce(), i.e. the units used by
    // the rest of this application for physical normal force.
    c3_->AddLinearConstraint(selector, normal_force_margin_,
                             std::numeric_limits<double>::infinity(),
                             c3::ConstraintVariable::FORCE);
  }
}

void C3Planner::Relinearize(const VectorXd& state) {
  DRAKE_DEMAND(c3_ != nullptr);
  c3_->UpdateLCS(MakeScaledLinearization(state));
  ConfigureForceTracking();
}

void C3Planner::UpdateTarget(const std::vector<VectorXd>& target) {
  DRAKE_DEMAND(c3_ != nullptr);
  DRAKE_DEMAND(static_cast<int>(target.size()) == config_.N + 1);
  horizon_target_ = target;
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
C3ContactForcePlan C3Planner::FirstPhysicalContactForcePlan() const {
  C3ContactForcePlan result;
  if (!c3_ || !c3_->GetLastSolveSucceeded() ||
      c3_->GetForceSolution().empty()) {
    return result;
  }

  const VectorXd force = FirstPhysicalForce();
  const std::vector<Eigen::Vector3d>& bases =
      model_->contact_force_bases();
  const std::vector<Eigen::Vector3d>& points =
      model_->contact_points_world();
  const int contact_count = dimensions_.contacts;
  if (force.size() != dimensions_.lambda ||
      bases.size() != static_cast<size_t>(dimensions_.lambda) ||
      points.size() != static_cast<size_t>(contact_count) ||
      static_cast<int>(active_fingers_.size()) != contact_count ||
      !force.allFinite()) {
    return result;
  }

  if (config_.contact_model == "stewart_and_trinkle") {
    // lambda = [gamma(contact_count), lambda_n(contact_count), beta(...)];
    // gamma is a friction-cone slack and is not a physical contact force.
    const int beta_per_contact = 2 * config_.num_friction_directions;
    const int expected_size =
        2 * contact_count + contact_count * beta_per_contact;
    if (force.size() != expected_size) return result;
    for (int contact = 0; contact < contact_count; ++contact) {
      const int finger = active_fingers_[contact];
      const int normal_index = contact_count + contact;
      result.force_on_cube_world[finger] +=
          force(normal_index) * bases[normal_index];
      const int beta_start = 2 * contact_count +
                             contact * beta_per_contact;
      for (int offset = 0; offset < beta_per_contact; ++offset) {
        const int beta_index = beta_start + offset;
        result.force_on_cube_world[finger] +=
            force(beta_index) * bases[beta_index];
      }
      result.contact_point_world[finger] = points[contact];
    }
  } else if (config_.contact_model == "anitescu") {
    const int rays_per_contact = 2 * config_.num_friction_directions;
    if (rays_per_contact <= 0 ||
        force.size() != contact_count * rays_per_contact) {
      return result;
    }
    for (int contact = 0; contact < contact_count; ++contact) {
      const int finger = active_fingers_[contact];
      const int start = contact * rays_per_contact;
      for (int offset = 0; offset < rays_per_contact; ++offset) {
        const int index = start + offset;
        result.force_on_cube_world[finger] += force(index) * bases[index];
      }
      result.contact_point_world[finger] = points[contact];
    }
  } else {
    return result;
  }

  result.valid = true;
  return result;
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

C3HoldPlanMetrics C3Planner::EvaluateHoldPlan() const {
  C3HoldPlanMetrics result;
  result.minimum_normal_force = std::numeric_limits<double>::infinity();
  if (!c3_ || c3_->GetForceSolution().empty() ||
      c3_->GetStateSolution().empty() || horizon_target_.empty()) {
    return result;
  }

  const std::vector<VectorXd> forces = c3_->GetForceSolution();
  const std::vector<VectorXd> states = c3_->GetStateSolution();
  const double lambda_scale = c3_->GetLambdaScaling();
  for (const VectorXd& force : forces) {
    const VectorXd physical_force = force / lambda_scale;
    for (const std::vector<int>& group : normal_groups_) {
      result.minimum_normal_force = std::min(
          result.minimum_normal_force, SumNormalForce(physical_force, group));
    }
  }

  // Reconstruct the actual force on the cube directly from the force basis
  // used by this C3 linearization.  This is more informative than inferring
  // Fz through the velocity dynamics: it separates lambda_n from the
  // tangential beta variables and preserves their signs in the world frame.
  const std::vector<Eigen::Vector3d>& force_bases =
      model_->contact_force_bases();
  if (config_.contact_model == "stewart_and_trinkle" &&
      force_bases.size() == static_cast<size_t>(dimensions_.lambda)) {
    const int beta_per_contact = 2 * config_.num_friction_directions;
    for (const VectorXd& force : forces) {
      const VectorXd physical_force = force / lambda_scale;
      if (physical_force.size() != dimensions_.lambda) break;

      std::vector<double> beta_fz(active_fingers_.size(), 0.0);
      std::vector<double> normal_fz(active_fingers_.size(), 0.0);
      double total_fz = 0.0;
      for (int variable = 0; variable < physical_force.size(); ++variable) {
        total_fz += physical_force(variable) * force_bases[variable].z();
      }
      for (size_t contact = 0; contact < active_fingers_.size(); ++contact) {
        for (const int normal_index : normal_groups_[contact]) {
          normal_fz[contact] +=
              physical_force(normal_index) * force_bases[normal_index].z();
        }
        const int beta_start = 2 * dimensions_.contacts +
                               static_cast<int>(contact) * beta_per_contact;
        for (int offset = 0; offset < beta_per_contact; ++offset) {
          const int beta_index = beta_start + offset;
          beta_fz[contact] +=
              physical_force(beta_index) * force_bases[beta_index].z();
        }
      }
      const double total_beta =
          std::accumulate(beta_fz.begin(), beta_fz.end(), 0.0);
      const double total_normal =
          std::accumulate(normal_fz.begin(), normal_fz.end(), 0.0);
      result.planned_beta_vertical_force.push_back(std::move(beta_fz));
      result.planned_normal_vertical_force.push_back(std::move(normal_fz));
      result.planned_total_beta_vertical_force.push_back(total_beta);
      result.planned_total_normal_vertical_force.push_back(total_normal);
      result.planned_total_vertical_contact_force.push_back(total_fz);
    }
  }

  const size_t knots = std::min(states.size(), horizon_target_.size());
  constexpr double kQuaternionEpsilon = 1e-12;
  for (size_t knot = 0; knot < knots; ++knot) {
    const VectorXd& predicted = states[knot];
    const VectorXd& target = horizon_target_[knot];
    if (predicted.size() != dimensions_.state ||
        target.size() != dimensions_.state) {
      return C3HoldPlanMetrics{};
    }

    // Drake's free-body q is [qw qx qy qz x y z] immediately after the
    // Allegro hand positions in this model.
    const int cube_q = dimensions_.hand_positions;
    const Eigen::Vector3d translation_error =
        predicted.segment<3>(cube_q + 4) - target.segment<3>(cube_q + 4);
    if (translation_error.norm() > result.maximum_translation_error) {
      result.maximum_translation_error = translation_error.norm();
      result.translation_error_at_maximum = translation_error;
      result.maximum_translation_error_knot = static_cast<int>(knot);
    }
    const Eigen::Vector4d q_pred = predicted.segment<4>(cube_q);
    const Eigen::Vector4d q_target = target.segment<4>(cube_q);
    const double q_norm = q_pred.norm() * q_target.norm();
    if (q_norm <= kQuaternionEpsilon) {
      result.maximum_orientation_error =
          std::numeric_limits<double>::infinity();
    } else {
      const double cosine = std::clamp(
          std::abs(q_pred.dot(q_target) / q_norm), 0.0, 1.0);
      result.maximum_orientation_error = std::max(
          result.maximum_orientation_error, 2.0 * std::acos(cosine));
    }

    // Free-body velocity is [angular; translational] after hand velocity.
    const int cube_v = dimensions_.positions + dimensions_.hand_velocities;
    result.maximum_angular_speed = std::max(
        result.maximum_angular_speed, predicted.segment<3>(cube_v).norm());
    result.maximum_linear_speed = std::max(
        result.maximum_linear_speed,
        predicted.segment<3>(cube_v + 3).norm());
  }

  // This is the same contact-only Fz diagnostic printed by --legacy_log,
  // but it is needed here because a spider support candidate is normally
  // rejected before the index is lifted.  A value near cube_weight means the
  // planned contact forces approximately balance gravity; Fz - weight is the
  // corresponding vertical force imbalance in this local C3 model.
  const auto& lcs = c3_->GetLCS();
  const std::vector<MatrixXd>& D = lcs.D();
  std::vector<VectorXd> cube_z_velocity_rows;
  cube_z_velocity_rows.reserve(forces.size());
  const int cube_z_velocity_row =
      dimensions_.positions + dimensions_.hand_velocities + 5;
  for (size_t knot = 0; knot < forces.size() && !D.empty(); ++knot) {
    const MatrixXd& D_k = D.at(std::min(knot, D.size() - 1));
    if (cube_z_velocity_row >= D_k.rows()) break;
    cube_z_velocity_rows.push_back(
        D_k.row(cube_z_velocity_row).transpose() / lambda_scale);
  }
  auto& plant = model_->plant();
  const auto cube_body =
      plant.GetBodyIndices(model_->cube_model()).at(0);
  const double cube_mass =
      plant.get_body(cube_body).get_mass(model_->context());
  result.cube_weight = cube_mass * 9.81;
  const int first_contact_force =
      config_.contact_model == "stewart_and_trinkle" ? dimensions_.contacts
                                                       : 0;
  result.planned_cube_vertical_contact_force =
      ComputeCandidateVerticalContactForces(
          cube_mass, config_.c3_dt, first_contact_force,
          cube_z_velocity_rows, forces);

  result.has_solution = c3_->GetLastSolveSucceeded() && knots > 0 &&
      std::isfinite(result.minimum_normal_force) &&
      std::isfinite(result.maximum_translation_error) &&
      std::isfinite(result.maximum_orientation_error) &&
      std::isfinite(result.maximum_linear_speed) &&
      std::isfinite(result.maximum_angular_speed);
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
