#include "allegro_grasp_c3_squeeze_maneuver_controller.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <drake/math/roll_pitch_yaw.h>

#include "allegro_grasp_c3_squeeze_maneuver.h"

namespace dairlib::allegro_grasp_c3 {

namespace {

// OSQP/ADMM can leave a sub-millinewton residual on a hard C3 contact-force
// bound. Treat that numerical residual as feasible, while leaving the bound
// itself at the requested safety margin. This is not a prescribed force.
constexpr double kC3NormalForceSolverTolerance = 1e-3;

// This is deliberately a measurement gate, separate from the C3 candidate
// plan gate below.  Rebuilding C3 with the index removed while the four-contact
// cube is still moving asks the new, smaller contact model to recover a
// transient that the established four-contact grasp could simply wait out.
struct SpiderSupportSettleMetrics {
  double yaw_error{std::numeric_limits<double>::infinity()};
  double translation_error{std::numeric_limits<double>::infinity()};
  double linear_speed{std::numeric_limits<double>::infinity()};
  double angular_speed{std::numeric_limits<double>::infinity()};
};

SpiderSupportSettleMetrics MeasureSpiderSupportSettling(
    const GraspSetup& grasp, const Eigen::VectorXd& state,
    const drake::math::RigidTransform<double>& cube_pose,
    const C3Planner& planner, double target_yaw) {
  SpiderSupportSettleMetrics result;
  const PlannerDimensions& dimensions = planner.dimensions();
  const int cube_velocity =
      dimensions.positions + dimensions.hand_velocities;
  if (state.size() < cube_velocity + 6) return result;

  // The spider turn pre-multiplies the initial cube orientation by a world-Z
  // yaw while keeping the cube centre stationary.  This exactly matches the
  // raw C3 reference generated in MakeCubeReference().
  const drake::math::RotationMatrix<double> measured_world_delta(
      cube_pose.rotation() * grasp.initial_cube_pose.rotation().inverse());
  const double measured_yaw =
      drake::math::RollPitchYaw<double>(measured_world_delta).yaw_angle();
  result.yaw_error = std::abs(
      std::remainder(measured_yaw - target_yaw, 2.0 * M_PI));
  result.translation_error =
      (cube_pose.translation() - grasp.initial_cube_pose.translation()).norm();
  result.angular_speed = state.segment<3>(cube_velocity).norm();
  result.linear_speed = state.segment<3>(cube_velocity + 3).norm();
  return result;
}

bool IsSpiderSupportSettled(const SpiderSupportSettleMetrics& metrics,
                            const SqueezeConfig& config) {
  return metrics.yaw_error <= config.spider_support_settle_yaw_error &&
         metrics.translation_error <=
             config.spider_support_settle_translation_error &&
         metrics.linear_speed <= config.spider_support_settle_linear_speed &&
         metrics.angular_speed <= config.spider_support_settle_angular_speed;
}

void PrintSpiderSupportSettleMetrics(
    double time, const char* prefix, const SpiderSupportSettleMetrics& metrics) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(6)
            << "[t=" << time << "] spider: " << prefix
            << " (yaw_err=" << metrics.yaw_error << " rad"
            << ", pos_err=" << metrics.translation_error << " m"
            << ", linear_speed=" << metrics.linear_speed << " m/s"
            << ", angular_speed=" << metrics.angular_speed << " rad/s)\n";
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

// This gate is deliberately based on the measured SAP force and contact
// state, not C3's assumed contacts.  Before ring joins the planner, C3 still
// has the established index-middle-thumb topology, so only the simulator can
// tell us whether ring touchdown disturbed one of those contacts.
struct SpiderRingHandoffMetrics {
  bool all_contacts{};
  bool near_target{};
  bool vertical_support{};
  double sap_vertical_force{};
  double minimum_vertical_force{};
  SpiderSupportSettleMetrics settle;
};

bool IsSpiderRingHandoffReady(const SpiderRingHandoffMetrics& metrics,
                              const SqueezeConfig& config) {
  return metrics.all_contacts && metrics.near_target &&
         metrics.vertical_support &&
         IsSpiderSupportSettled(metrics.settle, config);
}

void PrintSpiderRingHandoffMetrics(
    double time, const char* prefix,
    const std::array<bool, 4>& touching,
    const SpiderRingHandoffMetrics& metrics) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(6)
            << "[t=" << time << "] spider: " << prefix
            << " (touch=[index=" << touching[0]
            << " middle=" << touching[1]
            << " thumb=" << touching[2]
            << " ring=" << touching[3] << "]"
            << ", near_target=" << metrics.near_target
            << ", SAP_Fz=" << metrics.sap_vertical_force
            << " N, required>=" << metrics.minimum_vertical_force << " N"
            << ", yaw_err=" << metrics.settle.yaw_error << " rad"
            << ", pos_err=" << metrics.settle.translation_error << " m"
            << ", linear_speed=" << metrics.settle.linear_speed << " m/s"
            << ", angular_speed=" << metrics.settle.angular_speed
            << " rad/s)\n";
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

const char* FingerName(int finger) {
  constexpr std::array<const char*, 4> kNames{
      "index", "middle", "thumb", "ring"};
  return finger >= 0 && finger < static_cast<int>(kNames.size())
      ? kNames[finger]
      : "unknown";
}

// Prints the C3 plan itself, in the force directions used for its current
// linearization.  beta is the tangential/friction portion of the
// Stewart--Trinkle force; it is not a simulator-resolved SAP force.
void PrintSpiderCandidateVerticalForcePlan(
    double time, double force_knot_dt, const C3Planner& planner,
    const C3HoldPlanMetrics& hold) {
  const std::vector<int>& fingers = planner.active_fingers();
  const size_t knots = hold.planned_beta_vertical_force.size();
  if (knots == 0 || hold.planned_normal_vertical_force.size() != knots ||
      hold.planned_total_beta_vertical_force.size() != knots ||
      hold.planned_total_normal_vertical_force.size() != knots ||
      hold.planned_total_vertical_contact_force.size() != knots) {
    std::cout << "  C3 beta-F_z breakdown unavailable: force-basis data "
                 "did not match the current contact plan\n";
    return;
  }

  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n=== SPIDER C3 CANDIDATE FORCE PLAN @ t=" << time
            << " s ===\n"
            << "  Values are planned forces ON THE CUBE in the current "
               "C3 linearization (N; +F_z is upward). beta-F_z is the "
               "tangential/friction contribution. It is not SAP force.\n"
            << "  Each force knot is " << force_knot_dt
            << " s apart; cube weight=" << hold.cube_weight << " N.\n";
  std::cout << "  knot |";
  for (const int finger : fingers)
    std::cout << " " << FingerName(finger) << " beta-F_z |";
  std::cout << " sum beta-F_z | sum normal-F_z | total contact F_z | "
               "total-minus-weight\n";
  for (size_t knot = 0; knot < knots; ++knot) {
    if (hold.planned_beta_vertical_force[knot].size() != fingers.size() ||
        hold.planned_normal_vertical_force[knot].size() != fingers.size()) {
      std::cout << "  " << knot << " | inconsistent per-contact data\n";
      continue;
    }
    std::cout << "  " << std::setw(4) << knot << " |";
    for (const double beta_fz : hold.planned_beta_vertical_force[knot])
      std::cout << " " << std::setw(16) << beta_fz << " |";
    std::cout << " " << std::setw(12)
              << hold.planned_total_beta_vertical_force[knot] << " |"
              << " " << std::setw(14)
              << hold.planned_total_normal_vertical_force[knot] << " |"
              << " " << std::setw(18)
              << hold.planned_total_vertical_contact_force[knot] << " |"
              << " " << std::setw(18)
              << hold.planned_total_vertical_contact_force[knot] -
                     hold.cube_weight
              << "\n";
  }
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

// These checks are deliberately evaluated over the complete C3 horizon, not
// only at its first force knot.
bool HasSpiderCandidateForceSupport(const C3HoldPlanMetrics& hold,
                                    const SqueezeConfig& config) {
  return hold.has_solution &&
         hold.minimum_normal_force >=
             config.spider_support_normal_margin -
                 kC3NormalForceSolverTolerance;
}

bool HasSpiderCandidatePredictedMotionWithinBounds(
    const C3HoldPlanMetrics& hold, const SqueezeConfig& config) {
  return hold.maximum_translation_error <=
             config.spider_support_max_translation_error &&
         hold.maximum_orientation_error <=
             config.spider_support_max_orientation_error &&
         hold.maximum_linear_speed <=
             config.spider_support_max_linear_speed &&
         hold.maximum_angular_speed <=
             config.spider_support_max_angular_speed;
}

bool IsSpiderCandidateViable(const C3HoldPlanMetrics& hold,
                             const SqueezeConfig& config) {
  return HasSpiderCandidateForceSupport(hold, config) &&
         (config.spider_allow_unverified_index_lift ||
          HasSpiderCandidatePredictedMotionWithinBounds(hold, config));
}

void PrintSpiderCandidateAcceptanceCheck(double time,
                                         const C3HoldPlanMetrics& hold,
                                         const SqueezeConfig& config,
                                         bool viable) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  const double normal_force_limit =
      config.spider_support_normal_margin - kC3NormalForceSolverTolerance;
  const bool predicted_motion_within_bounds =
      HasSpiderCandidatePredictedMotionWithinBounds(hold, config);
  const bool predicted_motion_override_used =
      config.spider_allow_unverified_index_lift &&
      !predicted_motion_within_bounds;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n=== SPIDER THREE-CONTACT C3 ACCEPTANCE CHECK @ t=" << time
            << " s ===\n"
            << "  Values cover the complete C3 horizon for middle, thumb, "
               "and ring. Solver success and normal-force margin are "
               "always enforced.\n";
  if (config.spider_allow_unverified_index_lift) {
    std::cout << "  Diagnostic override enabled: predicted position, "
                 "orientation, and speed limits do not block index lift.\n";
  }
  std::cout
            << "  condition                         | C3 value       | "
               "required                         | result\n"
            << "  C3 solver result                  | "
            << std::setw(14) << (hold.has_solution ? "solved" : "no solution")
            << " | solved                           | "
            << (hold.has_solution ? "PASS" : "FAIL") << "\n";
  if (hold.has_solution) {
    const auto print_minimum = [&] {
      const bool pass = hold.minimum_normal_force >= normal_force_limit;
      std::cout << "  min contact normal force [N]       | " << std::setw(14)
                << hold.minimum_normal_force << " | >= "
                << config.spider_support_normal_margin << " requested ("
                << normal_force_limit << " with solver tol) | "
                << (pass ? "PASS" : "FAIL") << "\n";
    };
    const auto print_maximum = [](const char* label, double value,
                                  double limit, const char* unit) {
      std::cout << "  " << std::left << std::setw(34) << label << std::right
                << " | " << std::setw(14) << value << " | <= "
                << std::setw(12) << limit << " " << std::setw(7) << unit
                << " | " << (value <= limit ? "PASS" : "FAIL") << "\n";
    };
    print_minimum();
    print_maximum("max predicted position error [m]",
                  hold.maximum_translation_error,
                  config.spider_support_max_translation_error, "m");
    std::cout << "    position-error knot="
              << hold.maximum_translation_error_knot << " delta_xyz=["
              << hold.translation_error_at_maximum.transpose() << "]\n";
    print_maximum("max predicted orientation error [rad]",
                  hold.maximum_orientation_error,
                  config.spider_support_max_orientation_error, "rad");
    print_maximum("max predicted linear speed [m/s]",
                  hold.maximum_linear_speed,
                  config.spider_support_max_linear_speed, "m/s");
    print_maximum("max predicted angular speed [rad/s]",
                  hold.maximum_angular_speed,
                  config.spider_support_max_angular_speed, "rad/s");
  }
  if (predicted_motion_override_used) {
    std::cout << "  predicted-motion gate              | "
              << std::setw(14) << "override on"
              << " | position/orientation/speed limits | OVERRIDDEN\n";
  }
  std::cout << "  ---------------------------------------------------------------\n"
            << "  three-contact C3 candidate: "
            << (viable
                    ? (predicted_motion_override_used
                           ? "PASS -- prediction override; start 0.25 s verification"
                           : "PASS -- start 0.25 s verification")
                    : "FAIL -- keep index in contact")
            << "\n";
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

// One entry in the non-actuating ring-placement scan.  The state passed to
// C3 contains a virtual ring posture at target_C, but the live simulator
// state is never changed.
struct SpiderVirtualRingCandidate {
  int row{};
  int col{};
  Eigen::Vector3d target_C{Eigen::Vector3d::Zero()};
  bool ik_succeeded{};
  C3HoldPlanMetrics hold;
  double minimum_vertical_force{std::numeric_limits<double>::quiet_NaN()};
  double minimum_vertical_gravity_margin{
      std::numeric_limits<double>::quiet_NaN()};
  double stability_score{std::numeric_limits<double>::infinity()};
};

bool ComputeSpiderVirtualGravitySupport(SpiderVirtualRingCandidate* candidate) {
  DRAKE_DEMAND(candidate != nullptr);
  const C3HoldPlanMetrics& hold = candidate->hold;
  if (!hold.has_solution || hold.cube_weight <= 0.0 ||
      hold.planned_total_vertical_contact_force.empty()) {
    return false;
  }
  double minimum_force = std::numeric_limits<double>::infinity();
  for (const double force : hold.planned_total_vertical_contact_force) {
    if (!std::isfinite(force)) return false;
    minimum_force = std::min(minimum_force, force);
  }
  if (!std::isfinite(minimum_force)) return false;
  candidate->minimum_vertical_force = minimum_force;
  candidate->minimum_vertical_gravity_margin =
      minimum_force - hold.cube_weight;
  return true;
}

double ComputeSpiderVirtualStabilityScore(
    const SpiderVirtualRingCandidate& candidate,
    const SqueezeConfig& config) {
  const C3HoldPlanMetrics& hold = candidate.hold;
  if (!candidate.ik_succeeded || !hold.has_solution ||
      !std::isfinite(candidate.minimum_vertical_gravity_margin) ||
      !std::isfinite(hold.maximum_translation_error) ||
      !std::isfinite(hold.maximum_orientation_error) ||
      !std::isfinite(hold.maximum_linear_speed) ||
      !std::isfinite(hold.maximum_angular_speed)) {
    return std::numeric_limits<double>::infinity();
  }

  // This is a ranking score, not another acceptance gate.  Existing spider
  // limits merely normalize the five quantities to comparable magnitudes.
  // A vertical deficit is penalized, while the predicted pose/twist terms
  // naturally penalize excessive upward force as well as a falling plan.
  const auto square = [](double value) { return value * value; };
  const double gravity_deficit = std::max(
      0.0, -candidate.minimum_vertical_gravity_margin / hold.cube_weight);
  return square(hold.maximum_translation_error /
                config.spider_support_max_translation_error) +
         square(hold.maximum_orientation_error /
                config.spider_support_max_orientation_error) +
         square(hold.maximum_linear_speed /
                config.spider_support_max_linear_speed) +
         square(hold.maximum_angular_speed /
                config.spider_support_max_angular_speed) +
         square(gravity_deficit);
}

void PrintSpiderVirtualRingSearch(
    double time, const std::vector<SpiderVirtualRingCandidate>& candidates,
    int rows, int cols, double x_C, const SqueezeConfig& config,
    double elapsed_seconds) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  int ik_count = 0;
  int solved_count = 0;
  int scored_count = 0;
  for (const SpiderVirtualRingCandidate& candidate : candidates) {
    ik_count += candidate.ik_succeeded;
    solved_count += candidate.hold.has_solution;
    scored_count += std::isfinite(candidate.stability_score);
  }

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "\n=== SPIDER VIRTUAL RING 3-CONTACT C3 SEARCH @ t=" << time
            << " s ===\n"
            << "  Real hand is unchanged: index, middle, and thumb remain "
               "the live C3 contacts; ring stays parked.\n"
            << "  Every row below is a separate hypothetical "
               "middle-thumb-ring C3 problem; index is excluded only from "
               "that temporary problem.\n"
            << "  Target face is red/+X: x_C=" << x_C << " m, " << rows
            << " Z rows x " << cols << " Y columns. Values are predicted "
               "over the full C3 horizon relative to the frozen measured "
               "cube pose.\n"
            << "  score = normalized pose/twist error squared + normalized "
               "vertical-force deficit squared; lower is better.\n"
            << "  row col | y_C [m] | z_C [m] | IK | C3 | min_fn | min_Fz "
               "| min(Fz-W) | max_pos | max_rot | max_v | max_w | score\n";
  for (const SpiderVirtualRingCandidate& candidate : candidates) {
    std::cout << "  " << std::setw(3) << candidate.row << " "
              << std::setw(3) << candidate.col << " | " << std::setw(8)
              << candidate.target_C.y() << " | " << std::setw(8)
              << candidate.target_C.z() << " | "
              << (candidate.ik_succeeded ? "OK " : "NO ") << "| "
              << (candidate.hold.has_solution ? "OK " : "NO ") << "| ";
    if (!candidate.hold.has_solution) {
      std::cout << "    n/a |    n/a |      n/a |     n/a |     n/a | "
                   " n/a |  n/a |   n/a\n";
      continue;
    }
    std::cout << std::setw(6) << candidate.hold.minimum_normal_force << " | "
              << std::setw(6) << candidate.minimum_vertical_force << " | "
              << std::setw(8)
              << candidate.minimum_vertical_gravity_margin << " | "
              << std::setw(7) << candidate.hold.maximum_translation_error
              << " | " << std::setw(7)
              << candidate.hold.maximum_orientation_error << " | "
              << std::setw(5) << candidate.hold.maximum_linear_speed << " | "
              << std::setw(5) << candidate.hold.maximum_angular_speed
              << " | ";
    if (std::isfinite(candidate.stability_score)) {
      std::cout << std::setw(7) << candidate.stability_score;
    } else {
      std::cout << "    n/a";
    }
    std::cout << "\n";
  }

  const auto best = std::min_element(
      candidates.begin(), candidates.end(),
      [](const SpiderVirtualRingCandidate& a,
         const SpiderVirtualRingCandidate& b) {
        return a.stability_score < b.stability_score;
      });
  std::cout << "  --------------------------------------------------------------\n"
            << "  candidates: " << candidates.size() << " total, " << ik_count
            << " IK-reachable, " << solved_count << " C3-solved, "
            << scored_count << " fully scored. Wall time=" << elapsed_seconds
            << " s.\n";
  if (best != candidates.end() && std::isfinite(best->stability_score)) {
    std::cout << "  BEST virtual ring point: row=" << best->row
              << ", col=" << best->col << ", target_C=["
              << best->target_C.transpose() << "] m\n"
              << "    score=" << best->stability_score
              << ", min_Fz-W=" << best->minimum_vertical_gravity_margin
              << " N, max_pos=" << best->hold.maximum_translation_error
              << " m, max_rot=" << best->hold.maximum_orientation_error
              << " rad, max_v=" << best->hold.maximum_linear_speed
              << " m/s, max_w=" << best->hold.maximum_angular_speed
              << " rad/s.\n";
  } else {
    std::cout << "  No virtual point produced a scored C3 hold.\n";
  }
  std::cout << "  No candidate was actuated: ring remains parked and the "
               "live C3 topology remains index-middle-thumb.\n";
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

void RunSpiderVirtualRingSearch(
    double time, const Eigen::VectorXd& state,
    const drake::math::RigidTransform<double>& cube_pose,
    const C3Planner& live_planner, GraspSetup* grasp,
    const SqueezeConfig& config) {
  DRAKE_DEMAND(grasp != nullptr);
  const PlannerDimensions& dimensions = live_planner.dimensions();
  if (state.size() != dimensions.state) {
    std::cout << "[t=" << time
              << "] spider: virtual ring search skipped because the live "
                 "state size does not match the C3 state layout\n";
    return;
  }

  const int rows = config.spider_ring_search_rows;
  const int cols = config.spider_ring_search_cols;
  const double half = grasp->cube_size / 2.0;
  const double x_C = half - config.relay_ring_press;
  const double edge = config.spider_ring_search_face_margin;
  const double in_face_min = -half + edge;
  const double in_face_max = half - edge;
  const auto grid_coordinate = [in_face_min, in_face_max](int index,
                                                           int count) {
    if (count <= 1) return 0.5 * (in_face_min + in_face_max);
    return in_face_min + (in_face_max - in_face_min) * index /
                             static_cast<double>(count - 1);
  };

  // The scan uses a dedicated model so C3's temporary witness points and
  // contact bases cannot overwrite the live planner's linearization.
  SqueezeConfig analysis_config = config;
  analysis_config.input_scale = live_planner.input_scale();
  LcsModel analysis_model(config.cube_size_scale);
  std::vector<SpiderVirtualRingCandidate> candidates;
  candidates.reserve(rows * cols);
  const auto scan_start = std::chrono::steady_clock::now();

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      SpiderVirtualRingCandidate candidate;
      candidate.row = row;
      candidate.col = col;
      candidate.target_C = Eigen::Vector3d(
          x_C, grid_coordinate(col, cols), grid_coordinate(row, rows));

      bool ik_ok = false;
      const Eigen::VectorXd virtual_hand = grasp->SolveLegIk(
          cube_pose, 3, candidate.target_C,
          state.head(dimensions.hand_positions), &ik_ok);
      candidate.ik_succeeded = ik_ok &&
          virtual_hand.size() >= GraspSetup::kFingerStarts[3] + 4;
      if (!candidate.ik_succeeded) {
        candidates.push_back(std::move(candidate));
        continue;
      }

      // Retain the measured index/middle/thumb state.  Only ring's four
      // virtual joint positions and velocity are substituted into the C3
      // linearization state; this is never sent to the simulator.
      Eigen::VectorXd virtual_state = state;
      const int ring_start = GraspSetup::kFingerStarts[3];
      virtual_state.segment<4>(ring_start) =
          virtual_hand.segment<4>(ring_start);
      virtual_state.segment<4>(dimensions.positions + ring_start).setZero();

      // Freeze the desired cube pose at the measured state and ask for zero
      // velocity.  The resulting prediction answers exactly whether this
      // virtual support geometry can hold the cube where it is now.
      Eigen::VectorXd virtual_target = virtual_state;
      virtual_target.tail(dimensions.state - dimensions.positions).setZero();

      C3Planner virtual_planner(analysis_config, &analysis_model);
      virtual_planner.SetBaseTarget(virtual_target);
      virtual_planner.Rebuild({1, 2, 3}, virtual_state, time,
                              config.spider_support_normal_margin,
                              false /* log_rebuild */);
      candidate.hold = virtual_planner.EvaluateHoldPlan();
      ComputeSpiderVirtualGravitySupport(&candidate);
      candidate.stability_score =
          ComputeSpiderVirtualStabilityScore(candidate, config);
      candidates.push_back(std::move(candidate));
    }
  }

  const double elapsed_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - scan_start).count();
  PrintSpiderVirtualRingSearch(time, candidates, rows, cols, x_C, config,
                               elapsed_seconds);
}

}  // namespace

void ManeuverController::UpdateGait(
    double time, double reference_time,
    const std::array<bool, 4>& touching, const Eigen::VectorXd& state,
    const drake::math::RigidTransform<double>& cube_pose,
    double sap_vertical_contact_force, double cube_weight, C3Planner* planner,
    Eigen::VectorXd* contact_start, Eigen::VectorXd* contact_end) {
  if (gait_phase_ == GaitPhase::kDone) return;
  if (gait_retreating_) {
    if (time < gait_retreat_start_time_ + gait_retreat_trajectory_.end_time())
      return;
    gait_retreating_ = false;
    gait_touch_hold_active_ = false;
    gait_touch_latched_ = false;
    gait_entered_ = false;
    gait_phase_ = GaitPhase::kDone;
    std::cout << "[t=" << time
              << "] spider: ring handoff aborted safely; ring is parked and "
                 "the original index-middle-thumb C3 hold remains active\n";
    return;
  }
  if (IsSpider() && ring_parked_positions_.size() == 0) {
    // The first spider-walk primitive begins from an established three-finger
    // grasp.  Keep the already-clear ring finger exactly where it is; there is
    // no reason to solve a new IK target until the later support-post step.
    ring_parked_positions_ =
        state.head(planner->dimensions().hand_positions);
  } else if (UsesParkedRing() && ring_parked_positions_.size() == 0) {
    bool parked_ok = false;
    ring_parked_positions_ = grasp_->SolveLegIk(
        cube_pose, 3, grasp_->relay_ring_park,
        state.head(planner->dimensions().hand_positions), &parked_ok);
    if (!parked_ok)
      std::cout << "[t=" << time << "] gait: ring park IK infeasible\n";
  }
  const std::vector<int> rotation_contacts =
      UsesParkedRing() ? std::vector<int>{0, 1, 2}
                       : std::vector<int>{0, 1, 2, 3};

  if (gait_phase_ == GaitPhase::kRotate) {
    const bool ready = gait_cycle_ > 0 || reference_time >= config_.gait_hold_time;
    if (!gait_entered_ && ready) {
      const double delta = IsSpider()
          ? config_.spider_yaw_delta
          : config_.gait_delta;
      gait_theta_start_ = gait_cycle_ * delta;
      gait_theta_target_ = (gait_cycle_ + 1) * delta;
      gait_rotate_start_ = reference_time;
      gait_entered_ = true;
      if (planner->active_fingers() != rotation_contacts)
        planner->Rebuild(rotation_contacts, state, time);
      ring_engaged_ = !UsesParkedRing();
      std::cout << "[t=" << time << "] "
                << (IsSpider() ? "spider small turn" : "gait cycle ")
                << (IsSpider() ? "" : std::to_string(gait_cycle_ + 1))
                << ": rotating about "
                << (IsSpider() ? "world +Z" : "cube +Y") << " to "
                << gait_theta_target_ * 180.0 / M_PI << " deg\n";
    }
    const double rotate_duration = IsSpider()
        ? config_.spider_yaw_duration
        : config_.gait_rotate_duration;
    if (gait_entered_ &&
        reference_time >= gait_rotate_start_ + rotate_duration +
                              config_.gait_hold_time) {
      if (IsSpider()) {
        if (grasp_->grasp_finger_count == 4) {
          gait_phase_ = GaitPhase::kDone;
          gait_entered_ = false;
          std::cout << "[t=" << time
                    << "] spider: four-contact yaw complete; ring was "
                       "already seated, no ring placement commanded\n";
          return;
        }
        if (config_.spider_yaw_only) {
          gait_phase_ = GaitPhase::kDone;
          gait_entered_ = false;
          std::cout << "[t=" << time
                    << "] spider: yaw-only test complete at "
                    << gait_theta_target_ * 180.0 / M_PI
                    << " deg; ring remains parked and the original "
                       "index-middle-thumb C3 hold remains active\n";
          return;
        }
        if (config_.spider_virtual_ring_search) {
          std::cout << "[t=" << time
                    << "] spider: yaw complete; running virtual 10x10 ring "
                       "placement C3 search while keeping the real "
                       "index-middle-thumb hold\n";
          RunSpiderVirtualRingSearch(time, state, cube_pose, *planner,
                                     grasp_, config_);
          gait_phase_ = GaitPhase::kDone;
          std::cout << "[t=" << time
                    << "] spider: virtual ring search complete; ring stays "
                       "parked and no index lift is commanded\n";
          return;
        }
        gait_phase_ = GaitPhase::kMove;
        gait_leg_ = 0;
        gait_entered_ = false;
        std::cout << "[t=" << time
                  << "] spider small turn complete at "
                  << gait_theta_target_ * 180.0 / M_PI
                  << " deg yaw; placing ring support\n";
        return;
      }
      gait_phase_ = GaitPhase::kMove;
      gait_leg_ = 0;
      gait_entered_ = false;
    }
    return;
  }

  const int finger = gait_plan_[gait_leg_].first;
  const GaitLegKind kind = gait_plan_[gait_leg_].second;
  if (!gait_entered_ && time < gait_leg_done_time_ + config_.gait_leg_gap)
    return;

  const auto complete_gait_leg = [&](const Eigen::VectorXd& joint_target) {
    CompleteFinger(finger, joint_target, gait_target_C_, state, time, planner,
                   contact_start, contact_end);
    if (kind == GaitLegKind::kEngage) ring_engaged_ = true;
    gait_entered_ = false;
    gait_left_surface_ = false;
    gait_touch_latched_ = false;
    gait_touch_hold_active_ = false;
    spider_ring_handoff_valid_since_ = -1.0;
    gait_leg_done_time_ = time;
    std::cout << "[t=" << time << "] gait: finger " << finger
              << " joined C3\n";

    if (++gait_leg_ < static_cast<int>(gait_plan_.size())) return;
    gait_leg_ = 0;
    if (IsSpider()) {
      gait_phase_ = GaitPhase::kDone;
      if (config_.spider_index_crawl_after_ring) {
        std::cout << "[t=" << time
                  << "] spider index crawl complete; holding four contacts\n";
      } else {
        std::cout << "[t=" << time
                  << "] spider ring placement complete; holding four contacts "
                     "with index on yellow\n";
      }
      return;
    }
    if (gait_realign_mode_) {
      gait_phase_ = GaitPhase::kDone;
      std::cout << "[t=" << time << "] gait: realignment complete\n";
    } else if (++gait_cycle_ >= config_.gait_cycles) {
      if (config_.gait_realign && config_.gait_scheme != "relay") {
        gait_realign_mode_ = true;
      } else {
        gait_phase_ = GaitPhase::kDone;
      }
    } else {
      gait_phase_ = GaitPhase::kRotate;
    }
  };

  if (!gait_entered_) {
    if (kind == GaitLegKind::kSpiderCrawl) {
      // Do not take index away merely because middle, thumb, and ring touch
      // the cube. First make C3 solve the candidate three-contact hold and
      // prove that its complete short horizon has compression margin and
      // bounded cube motion. This keeps the force allocation inside C3;
      // there are no prescribed middle/thumb/ring force targets here.
      constexpr std::array<int, 3> kSupportFingers{1, 2, 3};
      const bool support_ready = std::all_of(
          kSupportFingers.begin(), kSupportFingers.end(),
          [&touching](int support) { return touching[support]; });
      if (!support_ready) {
        spider_support_settled_since_ = -1.0;
        if (gait_support_wait_start_time_ < 0.0)
          gait_support_wait_start_time_ = time;
        if (time >= gait_support_wait_start_time_ + config_.gait_leg_timeout) {
          if (spider_support_candidate_start_time_ >= 0.0)
            planner->Rebuild({0, 1, 2, 3}, state, time);
          gait_phase_ = GaitPhase::kDone;
          std::cout << "[t=" << time
                    << "] spider: support check failed; keeping index on "
                       "yellow and stopping before crawl\n";
        }
        return;
      }
      gait_support_wait_start_time_ = -1.0;

      if (!spider_support_candidate_approved_) {
        // Keep the known four-contact C3 topology until the ring has not only
        // touched, but has helped the cube settle near the completed yaw
        // target. This avoids creating a three-contact QP during the ring
        // touchdown transient.
        if (spider_support_candidate_start_time_ < 0.0) {
          if (spider_support_settle_wait_start_time_ < 0.0) {
            spider_support_settle_wait_start_time_ = time;
            spider_support_settled_since_ = -1.0;
            std::cout << "[t=" << time
                      << "] spider: ring support is present; keeping the "
                         "four-contact C3 hold until the measured cube "
                         "settles before the three-contact evaluation\n";
          }
          const SpiderSupportSettleMetrics settle =
              MeasureSpiderSupportSettling(*grasp_, state, cube_pose,
                                           *planner, gait_theta_target_);
          if (!IsSpiderSupportSettled(settle, config_)) {
            spider_support_settled_since_ = -1.0;
            if (time >= spider_support_settle_wait_start_time_ +
                            config_.gait_leg_timeout) {
              gait_phase_ = GaitPhase::kDone;
              PrintSpiderSupportSettleMetrics(
                  time,
                  "four-contact settle gate timed out; keeping index on "
                  "yellow and stopping before crawl",
                  settle);
            }
            return;
          }
          if (spider_support_settled_since_ < 0.0) {
            spider_support_settled_since_ = time;
            PrintSpiderSupportSettleMetrics(
                time,
                "four-contact cube settled; verifying before three-contact "
                "C3 evaluation",
                settle);
          }
          if (time < spider_support_settled_since_ +
                         config_.spider_support_settle_time) {
            return;
          }
        }
        if (spider_support_candidate_start_time_ < 0.0) {
          planner->Rebuild(
              {1, 2, 3}, state, time,
              config_.spider_support_normal_margin);
          spider_support_candidate_start_time_ = time;
          spider_support_settle_wait_start_time_ = -1.0;
          spider_support_settled_since_ = -1.0;
          spider_support_valid_since_ = -1.0;
          spider_support_force_plan_logged_ = false;
          std::cout << "[t=" << time
                    << "] spider: evaluating C3 three-contact hold before "
                       "index lift (middle, thumb, ring)\n";
          return;
        }

        const C3HoldPlanMetrics hold = planner->EvaluateHoldPlan();
        const bool first_force_plan =
            hold.has_solution && !spider_support_force_plan_logged_;
        if (first_force_plan) {
          spider_support_force_plan_logged_ = true;
          std::cout << "[t=" << time
                    << "] spider: first C3 three-contact force plan "
                       "(middle, thumb, ring; before acceptance gate)\n";
          PrintSpiderCandidateVerticalForcePlan(
              time, config_.c3_dt, *planner, hold);
        }
        const bool viable = IsSpiderCandidateViable(hold, config_);
        if (first_force_plan)
          PrintSpiderCandidateAcceptanceCheck(time, hold, config_, viable);
        if (!viable) {
          spider_support_valid_since_ = -1.0;
          if (time >= spider_support_candidate_start_time_ +
                          config_.gait_leg_timeout) {
            gait_phase_ = GaitPhase::kDone;
            const std::ios::fmtflags saved_flags = std::cout.flags();
            const std::streamsize saved_precision = std::cout.precision();
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "[t=" << time
                      << "] spider: C3 rejected the three-contact hold; "
                         "keeping index on yellow and stopping before crawl"
                      << " (min_fn=" << hold.minimum_normal_force
                      << ", pos_err=" << hold.maximum_translation_error
                      << " at k=" << hold.maximum_translation_error_knot
                      << " delta_xyz=["
                      << hold.translation_error_at_maximum.transpose() << "]"
                      << ", rot_err=" << hold.maximum_orientation_error
                      << ", linear_speed=" << hold.maximum_linear_speed
                      << ", angular_speed=" << hold.maximum_angular_speed
                      << ")\n";
            PrintSpiderCandidateVerticalForcePlan(
                time, config_.c3_dt, *planner, hold);
            PrintSpiderCandidateAcceptanceCheck(time, hold, config_, viable);
            std::cout.flags(saved_flags);
            std::cout.precision(saved_precision);
            // Print the candidate while its three-contact topology is still
            // active; only then restore the established four-contact hold.
            planner->Rebuild({0, 1, 2, 3}, state, time);
          }
          return;
        }
        if (spider_support_valid_since_ < 0.0) {
          spider_support_valid_since_ = time;
          if (!first_force_plan)
            PrintSpiderCandidateAcceptanceCheck(time, hold, config_, viable);
          std::cout << "[t=" << time
                    << "] spider: candidate three-contact C3 hold is "
                       "viable; verifying for "
                    << config_.spider_support_verify_time << " s\n";
          return;
        }
        if (time < spider_support_valid_since_ +
                       config_.spider_support_verify_time) {
          return;
        }
        spider_support_candidate_approved_ = true;
        std::cout << "[t=" << time
                  << "] spider: C3 approved three-contact hold; lifting "
                     "index for crawl\n";
      }
    }

    if (kind != GaitLegKind::kEngage &&
        !(kind == GaitLegKind::kSpiderCrawl &&
          spider_support_candidate_approved_)) {
      std::vector<int> remaining;
      for (const int active : planner->active_fingers())
        if (active != finger) remaining.push_back(active);
      planner->Rebuild(remaining, state, time);
    }

    if (kind == GaitLegKind::kEngage) {
      gait_target_C_ = IsSpider() ? grasp_->spider_ring_hold
                                   : grasp_->relay_ring_hold;
    } else if (kind == GaitLegKind::kSpiderCrawl) {
      gait_target_C_ = grasp_->footprints[finger];
      // Yellow/4 is -Y; blue/2 is its -X neighbour. Crawl along the current
      // yellow face without crossing its edge.
      gait_target_C_.x() -= config_.spider_index_crawl;
    } else if (gait_realign_mode_) {
      gait_target_C_ = grasp_->initial_footprints[finger];
    } else {
      const drake::math::RotationMatrix<double> walk_back(
          drake::math::RollPitchYaw<double>(0.0, -config_.gait_delta, 0.0));
      gait_target_C_ = walk_back * grasp_->footprints[finger];
    }
    Eigen::Vector3d midpoint = kind == GaitLegKind::kEngage
                                   ? (IsSpider() ? grasp_->spider_ring_park
                                                 : grasp_->relay_ring_park)
                                   : 0.5 * (grasp_->footprints[finger] +
                                            gait_target_C_);
    if (kind != GaitLegKind::kEngage)
      midpoint.y() = -(grasp_->cube_size / 2.0 +
                       (kind == GaitLegKind::kSpiderCrawl
                            ? config_.spider_index_arc_clearance
                            : config_.regrasp_arc_clearance));
    bool destination_ok = false;
    bool midpoint_ok = false;
    Eigen::VectorXd q_mid;
    const auto solve_engage_target = [&]() {
      q_mid = grasp_->SolveLegIk(cube_pose, finger, midpoint,
                                  *contact_start, &midpoint_ok);
      // For the spider's ring-engagement leg, the outside waypoint is also a
      // better initial guess for the final near-face IK than the old, parked
      // posture. This follows the physical approach path before considering
      // a more central fallback point.
      const Eigen::VectorXd& destination_seed =
          kind == GaitLegKind::kEngage && IsSpider() ? q_mid
                                                      : *contact_start;
      gait_destination_ = grasp_->SolveLegIk(
          cube_pose, finger, gait_target_C_, destination_seed,
          &destination_ok);
    };
    solve_engage_target();

    if (kind == GaitLegKind::kEngage && IsSpider() &&
        (!destination_ok || !midpoint_ok)) {
      // The preferred point is on yellow / -Y a few millimetres above face
      // centre. If it is just beyond the joint limits, walk only its height
      // toward face centre in 1 mm increments, while preserving the yellow
      // contact plane and the outside -Y approach waypoint.
      const Eigen::Vector3d requested_target = gait_target_C_;
      const double toward_centre = gait_target_C_.z() < 0.0
          ? 1.0 : (gait_target_C_.z() > 0.0 ? -1.0 : 0.0);
      for (int step = 1; toward_centre != 0.0 && step <= 15 &&
                         (!destination_ok || !midpoint_ok); ++step) {
        gait_target_C_.z() = requested_target.z() +
            toward_centre * 0.001 * step;
        if ((toward_centre > 0.0 && gait_target_C_.z() > 0.0) ||
            (toward_centre < 0.0 && gait_target_C_.z() < 0.0)) {
          gait_target_C_.z() = 0.0;
        }
        midpoint = gait_target_C_;
        midpoint.y() = -(grasp_->cube_size / 2.0 +
                         config_.relay_ring_retract);
        destination_ok = false;
        midpoint_ok = false;
        solve_engage_target();
        if (gait_target_C_.z() == 0.0) break;
      }
      if (destination_ok && midpoint_ok) {
        std::cout << "[t=" << time
                  << "] spider: ring target adjusted toward face centre "
                  << "for reachable IK: yellow-z=" << gait_target_C_.z()
                  << " m\n";
      } else {
        gait_phase_ = GaitPhase::kDone;
        std::cout << "[t=" << time
                  << "] spider: no reachable ring-support target; "
                  << "stopping before handoff\n";
        return;
      }
    }
    if (kind == GaitLegKind::kSpiderCrawl &&
        (!destination_ok || !midpoint_ok)) {
      // An infeasible crawl is not a reason to drop index support. Restore
      // the just-removed topology before stopping, and never command the
      // solver's rejected last iterate.
      planner->Rebuild({0, 1, 2, 3}, state, time);
      gait_phase_ = GaitPhase::kDone;
      std::cout << "[t=" << time
                << "] spider: index crawl IK infeasible; keeping the "
                   "four-contact hold\n";
      return;
    }
    gait_target_W_ = cube_pose * gait_target_C_;
    if (kind == GaitLegKind::kEngage && IsSpider()) {
      std::cout << "[t=" << time
                << "] spider: placing ring at selected target_C=["
                << gait_target_C_.transpose() << "] m\n";
    }
    const double trajectory_duration =
        kind == GaitLegKind::kSpiderCrawl
            ? config_.spider_index_duration
            : (kind == GaitLegKind::kEngage && IsSpider()
                   ? config_.spider_ring_placement_duration
                   : config_.regrasp_duration);
    gait_trajectory_ = MakeThreeKnotJointTrajectory(
        state.head(planner->dimensions().hand_positions), q_mid,
        gait_destination_, trajectory_duration);
    gait_midpoint_ = q_mid;
    gait_trajectory_start_ = time;
    gait_left_surface_ = kind == GaitLegKind::kEngage;
    gait_touch_latched_ = false;
    gait_touch_hold_active_ = false;
    spider_ring_handoff_valid_since_ = -1.0;
    spider_ring_handoff_gate_invalid_reported_ = false;
    gait_entered_ = true;
    if (!destination_ok || !midpoint_ok) {
      std::cout << "[t=" << time << "] gait: warning, finger " << finger
                << " IK infeasible (target=" << destination_ok
                << ", outside_waypoint=" << midpoint_ok << ")\n";
    }
    return;
  }

  const double elapsed = time - gait_trajectory_start_;
  const bool spider_ring_engage =
      IsSpider() && kind == GaitLegKind::kEngage && finger == 3;
  const auto begin_spider_ring_retreat = [&](const char* reason) {
    const Eigen::VectorXd current =
        state.head(planner->dimensions().hand_positions);
    if (gait_midpoint_.size() != current.size() ||
        ring_parked_positions_.size() != current.size()) {
      gait_touch_hold_active_ = false;
      gait_touch_latched_ = false;
      gait_entered_ = false;
      gait_phase_ = GaitPhase::kDone;
      std::cout << "[t=" << time << "] spider: " << reason
                << "; could not construct a safe ring retreat, stopping "
                   "with the original three-contact C3 hold\n";
      return;
    }
    gait_retreat_trajectory_ = MakeThreeKnotJointTrajectory(
        current, gait_midpoint_, ring_parked_positions_,
        config_.spider_ring_placement_duration);
    gait_retreat_start_time_ = time;
    gait_retreating_ = true;
    gait_touch_hold_active_ = false;
    gait_touch_latched_ = false;
    spider_ring_handoff_valid_since_ = -1.0;
    std::cout << "[t=" << time << "] spider: " << reason
              << "; retracting ring while retaining the original "
                 "index-middle-thumb C3 hold\n";
  };

  if (spider_ring_engage && !gait_touch_hold_active_) {
    // A first contact can be a grazing touch before the final, deliberately
    // slow seating portion of the trajectory. Record it, but let that
    // already-smooth trajectory finish instead of freezing short of the
    // intended small penetration. The handoff gate begins only once ring is
    // seated at its final yellow-face target.
    if (gait_left_surface_ && touching[finger] && !gait_touch_latched_) {
      gait_touch_latched_ = true;
      std::cout << "[t=" << time
                << "] spider: ring first touch; continuing the gentle final "
                   "seat before four-contact verification\n";
    }
    if (elapsed >= gait_trajectory_.end_time() && gait_left_surface_ &&
        touching[finger]) {
      gait_touch_hold_positions_ = gait_destination_;
      gait_touch_hold_active_ = true;
      gait_touch_time_ = time;
      gait_target_W_ = cube_pose * gait_target_C_;
      spider_ring_handoff_valid_since_ = -1.0;
      spider_ring_handoff_gate_invalid_reported_ = false;
      std::cout << "[t=" << time
                << "] spider: ring seated at its yellow-face target; "
                   "freezing its joint target and retaining the "
                   "index-middle-thumb C3 hold\n";
    } else {
      if (elapsed > gait_trajectory_.end_time() + config_.gait_leg_timeout) {
        begin_spider_ring_retreat(
            "ring did not produce a detectable seated contact before the "
            "deadline");
      }
      return;
    }
  }

  if (spider_ring_engage && gait_touch_hold_active_) {
    const bool near_target =
        (grasp_->FingertipPosition(finger) - gait_target_W_).norm() <
        config_.regrasp_touch_tol;
    SpiderRingHandoffMetrics metrics;
    metrics.all_contacts =
        std::all_of(touching.begin(), touching.end(),
                    [](bool contact) { return contact; });
    metrics.near_target = near_target;
    metrics.sap_vertical_force = sap_vertical_contact_force;
    metrics.minimum_vertical_force = std::max(
        0.0, cube_weight - config_.spider_ring_handoff_vertical_force_deficit);
    metrics.vertical_support = cube_weight > 0.0 &&
        sap_vertical_contact_force >= metrics.minimum_vertical_force;
    metrics.settle = MeasureSpiderSupportSettling(
        *grasp_, state, cube_pose, *planner, gait_theta_target_);

    if (IsSpiderRingHandoffReady(metrics, config_)) {
      spider_ring_handoff_gate_invalid_reported_ = false;
      if (spider_ring_handoff_valid_since_ < 0.0) {
        spider_ring_handoff_valid_since_ = time;
        PrintSpiderRingHandoffMetrics(
            time, "ring handoff conditions met; verifying while C3 remains "
                  "three-contact",
            touching, metrics);
        return;
      }
      if (time < spider_ring_handoff_valid_since_ +
                     config_.spider_support_settle_time) {
        return;
      }
      PrintSpiderRingHandoffMetrics(
          time, "ring handoff verified; rebuilding C3 with four confirmed "
                "contacts",
          touching, metrics);
      // Seed the new C3 target from the measured hand posture. This prevents
      // the topology change from simultaneously snapping the ring farther
      // into the cube; the normal C3/IK tracking resumes after the handoff.
      complete_gait_leg(state.head(planner->dimensions().hand_positions));
      return;
    }

    if (spider_ring_handoff_valid_since_ >= 0.0 ||
        !spider_ring_handoff_gate_invalid_reported_) {
      PrintSpiderRingHandoffMetrics(
          time, "ring handoff gate waiting; retaining three-contact C3",
          touching, metrics);
      spider_ring_handoff_gate_invalid_reported_ = true;
    }
    spider_ring_handoff_valid_since_ = -1.0;
    if (time >= gait_touch_time_ + config_.gait_leg_timeout) {
      begin_spider_ring_retreat(
          "ring handoff verification timed out without four stable contacts");
    }
    return;
  }

  if (elapsed > gait_trajectory_.end_time() &&
      static_cast<long>(time / 0.001) %
              std::max(1, config_.gait_seek_period_steps) ==
          0) {
    bool seek_ok = false;
    const Eigen::VectorXd updated = grasp_->SolveLegIk(
        cube_pose, finger, gait_target_C_, *contact_start, &seek_ok);
    if (seek_ok) {
      gait_destination_ = updated;
      gait_target_W_ = cube_pose * gait_target_C_;
    }
  }
  if (!gait_left_surface_ && !touching[finger]) gait_left_surface_ = true;
  const bool near_target =
      (grasp_->FingertipPosition(finger) - gait_target_W_).norm() <
      config_.regrasp_touch_tol;
  if (!gait_touch_latched_ &&
      elapsed >= gait_trajectory_.end_time() && gait_left_surface_ &&
      touching[finger] && near_target) {
    gait_touch_latched_ = true;
    gait_touch_time_ = time;
    std::cout << "[t=" << time << "] gait: finger " << finger
              << " re-contacted; settling\n";
  }
  if (!gait_touch_latched_ &&
      elapsed > gait_trajectory_.end_time() + config_.gait_leg_timeout) {
    if (spider_ring_engage) {
      begin_spider_ring_retreat(
          "ring did not produce a detectable contact before the deadline");
      return;
    }
    gait_phase_ = GaitPhase::kDone;
    const Eigen::Vector3d tip_W = grasp_->FingertipPosition(finger);
    std::cout << "[t=" << time << "] gait: finger " << finger
              << " failed to re-contact; stopping gait (tip-target="
              << (tip_W - gait_target_W_).norm() << " m, tip_W=["
              << tip_W.transpose() << "], target_W=["
              << gait_target_W_.transpose() << "])\n";
    return;
  }
  if (!gait_touch_latched_ ||
      time < gait_touch_time_ + config_.regrasp_settle_time) {
    return;
  }

  complete_gait_leg(gait_destination_);
}

}  // namespace dairlib::allegro_grasp_c3
