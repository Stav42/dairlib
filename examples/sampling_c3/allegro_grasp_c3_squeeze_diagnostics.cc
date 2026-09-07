#include "allegro_grasp_c3_squeeze_diagnostics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "c3/core/lcs.h"

namespace dairlib::allegro_grasp_c3 {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr std::array<const char*, 4> kFingerNames{
    "index", "middle", "thumb", "ring"};

std::string FingerName(int finger) {
  if (finger >= 0 && finger < static_cast<int>(kFingerNames.size()))
    return kFingerNames[finger];
  return "finger" + std::to_string(finger);
}

double NormalForce(const VectorXd& scaled_force,
                   const std::vector<int>& normal_group,
                   double lambda_scaling) {
  if (lambda_scaling == 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  double result = 0.0;
  for (const int index : normal_group) {
    if (index < 0 || index >= scaled_force.size())
      return std::numeric_limits<double>::quiet_NaN();
    result += scaled_force(index) / lambda_scaling;
  }
  return result;
}

double YawMomentAboutCubeCenter(
    const std::array<Vector3d, 4>& force_on_cube_world,
    const std::array<Vector3d, 4>& contact_point_world,
    const Vector3d& cube_center_world) {
  double result = 0.0;
  for (int finger = 0; finger < 4; ++finger) {
    result += (contact_point_world.at(finger) - cube_center_world)
                  .cross(force_on_cube_world.at(finger))
                  .z();
  }
  return result;
}

void PrintFingerNorms(const VectorXd& torque,
                      const std::array<int, 4>& finger_start) {
  std::cout << " [";
  for (int finger = 0; finger < 4; ++finger) {
    const int start = finger_start[finger];
    const double norm = start >= 0 && start + 4 <= torque.size()
                            ? torque.segment(start, 4).norm()
                            : 0.0;
    if (finger > 0) std::cout << ", ";
    std::cout << kFingerNames[finger] << " " << norm;
  }
  std::cout << "]";
}

}  // namespace

void PrintCubeLambdaMap(const c3::LCS& lcs, double time,
                        const LambdaMapDiagnostic& d) {
  const int cube_position_row = d.hand_position_count + 4;
  const int cube_velocity_row = d.position_count + d.hand_velocity_count + 3;
  const int beta_per_contact = 2 * d.friction_direction_count;
  const MatrixXd physical_D = lcs.D()[0] / d.lambda_scaling;
  const MatrixXd D_xyz =
      physical_D.block(cube_position_row, 0, 3, d.lambda_count);
  const MatrixXd D_vxyz =
      physical_D.block(cube_velocity_row, 0, 3, d.lambda_count);

  std::cout << std::fixed << std::setprecision(6)
            << "\n=== D: cube xyz <- lambda @ t=" << time
            << " s (physical, unscaled LCS) ===\n"
            << "  state rows: cube_x=" << cube_position_row
            << " cube_y=" << cube_position_row + 1
            << " cube_z=" << cube_position_row + 2
            << "   (n_x=" << d.state_count
            << ", n_lambda=" << d.lambda_count << ")\n"
            << "  C3 AnDn_=" << d.lambda_scaling << "\n";

  const char* row_names[3] = {"cube_x", "cube_y", "cube_z"};
  auto print_block = [&](const char* title, int first_column, int count) {
    std::cout << "  -- " << title << " (cols " << first_column << ".."
              << first_column + count - 1 << ") --\n           ";
    for (int column = 0; column < count; ++column)
      std::cout << std::setw(11)
                << ("c" + std::to_string(first_column + column));
    std::cout << "\n";
    for (int row = 0; row < 3; ++row) {
      std::cout << "  " << std::setw(8) << row_names[row];
      for (int column = 0; column < count; ++column)
        std::cout << std::setw(11) << D_xyz(row, first_column + column);
      std::cout << "\n";
    }
  };

  if (d.contact_model == "stewart_and_trinkle") {
    print_block("gamma (slacks; expect ~0)", 0, d.contact_count);
    print_block("lambda_n", d.contact_count, d.contact_count);
    for (int contact = 0; contact < d.contact_count; ++contact) {
      const int first = 2 * d.contact_count + contact * beta_per_contact;
      const std::string title = "beta contact " + std::to_string(contact);
      print_block(title.c_str(), first, beta_per_contact);
    }

    const VectorXd D_z = D_xyz.row(2).transpose();
    const VectorXd D_zn =
        D_z.segment(d.contact_count, d.contact_count);
    const VectorXd D_zb = D_z.tail(d.lambda_count - 2 * d.contact_count);
    std::cout << "  -- cube_z row summaries --\n"
              << "  D_z[lambda_n] (m per N) = " << D_zn.transpose() << "\n"
              << "  |D_z[gamma]|_max="
              << D_z.head(d.contact_count).cwiseAbs().maxCoeff()
              << "  |D_z[lambda_n]|_max=" << D_zn.cwiseAbs().maxCoeff()
              << "  |D_z[beta]|_max=" << D_zb.cwiseAbs().maxCoeff() << "\n"
              << "  Delta z from unit lambda_n: " << D_zn.sum()
              << " m/step\n";
    if (d.contact_count == 3) {
      std::cout << "  Delta z from alpha_m target: "
                << d.alpha_m * D_zn(0) + d.alpha_m * D_zn(1) +
                       2.0 * d.alpha_m * D_zn(2)
                << " m/step\n";
    }
    std::cout << "  freefall Delta z over dt: "
              << -0.5 * 9.81 * d.dt * d.dt << " m\n"
              << "  -- cube linear velocity <- lambda_n --\n"
              << "    vx: "
              << D_vxyz.block(0, d.contact_count, 1, d.contact_count) << "\n"
              << "    vy: "
              << D_vxyz.block(1, d.contact_count, 1, d.contact_count) << "\n"
              << "    vz: "
              << D_vxyz.block(2, d.contact_count, 1, d.contact_count) << "\n";
  } else {
    print_block("combined cone forces", 0, d.lambda_count);
  }
  std::cout << std::defaultfloat;
}

void PrintLegacyC3Plan(const LegacyPlanDiagnostic& d) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  const int force_knots = static_cast<int>(d.force_plan.size());

  std::cout << std::fixed << std::setprecision(4)
            << "\n=== C3 PLAN @ t=" << d.time << " s ("
            << force_knots << " force knots, dt=" << d.knot_dt
            << " s; active fingers=[";
  for (size_t i = 0; i < d.active_fingers.size(); ++i) {
    if (i > 0) std::cout << ",";
    std::cout << FingerName(d.active_fingers[i]);
  }
  std::cout << "]) ===\n";

  if (force_knots == 0) {
    std::cout << "  No force plan is available.\n";
    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
    return;
  }

  std::cout << "  normal force from C3 lambda (N):\n"
            << std::setw(12) << "finger";
  for (int knot = 0; knot < force_knots; ++knot)
    std::cout << std::setw(11) << ("k" + std::to_string(knot));
  std::cout << "\n";
  const size_t rows = std::min(d.active_fingers.size(), d.normal_groups.size());
  for (size_t row = 0; row < rows; ++row) {
    std::cout << std::setw(12) << FingerName(d.active_fingers[row]);
    for (const VectorXd& force : d.force_plan)
      std::cout << std::setw(11)
                << NormalForce(force, d.normal_groups[row], d.lambda_scaling);
    std::cout << "\n";
  }

  std::cout << "  physical C3 command norm |u_k| (N m):"
            << std::setw(0);
  for (const VectorXd& input : d.input_plan)
    std::cout << " " << d.input_scale * input.norm();
  std::cout << "\n";

  if (!d.planned_cube_vertical_contact_force.empty()) {
    std::cout << "  C3-planned sum F_z on cube (N; +up):";
    for (const double force_z : d.planned_cube_vertical_contact_force)
      std::cout << " " << force_z;
    std::cout << "\n";
  }

  if (d.cube_z_position_index >= 0) {
    std::cout << "  planned cube z (m):";
    for (const VectorXd& state : d.state_plan) {
      if (d.cube_z_position_index < state.size())
        std::cout << " " << state(d.cube_z_position_index);
      else
        std::cout << " n/a";
    }
    std::cout << "\n";
  }
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

std::vector<double> ComputePlannedVerticalContactForces(
    const PlannedVerticalContactForceRequest& request) {
  const size_t knot_count = std::min(
      request.cube_z_velocity_increment_per_lambda.size(),
      request.force_plan.size());
  std::vector<double> result;
  result.reserve(knot_count);
  for (size_t knot = 0; knot < knot_count; ++knot) {
    const VectorXd& velocity_row =
        request.cube_z_velocity_increment_per_lambda[knot];
    const VectorXd& lambda = request.force_plan[knot];
    const int first = request.first_contact_force_index;
    if (request.knot_dt <= 0.0 || first < 0 || first >= velocity_row.size() ||
        first >= lambda.size() || velocity_row.size() != lambda.size()) {
      result.push_back(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    const double delta_vz = velocity_row.tail(velocity_row.size() - first)
                                .dot(lambda.tail(lambda.size() - first));
    result.push_back(request.cube_mass * delta_vz / request.knot_dt);
  }
  return result;
}

void PrintC3JointPlan(const JointPlanDiagnostic& d) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  const int n_q = std::min(d.hand_position_count,
                           static_cast<int>(d.measured_hand_positions.size()));
  if (n_q <= 0) return;

  const VectorXd q_measured = d.measured_hand_positions.head(n_q);
  std::cout << "[C3 QPLAN t=" << d.time << "] q_meas=["
            << q_measured.transpose() << "]\n";
  for (size_t knot = 0; knot < d.state_plan.size(); ++knot) {
    const VectorXd& state = d.state_plan[knot];
    if (state.size() < n_q) continue;
    const VectorXd q_k = state.head(n_q);
    const VectorXd delta = q_k - q_measured;
    std::cout << "  q[" << knot << "]=[" << q_k.transpose()
              << "]  |qk-qmeas|=" << delta.norm()
              << " max=" << delta.cwiseAbs().maxCoeff();
    if (knot == 1 && n_q >= 16) {
      std::cout << "  q1err_by=[idx " << delta.segment<4>(0).norm()
                << ", mid " << delta.segment<4>(4).norm()
                << ", thu " << delta.segment<4>(12).norm()
                << ", ring " << delta.segment<4>(8).norm() << "]";
    }
    std::cout << "\n";
  }
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

void PrintOscTorqueSplit(const OscTorqueSplitDiagnostic& d) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  const double command_norm = d.commanded_torque.norm();
  const double pd_norm = d.pd_torque.norm();
  const double c3_normal_norm = d.c3_normal_force_jacobian_torque.norm();
  const double magnitude_sum = pd_norm + c3_normal_norm;
  const double pd_percent = magnitude_sum > 1e-12 ? 100.0 * pd_norm / magnitude_sum
                                                    : 0.0;
  const double c3_percent = magnitude_sum > 1e-12
                                ? 100.0 * c3_normal_norm / magnitude_sum
                                : 0.0;

  std::cout << std::fixed << std::setprecision(3)
            << "[OSC TORQUE SPLIT t=" << d.time << "] "
            << "|tau_cmd|=" << command_norm << " N m, "
            << "|tau_PD|=" << pd_norm << " N m, "
            << "|tau_JTf(C3 normal)|=" << c3_normal_norm << " N m; "
            << "PD=" << pd_percent << "%, C3-normal=" << c3_percent
            << "% (shares of |PD|+|J^T f_n|; gravity excluded)\n"
            << "  PD by finger:";
  PrintFingerNorms(d.pd_torque, d.finger_start);
  std::cout << "\n  C3 normal Jacobian by finger:";
  PrintFingerNorms(d.c3_normal_force_jacobian_torque, d.finger_start);
  std::cout << "\n\n"
            << "  TORQUE-TO-FORCE PROXIES (world frame, N)\n"
            << "  Each F solves J_tip^T F approximately equal to that "
               "finger's torque. tau_cmd is the final saturated command; "
               "this is a torque-equivalent proxy, not Drake's resolved "
               "contact force. n.F uses OSC's C3 press direction; C3 f_n "
               "applied includes its scale and join crossfade.\n"
            << "  finger | F_cmd=[x y z]          | n.F_cmd | "
               "F_PD=[x y z]           | n.F_PD | C3 f_n raw | "
               "C3 f_n applied | fit cmd/PD\n";
  for (int finger = 0; finger < 4; ++finger) {
    const Vector3d& command_force = d.commanded_force_world.at(finger);
    const Vector3d& pd_force = d.pd_force_world.at(finger);
    const Vector3d& normal = d.normal_direction_world.at(finger);
    std::cout << "  " << std::left << std::setw(6) << kFingerNames[finger]
              << std::right << " | ["
              << std::setw(7) << command_force.x() << " "
              << std::setw(7) << command_force.y() << " "
              << std::setw(7) << command_force.z() << "] |"
              << std::setw(8) << normal.dot(command_force) << " | ["
              << std::setw(7) << pd_force.x() << " " << std::setw(7)
              << pd_force.y() << " " << std::setw(7) << pd_force.z() << "]"
              << " |" << std::setw(7) << normal.dot(pd_force) << " |"
              << std::setw(11) << d.c3_normal_force_target.at(finger) << " |"
              << std::setw(14) << d.c3_normal_force_applied.at(finger) << " | "
              << std::setw(5) << d.commanded_force_fit_residual.at(finger)
              << "/" << std::setw(5) << d.pd_force_fit_residual.at(finger)
              << "\n";
  }
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

SapFingertipCubeContactSummary SummarizeSapFingertipCubeContacts(
    const drake::multibody::ContactResults<double>& contacts,
    const drake::multibody::BodyIndex cube_body,
    const std::array<drake::multibody::BodyIndex, 4>& tip_bodies,
    const Eigen::Vector3d& cube_center_world) {
  SapFingertipCubeContactSummary result;
  for (int contact_index = 0;
       contact_index < contacts.num_point_pair_contacts(); ++contact_index) {
    const auto& info = contacts.point_pair_contact_info(contact_index);
    const bool cube_is_a = info.bodyA_index() == cube_body;
    const bool cube_is_b = info.bodyB_index() == cube_body;
    if (!cube_is_a && !cube_is_b) continue;

    int finger = -1;
    for (int candidate = 0; candidate < 4; ++candidate) {
      const auto tip_body = tip_bodies.at(candidate);
      if ((cube_is_a && info.bodyB_index() == tip_body) ||
          (cube_is_b && info.bodyA_index() == tip_body)) {
        finger = candidate;
        break;
      }
    }
    if (finger < 0) continue;

    // PointPairContactInfo stores f_Bc_W: force applied to body B by body A.
    // nhat_BA_W points from B into A.  Choose both the force and the normal in
    // the direction that presses *into the cube*, independent of pair order.
    const Vector3d force_on_cube =
        cube_is_b ? info.contact_force() : -info.contact_force();
    const Vector3d normal_into_cube =
        cube_is_b ? -info.point_pair().nhat_BA_W
                  : info.point_pair().nhat_BA_W;
    const double normal_force =
        std::max(0.0, force_on_cube.dot(normal_into_cube));
    const Vector3d tangential_force =
        force_on_cube - normal_force * normal_into_cube;
    // SAP reports force at the point-pair contact.  The two witness points
    // straddle the tiny penetration depth, so their midpoint is a stable
    // moment arm for the resolved contact force.
    const Vector3d contact_point =
        0.5 * (info.point_pair().p_WCa + info.point_pair().p_WCb);

    SapFingerContactForce& summary = result.fingers.at(finger);
    ++summary.point_contact_count;
    summary.normal_force += normal_force;
    summary.tangential_force_magnitude_sum += tangential_force.norm();
    summary.max_slip_speed =
        std::max(summary.max_slip_speed, std::abs(info.slip_speed()));
    summary.force_on_cube_world += force_on_cube;
    summary.contact_point_world += contact_point;
    summary.normal_into_cube_world += normal_into_cube;
    result.net_force_on_cube_world += force_on_cube;
    result.moment_about_cube_center_world +=
        (contact_point - cube_center_world).cross(force_on_cube);
  }
  for (SapFingerContactForce& summary : result.fingers) {
    if (summary.point_contact_count == 0) continue;
    summary.contact_point_world /= summary.point_contact_count;
    const double normal_norm = summary.normal_into_cube_world.norm();
    if (normal_norm > 1e-12)
      summary.normal_into_cube_world /= normal_norm;
  }
  return result;
}

void PrintSapContactForceDiagnostic(const SapContactForceDiagnostic& d) {
  const std::ios::fmtflags saved_flags = std::cout.flags();
  const std::streamsize saved_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(3)
            << "\n=== SAP FINGERTIP-CUBE FORCES @ t=" << d.time << " s ===\n"
            << "  SAP is the simulator-resolved force ON THE CUBE. f_n is "
               "the geometric-normal compression; sum|f_t| is the magnitude "
               "sum of its friction components.\n";
  if (d.has_osc_command) {
    std::cout << "  C3 raw / OSC applied values are the normal-force command "
                 "issued at t="
              << d.osc_command_time << " s (one 1 ms control tick earlier); "
                 "applied includes lambda_torque_scale="
              << d.lambda_torque_scale << " and any join crossfade.\n";
    if (d.osc_used_full_contact_force) {
      std::cout << "  OSC force execution: full C3 lambda_n + beta contact "
                   "force through J^T; C3 joint input u is not applied.\n";
    }
  } else {
    std::cout << "  No prior OSC normal-force command is available yet.\n";
  }
  std::cout << "  finger | pts | C3 f_n raw | OSC f_n applied | SAP f_n | "
               "SAP sum|f_t| | SAP F_z | max slip\n";

  int total_points = 0;
  double total_normal = 0.0;
  double total_tangential = 0.0;
  double total_vertical = 0.0;
  double max_slip = 0.0;
  for (int finger = 0; finger < 4; ++finger) {
    const SapFingerContactForce& summary = d.sap.fingers.at(finger);
    total_points += summary.point_contact_count;
    total_normal += summary.normal_force;
    total_tangential += summary.tangential_force_magnitude_sum;
    total_vertical += summary.force_on_cube_world.z();
    max_slip = std::max(max_slip, summary.max_slip_speed);
    std::cout << "  " << std::left << std::setw(6) << FingerName(finger)
              << std::right << " |" << std::setw(4)
              << summary.point_contact_count << " |";
    if (d.has_osc_command) {
      std::cout << std::setw(11) << d.c3_normal_force_target.at(finger)
                << " |" << std::setw(16)
                << d.c3_normal_force_applied.at(finger) << " |";
    } else {
      std::cout << std::setw(11) << "n/a" << " |" << std::setw(16)
                << "n/a" << " |";
    }
    std::cout << std::setw(8) << summary.normal_force << " |"
              << std::setw(13) << summary.tangential_force_magnitude_sum
              << " |" << std::setw(8) << summary.force_on_cube_world.z()
              << " |" << std::setw(9) << summary.max_slip_speed << "\n";
  }
  std::cout << "  " << std::left << std::setw(6) << "total" << std::right
            << " |" << std::setw(4) << total_points << " |"
            << std::setw(11) << "" << " |" << std::setw(16) << "" << " |"
            << std::setw(8) << total_normal << " |" << std::setw(13)
            << total_tangential << " |" << std::setw(8) << total_vertical
            << " |" << std::setw(9) << max_slip << "\n";
  std::cout << "\n=== CUBE YAW MOMENT @ t=" << d.time << " s ===\n"
            << "  Moments are ON THE CUBE about its current center; +M_z "
               "is world +Z. C3 uses its current linearization witness "
               "points.\n"
            << "  source                    | M_z [N m]\n"
            << std::setprecision(6);
  if (d.has_osc_command && d.osc_used_full_contact_force) {
    std::cout << "  M_z C3 first-knot plan    |" << std::setw(12)
              << YawMomentAboutCubeCenter(d.c3_force_on_cube_world,
                                          d.c3_contact_point_world,
                                          d.cube_center_world)
              << "\n"
              << "  M_z OSC force command     |" << std::setw(12)
              << YawMomentAboutCubeCenter(
                     d.osc_force_command_on_cube_world,
                     d.c3_contact_point_world, d.cube_center_world)
              << "\n";
  } else {
    std::cout << "  M_z C3 first-knot plan    |" << std::setw(12) << "n/a"
              << "\n"
              << "  M_z OSC force command     |" << std::setw(12) << "n/a"
              << "\n";
  }
  std::cout << "  M_z SAP resolved contact  |" << std::setw(12)
            << d.sap.moment_about_cube_center_world.z() << "\n";
  std::cout.flags(saved_flags);
  std::cout.precision(saved_precision);
}

}  // namespace dairlib::allegro_grasp_c3
