#include "allegro_grasp_c3_squeeze_diagnostics.h"

#include <iomanip>
#include <iostream>

#include <Eigen/Core>

#include "c3/core/lcs.h"

namespace dairlib::allegro_grasp_c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;

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

}  // namespace dairlib::allegro_grasp_c3
