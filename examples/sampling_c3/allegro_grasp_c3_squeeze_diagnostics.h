#pragma once

#include <string>

namespace c3 {
class LCS;
}

namespace dairlib::allegro_grasp_c3 {

struct LambdaMapDiagnostic {
  int contact_count{};
  int hand_position_count{};
  int hand_velocity_count{};
  int position_count{};
  int state_count{};
  int lambda_count{};
  int friction_direction_count{};
  double lambda_scaling{};
  double alpha_m{};
  double dt{};
  std::string contact_model;
};

void PrintCubeLambdaMap(const c3::LCS& lcs, double time,
                        const LambdaMapDiagnostic& dimensions);

}  // namespace dairlib::allegro_grasp_c3
