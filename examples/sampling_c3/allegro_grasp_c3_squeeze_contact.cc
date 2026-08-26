#include "examples/sampling_c3/allegro_grasp_c3_squeeze_contact.h"

#include <stdexcept>

namespace dairlib::allegro_grasp_c3 {

std::vector<std::vector<int>> MakeNormalForceGroups(
    const std::string& contact_model, int num_contacts,
    int num_friction_directions) {
  if (num_contacts < 0 || num_friction_directions < 0) {
    throw std::invalid_argument(
        "Contact count and friction-direction count must be nonnegative.");
  }

  std::vector<std::vector<int>> groups(num_contacts);
  if (contact_model == "anitescu") {
    const int block_size = 2 * num_friction_directions;
    for (int contact = 0; contact < num_contacts; ++contact) {
      for (int offset = 0; offset < block_size; ++offset) {
        groups[contact].push_back(contact * block_size + offset);
      }
    }
    return groups;
  }
  if (contact_model == "stewart_and_trinkle") {
    for (int contact = 0; contact < num_contacts; ++contact) {
      // Stewart--Trinkle layout:
      // lambda = [gamma(num_contacts), lambda_n(num_contacts), beta(...)].
      groups[contact].push_back(num_contacts + contact);
    }
    return groups;
  }

  throw std::invalid_argument(
      "contact_model must be 'stewart_and_trinkle' or 'anitescu'.");
}

double SumNormalForce(const Eigen::VectorXd& lambda,
                      const std::vector<int>& normal_group) {
  double result = 0.0;
  for (const int index : normal_group) {
    if (index < 0 || index >= lambda.size()) {
      throw std::out_of_range(
          "Normal-force group contains an index outside lambda.");
    }
    result += lambda(index);
  }
  return result;
}

std::vector<double> SumNormalForces(
    const Eigen::VectorXd& lambda,
    const std::vector<std::vector<int>>& normal_groups) {
  std::vector<double> result;
  result.reserve(normal_groups.size());
  for (const auto& group : normal_groups) {
    result.push_back(SumNormalForce(lambda, group));
  }
  return result;
}

}  // namespace dairlib::allegro_grasp_c3
