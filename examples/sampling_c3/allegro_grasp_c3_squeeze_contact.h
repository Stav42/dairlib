#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace dairlib::allegro_grasp_c3 {

// Returns, for each active contact, the entries in C3's stacked lambda vector
// whose sum is that contact's physical normal force. Keeping this contact-model
// bookkeeping separate makes force mapping testable without a Drake plant.
std::vector<std::vector<int>> MakeNormalForceGroups(
    const std::string& contact_model, int num_contacts,
    int num_friction_directions);

// Sums one contact's normal-force entries. Invalid indices are a programming
// error and are reported with std::out_of_range.
double SumNormalForce(const Eigen::VectorXd& lambda,
                      const std::vector<int>& normal_group);

// Applies SumNormalForce to every group in order.
std::vector<double> SumNormalForces(
    const Eigen::VectorXd& lambda,
    const std::vector<std::vector<int>>& normal_groups);

}  // namespace dairlib::allegro_grasp_c3
