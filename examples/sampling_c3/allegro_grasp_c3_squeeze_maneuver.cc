#include "allegro_grasp_c3_squeeze_maneuver.h"

#include <stdexcept>
#include <vector>

namespace dairlib::allegro_grasp_c3 {

RecontactEvent UpdateRecontactState(
    bool touching, bool near_target, double time, double settle_time,
    bool* has_left_surface, bool* contact_latched,
    double* contact_latched_time) {
  if (has_left_surface == nullptr || contact_latched == nullptr ||
      contact_latched_time == nullptr) {
    throw std::invalid_argument("Recontact state pointers must be non-null.");
  }
  if (!*has_left_surface && !touching) *has_left_surface = true;
  if (*has_left_surface && !*contact_latched && touching && near_target) {
    *contact_latched = true;
    *contact_latched_time = time;
    return RecontactEvent::kContactLatched;
  }
  if (*contact_latched && time >= *contact_latched_time + settle_time) {
    return RecontactEvent::kSettled;
  }
  return RecontactEvent::kNone;
}

drake::trajectories::PiecewisePolynomial<double>
MakeThreeKnotJointTrajectory(const Eigen::VectorXd& current,
                             const Eigen::VectorXd& midpoint,
                             const Eigen::VectorXd& destination,
                             double duration) {
  if (duration <= 0.0) {
    throw std::invalid_argument("Trajectory duration must be positive.");
  }
  if (current.size() != midpoint.size() || current.size() != destination.size()) {
    throw std::invalid_argument("Trajectory knot dimensions must match.");
  }
  const std::vector<Eigen::MatrixXd> knots{current, midpoint, destination};
  return drake::trajectories::PiecewisePolynomial<double>::CubicShapePreserving(
      {0.0, 0.5 * duration, duration}, knots, true);
}

}  // namespace dairlib::allegro_grasp_c3
