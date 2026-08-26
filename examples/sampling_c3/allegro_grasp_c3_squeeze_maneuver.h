#pragma once

#include <Eigen/Core>

#include <drake/common/trajectories/piecewise_polynomial.h>

namespace dairlib::allegro_grasp_c3 {

enum class RecontactEvent { kNone, kContactLatched, kSettled };

// Pure transition function for the break-contact, remake-contact, settle
// sequence shared by ring, middle, and index regrasp legs.
RecontactEvent UpdateRecontactState(
    bool touching, bool near_target, double time, double settle_time,
    bool* has_left_surface, bool* contact_latched,
    double* contact_latched_time);

drake::trajectories::PiecewisePolynomial<double>
MakeThreeKnotJointTrajectory(const Eigen::VectorXd& current,
                             const Eigen::VectorXd& midpoint,
                             const Eigen::VectorXd& destination,
                             double duration);

}  // namespace dairlib::allegro_grasp_c3
