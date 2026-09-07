#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <drake/multibody/plant/contact_results.h>
#include <drake/multibody/tree/multibody_tree_indexes.h>

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

// A compact version of the historical per-solve C3 plan report.  The force
// solution is in C3's scaled lambda units; lambda_scaling converts it back to
// physical normal-force units before it is printed.
struct LegacyPlanDiagnostic {
  double time{};
  double knot_dt{};
  int cube_z_position_index{};
  double lambda_scaling{1.0};
  double input_scale{1.0};
  std::vector<int> active_fingers;
  std::vector<std::vector<int>> normal_groups;
  std::vector<Eigen::VectorXd> state_plan;
  std::vector<Eigen::VectorXd> input_plan;
  std::vector<Eigen::VectorXd> force_plan;
  std::vector<double> planned_cube_vertical_contact_force;
};

void PrintLegacyC3Plan(const LegacyPlanDiagnostic& diagnostic);

// Each supplied row maps C3's stored force-solution vector at one knot to the
// cube's vertical velocity increment.  The helper converts that increment to
// net planned contact force via m_cube / dt.  Keeping this algebra independent
// of Drake and C3 makes it directly unit-testable.
struct PlannedVerticalContactForceRequest {
  double cube_mass{};
  double knot_dt{};
  int first_contact_force_index{};
  std::vector<Eigen::VectorXd> cube_z_velocity_increment_per_lambda;
  std::vector<Eigen::VectorXd> force_plan;
};

std::vector<double> ComputePlannedVerticalContactForces(
    const PlannedVerticalContactForceRequest& request);

// The complete hand-joint trajectory is intentionally a separate diagnostic:
// it is useful when the OSC PD target comes from C3, but is much longer than
// the compact C3 PLAN report above.
struct JointPlanDiagnostic {
  double time{};
  int hand_position_count{};
  Eigen::VectorXd measured_hand_positions;
  std::vector<Eigen::VectorXd> state_plan;
};

void PrintC3JointPlan(const JointPlanDiagnostic& diagnostic);

// Reports the two non-gravity pieces of OSC's applied torque on the same tick
// that a new C3 plan was solved.  Percentages normalize the two component
// norms, not their vector sum, because joint-wise cancellation can occur.
struct OscTorqueSplitDiagnostic {
  double time{};
  std::array<int, 4> finger_start{};
  Eigen::VectorXd commanded_torque;
  Eigen::VectorXd pd_torque;
  Eigen::VectorXd c3_normal_force_jacobian_torque;
  std::array<Eigen::Vector3d, 4> commanded_force_world{};
  std::array<Eigen::Vector3d, 4> pd_force_world{};
  std::array<Eigen::Vector3d, 4> normal_direction_world{};
  std::array<double, 4> commanded_force_fit_residual{};
  std::array<double, 4> pd_force_fit_residual{};
  std::array<double, 4> c3_normal_force_target{};
  std::array<double, 4> c3_normal_force_applied{};
};

void PrintOscTorqueSplit(const OscTorqueSplitDiagnostic& diagnostic);

// SAP's resolved force at one fingertip aggregated across every point-pair
// contact between that fingertip and the cube.  The normal force is the
// positive compression component on the cube, projected onto each contact
// pair's *geometric* normal; it is not a Jacobian or torque proxy.
struct SapFingerContactForce {
  int point_contact_count{};
  double normal_force{};
  double tangential_force_magnitude_sum{};
  double max_slip_speed{};
  Eigen::Vector3d force_on_cube_world{Eigen::Vector3d::Zero()};
  // Arithmetic mean of the resolved point-pair contact geometry for this
  // fingertip.  These are suitable for low-rate wrench feedback, not a
  // replacement for individual point-pair constraints.
  Eigen::Vector3d contact_point_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal_into_cube_world{Eigen::Vector3d::Zero()};
};

struct SapFingertipCubeContactSummary {
  std::array<SapFingerContactForce, 4> fingers{};
  Eigen::Vector3d net_force_on_cube_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d moment_about_cube_center_world{Eigen::Vector3d::Zero()};
};

SapFingertipCubeContactSummary SummarizeSapFingertipCubeContacts(
    const drake::multibody::ContactResults<double>& contacts,
    drake::multibody::BodyIndex cube_body,
    const std::array<drake::multibody::BodyIndex, 4>& tip_bodies,
    const Eigen::Vector3d& cube_center_world);

// Reports the actual contact forces applied to the cube by SAP.  When an OSC
// command is available, its raw C3 normal force and post-scale/crossfade normal
// force are included for a like-for-like requested-versus-resolved comparison.
// In experimental full-contact-force mode it also reports the planned and
// resolved yaw moments about the current cube center.
struct SapContactForceDiagnostic {
  double time{};
  double osc_command_time{};
  bool has_osc_command{};
  bool osc_used_full_contact_force{};
  double lambda_torque_scale{1.0};
  SapFingertipCubeContactSummary sap;
  std::array<double, 4> c3_normal_force_target{};
  std::array<double, 4> c3_normal_force_applied{};
  std::array<Eigen::Vector3d, 4> c3_force_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> osc_force_command_on_cube_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> c3_contact_point_world{
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::Vector3d cube_center_world{Eigen::Vector3d::Zero()};
};

void PrintSapContactForceDiagnostic(const SapContactForceDiagnostic& diagnostic);

}  // namespace dairlib::allegro_grasp_c3
