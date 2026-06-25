// Increment A: single-solve test of the three-finger grasp C3 problem.
//
// Builds the Allegro hand + cube plant, places the three fingertips at the
// grasp contact points via IK, linearizes the contact dynamics to an LCS, and
// runs ONE C3 solve with the force-reference cost active. It then prints the
// planned normal forces against the alpha targets. The point is to verify the
// plant -> contacts -> LCS -> C3 -> force-cost pipeline compiles and that
// --alpha_m actually flows into the solved contact forces, before adding any
// simulation/visualization (that is Increment B).
//
// Stewart-Trinkle lambda layout for 3 contacts: [gamma(3) | lambda_n(3) | lambda_t].
// With contact order [index, middle, thumb], the normal forces are lambda[3..5].

#include <iostream>
#include <memory>
#include <vector>

#include <gflags/gflags.h>

#include <drake/geometry/scene_graph.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/solvers/osqp_solver.h>
#include <drake/solvers/solver_options.h>
#include <drake/systems/framework/diagram_builder.h>

#include "c3/core/c3.h"
#include "c3/core/c3_options.h"
#include "c3/core/c3_qp.h"
#include "c3/core/lcs.h"
#include "c3/multibody/lcs_factory.h"
#include "c3/multibody/lcs_factory_options.h"

#include "allegro_hand_utils.h"
#include "common/find_resource.h"
#include "cube_kinematics.h"

namespace dairlib {

using drake::AutoDiffXd;
using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::math::RigidTransform;
using drake::math::RotationMatrix;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::solvers::OsqpSolver;
using drake::solvers::SolverOptions;
using drake::systems::DiagramBuilder;
using drake::systems::System;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using c3::C3;
using c3::C3Options;
using c3::C3QP;
using c3::LCS;
using c3::LCSFactoryOptions;
using c3::multibody::GetContactModelMap;
using c3::multibody::LCSFactory;

DEFINE_double(alpha_m, 1.0, "Middle-finger normal grip-force target (N).");
DEFINE_double(w_force, 100.0,
              "Diagonal weight on the normal-force tracking cost.");
DEFINE_double(w_cube, 100.0, "Q weight on the cube pose/velocity (anchor).");
DEFINE_double(w_state, 0.1, "Q weight on all other state rows.");
DEFINE_double(w_R, 0.01, "R weight on joint torques.");
DEFINE_double(w_G, 1.0, "ADMM G consensus weight.");
DEFINE_double(w_U, 1.0, "ADMM U consensus weight.");
DEFINE_int32(N, 10, "Prediction horizon (knot points).");
DEFINE_double(dt, 0.01, "LCS timestep (s).");
DEFINE_double(mu, 0.5, "Friction coefficient used by the LCS.");
DEFINE_int32(num_friction_directions, 2,
             "Friction directions per contact (tangential dim = 2x this).");
DEFINE_double(osqp_eps, 1e-3,
              "OSQP convergence tolerance (eps_abs/eps_rel). Looser than the "
              "1e-5 default so the stiff hand QP can actually converge.");
DEFINE_double(finger_inset, -0.009,
              "How far inside each face to place the fingertip FRAME (m). The "
              "tip collision sphere (~12 mm radius) adds to this, so a small "
              "negative value gives a shallow contact gap. Tune it watching the "
              "printed 'contact gaps phi' (aim ~ -0.003).");
DEFINE_double(input_scale, 0.0,
              "Torque-input rescaling s_u; 0 = auto = |A0|/|B0|. Shrinks B so "
              "the QP is well-conditioned. A change of variables on u only — it "
              "does not change the contact forces lambda.");

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // ---- 1. Build the model plant (continuous; the factory discretizes) ----
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);

  ModelInstanceIndex allegro_index =
      AddAllegroHandToPlant(&plant, &scene_graph);

  Parser parser(&plant, &scene_graph);
  ModelInstanceIndex cube_index = parser.AddModels(FindResourceOrThrow(
      "examples/sampling_c3/urdf/numbered_cube/numbered_cube.sdf"))[0];

  plant.Finalize();

  // AutoDiff twin for the linearization (mirrors the Franka controller).
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant_ad =
      System<double>::ToAutoDiffXd(plant);

  auto diagram = builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  auto plant_context_ad = plant_ad->CreateDefaultContext();

  const int num_allegro_joints = plant.num_positions(allegro_index);

  // ---- 2. Contact geometry pairs: [index, middle, thumb] <-> cube ----
  // This ordering fixes the normal forces at lambda indices 3, 4, 5.
  GeometryId cube_geom = plant.GetCollisionGeometriesForBody(
      plant.get_body(plant.GetBodyIndices(cube_index)[0]))[0];
  GeometryId index_geom = plant.GetCollisionGeometriesForBody(
      plant.GetBodyByName("link_3_tip", allegro_index))[0];
  GeometryId middle_geom = plant.GetCollisionGeometriesForBody(
      plant.GetBodyByName("link_7_tip", allegro_index))[0];
  GeometryId thumb_geom = plant.GetCollisionGeometriesForBody(
      plant.GetBodyByName("link_15_tip", allegro_index))[0];

  std::vector<SortedPair<GeometryId>> contact_pairs{
      SortedPair<GeometryId>(index_geom, cube_geom),
      SortedPair<GeometryId>(middle_geom, cube_geom),
      SortedPair<GeometryId>(thumb_geom, cube_geom)};

  // ---- 3. Place the fingertips at the grasp points via IK ----
  const double cube_size = 0.06;
  const double a = -0.02, b = 0.02, c = 0.0;  // face offsets (cube_kinematics)
  const RigidTransform<double> X_WC(RotationMatrix<double>(),
                                    Eigen::Vector3d(0.0, 0.0, 0.58));

  // Cube at rest in front of the fingers.
  VectorXd q_cube(7);
  q_cube << 1, 0, 0, 0, X_WC.translation();
  plant.SetPositions(&plant_context, cube_index, q_cube);

  // Seed the thumb so IK does not start fully retracted.
  VectorXd q_seed = VectorXd::Zero(num_allegro_joints);
  q_seed[12] = 1.0;
  q_seed[13] = 0.5;
  q_seed[14] = 0.5;
  q_seed[15] = 0.3;
  plant.SetPositions(&plant_context, allegro_index, q_seed);

  // Fingertip frame targets; finger_inset controls contact depth (the tip
  // collision sphere adds to it). A shallow gap keeps the contact dynamics from
  // becoming overly stiff.
  const VectorXd contact_positions =
      GetGraspPositions(X_WC, cube_size - 2 * FLAGS_finger_inset, a, b, c);
  plant.SetPositions(&plant_context,
                     SolveGraspIK(plant, &plant_context, contact_positions));

  const VectorXd x0 = plant.GetPositionsAndVelocities(plant_context);

  // --- Diagnostics: did IK seat the fingertips, and is x0 finite? ---
  auto tip_pos = [&](const char* name) {
    return plant
        .EvalBodyPoseInWorld(plant_context,
                             plant.GetBodyByName(name, allegro_index))
        .translation();
  };
  std::cout << "IK fingertip vs target (should match within IK tol):\n";
  std::cout << "  index : " << tip_pos("link_3_tip").transpose()
            << "  target " << contact_positions.segment<3>(0).transpose()
            << "  err " << (tip_pos("link_3_tip") - contact_positions.segment<3>(0)).norm() << "\n";
  std::cout << "  middle: " << tip_pos("link_7_tip").transpose()
            << "  target " << contact_positions.segment<3>(3).transpose()
            << "  err " << (tip_pos("link_7_tip") - contact_positions.segment<3>(3)).norm() << "\n";
  std::cout << "  thumb : " << tip_pos("link_15_tip").transpose()
            << "  target " << contact_positions.segment<3>(6).transpose()
            << "  err " << (tip_pos("link_15_tip") - contact_positions.segment<3>(6)).norm() << "\n";
  std::cout << "x0 hasNaN: " << x0.hasNaN() << "\n";

  const int n_x = plant.num_positions() + plant.num_velocities();
  const int n_u = plant.num_actuators();

  // ---- 4. LCS options + linearize ----
  LCSFactoryOptions lcs_options;
  lcs_options.contact_model = "stewart_and_trinkle";
  lcs_options.N = FLAGS_N;
  lcs_options.dt = FLAGS_dt;
  lcs_options.num_contacts = 3;
  lcs_options.num_friction_directions = FLAGS_num_friction_directions;
  lcs_options.mu = FLAGS_mu;

  const int n_lambda = LCSFactory::GetNumContactVariables(
      GetContactModelMap().at(lcs_options.contact_model), 3,
      FLAGS_num_friction_directions);

  LCS lcs = LCSFactory::LinearizePlantToLCS(
      plant, plant_context, *plant_ad, *plant_context_ad, contact_pairs,
      lcs_options, x0, VectorXd::Zero(n_u));

  const int n_z = n_x + n_u + n_lambda;
  std::cout << "n_x=" << n_x << " n_u=" << n_u << " n_lambda=" << n_lambda
            << " n_z=" << n_z << "\n";

  // --- Diagnostics: is the linearized LCS finite and is contact coupled? ---
  auto any_nan = [](const std::vector<MatrixXd>& v) {
    for (const auto& m : v)
      if (m.hasNaN()) return true;
    return false;
  };
  std::cout << "LCS norms: |A0|=" << lcs.A()[0].norm()
            << " |B0|=" << lcs.B()[0].norm() << " |D0|=" << lcs.D()[0].norm()
            << " |E0|=" << lcs.E()[0].norm() << " |F0|=" << lcs.F()[0].norm()
            << "\n";
  std::cout << "LCS hasNaN: A=" << any_nan(lcs.A()) << " B=" << any_nan(lcs.B())
            << " D=" << any_nan(lcs.D()) << " E=" << any_nan(lcs.E())
            << " F=" << any_nan(lcs.F()) << " H=" << any_nan(lcs.H()) << "\n";

  // Contact gaps: eta = E x + F lambda + H u + c; at lambda=u=0 the entries that
  // pair with the normal forces (indices 3,4,5) are the signed gaps phi. If any
  // is > 0 the contact is OPEN, and complementarity forces lambda_n = 0 there
  // regardless of the force-tracking cost.
  const VectorXd eta0 = lcs.E()[0] * x0 + lcs.c()[0];
  std::cout << "contact gaps phi [index, middle, thumb] = "
            << eta0.segment<3>(3).transpose()
            << "   (<=0 means touching)\n";

  // ---- Input (torque) rescaling: u = s_u * u_tilde, so B_tilde = B * s_u.
  // Pick s_u = |A0|/|B0| to bring |B| down to ~|A| and make the QP well
  // conditioned (otherwise OSQP hits IterationLimit and falls back to zero).
  // This is a change of variables on u only; lambda is unchanged. R is scaled
  // by s_u^2 so the control penalty on the real torque is preserved.
  const double s_u = FLAGS_input_scale > 0.0
                         ? FLAGS_input_scale
                         : lcs.A()[0].norm() / lcs.B()[0].norm();
  std::vector<MatrixXd> B_scaled = lcs.B();
  for (auto& Bk : B_scaled) Bk *= s_u;
  lcs.set_B(B_scaled);
  std::cout << "input scale s_u=" << s_u << "  ->  |B0| now "
            << lcs.B()[0].norm() << "\n";

  // ---- 5. Cost matrices (Q anchors the cube; R/G/U standard) ----
  // Cube is the last (free) body: its 7 q are the last 7 positions, its 6 v
  // are the last 6 velocities (which sit at the end of the stacked state).
  const int cube_q_start = plant.num_positions() - 7;
  const int cube_v_start = plant.num_positions() + plant.num_velocities() - 6;

  MatrixXd Q_knot = FLAGS_w_state * MatrixXd::Identity(n_x, n_x);
  for (int i = cube_q_start; i < cube_q_start + 7; ++i) Q_knot(i, i) = FLAGS_w_cube;
  for (int i = cube_v_start; i < cube_v_start + 6; ++i) Q_knot(i, i) = FLAGS_w_cube;

  std::vector<MatrixXd> Q(FLAGS_N + 1, Q_knot);
  std::vector<MatrixXd> R(
      FLAGS_N, (s_u * s_u) * FLAGS_w_R * MatrixXd::Identity(n_u, n_u));
  std::vector<MatrixXd> G(FLAGS_N, FLAGS_w_G * MatrixXd::Identity(n_z, n_z));
  std::vector<MatrixXd> U(FLAGS_N, FLAGS_w_U * MatrixXd::Identity(n_z, n_z));

  // Hold the current state (cube stays where it is).
  std::vector<VectorXd> x_desired(FLAGS_N + 1, x0);

  // Match the proven sampling_c3 config. admm_iter must stay small: G is scaled
  // by rho_scale each ADMM iteration, so a large admm_iter explodes the QP
  // Hessian and OSQP reports the KKT as non-convex.
  C3Options c3_options;
  c3_options.admm_iter = 3;
  c3_options.rho_scale = 3;
  c3_options.warm_start = false;
  c3_options.scale_lcs = true;
  c3_options.gamma = 1.0;  // unused when cost matrices are supplied directly

  C3QP c3(lcs, C3::CostMatrices(Q, R, G, U), x_desired, c3_options);

  // ---- 6. Activate the alpha force-reference cost on lambda[3,4,5] ----
  // rho = (a - c)/(c - b) = 1 for these offsets => alpha_I = alpha_M,
  // alpha_T = alpha_I + alpha_M.
  const double alpha_I = FLAGS_alpha_m;
  const double alpha_M = FLAGS_alpha_m;
  const double alpha_T = alpha_I + alpha_M;

  std::vector<MatrixXd> W(FLAGS_N, MatrixXd::Zero(n_lambda, n_lambda));
  std::vector<VectorXd> lambda_des(FLAGS_N, VectorXd::Zero(n_lambda));
  for (int i = 0; i < FLAGS_N; ++i) {
    W[i](3, 3) = FLAGS_w_force;
    W[i](4, 4) = FLAGS_w_force;
    W[i](5, 5) = FLAGS_w_force;
    lambda_des[i](3) = alpha_I;
    lambda_des[i](4) = alpha_M;
    lambda_des[i](5) = alpha_T;
  }
  c3.SetForceTrackingWeight(W);
  c3.UpdateForceTarget(lambda_des);

  // Replicate C3's tuned default OSQP options, but loosen the convergence
  // tolerance and raise the iteration budget so the stiff hand QP converges
  // instead of hitting IterationLimit and falling back to zero. (My earlier
  // partial override dropped polishing/rho/alpha and caused SolverSpecificError.)
  SolverOptions osqp_opts;
  const auto osqp_id = OsqpSolver::id();
  osqp_opts.SetOption(osqp_id, "max_iter", 4000);
  osqp_opts.SetOption(osqp_id, "verbose", 0);
  osqp_opts.SetOption(osqp_id, "warm_starting", 1);
  osqp_opts.SetOption(osqp_id, "polishing", 1);
  osqp_opts.SetOption(osqp_id, "polish_refine_iter", 3);
  osqp_opts.SetOption(osqp_id, "scaled_termination", 1);
  osqp_opts.SetOption(osqp_id, "check_termination", 25);
  osqp_opts.SetOption(osqp_id, "scaling", 15);
  osqp_opts.SetOption(osqp_id, "adaptive_rho", 1);
  osqp_opts.SetOption(osqp_id, "rho", 1e-4);
  osqp_opts.SetOption(osqp_id, "sigma", 1e-6);
  osqp_opts.SetOption(osqp_id, "alpha", 1.6);
  osqp_opts.SetOption(osqp_id, "eps_abs", FLAGS_osqp_eps);
  osqp_opts.SetOption(osqp_id, "eps_rel", FLAGS_osqp_eps);
  c3.SetSolverOptions(osqp_opts);

  // ---- 7. Solve once and report ----
  c3.Solve(x0);
  const std::vector<VectorXd> lambda_sol = c3.GetForceSolution();

  std::cout << "\nTargets:  alpha_I=" << alpha_I << "  alpha_M=" << alpha_M
            << "  alpha_T=" << alpha_T << "  (N)\n";
  std::cout << "Planned normal forces lambda_n = [index, middle, thumb] per knot:\n";
  for (int i = 0; i < static_cast<int>(lambda_sol.size()); ++i) {
    std::cout << "  k=" << i << ":  " << lambda_sol[i].segment<3>(3).transpose()
              << "\n";
  }
  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) { return dairlib::DoMain(argc, argv); }
