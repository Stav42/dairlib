#pragma once

#include <string>

namespace dairlib::allegro_grasp_c3 {

// Immutable snapshot of all command-line options. Runtime components receive
// this object explicitly and never reach into gflags global state.
struct SqueezeConfig {
  double kp;
  double kd;
  double tau_max;
  double contact_force_thresh;
  double penetration_index_middle;
  double penetration_thumb;
  double tip_surface_offset_z;
  double ring_tip_surface_offset_z;
  double ring_tip_surface_offset_y;
  bool track_cube_contact;
  std::string contact_ik_pose_source;
  bool contact_force_log;
  double contact_force_log_hz;
  bool lcm_publish;
  double lcm_publish_hz;
  bool isolate_cube;
  double contact_enable_t;
  double t_contact;
  double handoff_settle_time;
  double k_hold;
  double w_cube;
  double w_cube_vel;
  double w_vel;
  double w_R;
  double w_G;
  double w_U;
  double w_lambda;
  double alpha_m;
  std::string cube_motion_mode;
  double cube_move_dx;
  double cube_move_dy;
  double cube_move_dz;
  double cube_move_roll;
  double cube_move_pitch;
  double cube_move_yaw;
  double cube_move_period;
  double cube_move_duration;
  double cube_ik_lead_pos_max;
  double cube_ik_lead_rot_max;
  bool show_cube_target;
  bool show_cube_start;
  bool rot_log;
  double rot_log_period;
  bool lambda_map_debug;
  bool plan_debug;
  bool c3_joint_plan_log;
  bool osc_torque_split_log;
  double task_kp;
  double task_kd;
  std::string exec_mode;
  std::string osc_target_source;
  double osc_kp;
  double osc_kd;
  double osc_qd_filter_tau;
  bool exec_grav_comp;
  double force_floor;
  double lambda_torque_scale;
  bool osc_full_contact_force;
  // Optional measured-SAP-wrench outer loop around the full C3 force
  // executor.  The first implementation corrects world-Z yaw only.
  bool osc_wrench_feedback;
  double osc_wrench_feedback_period;
  double osc_wrench_feedback_yaw_kp;
  double osc_wrench_feedback_yaw_ki;
  double osc_wrench_feedback_max_yaw_moment;
  double osc_wrench_feedback_max_force_per_contact;
  double osc_wrench_feedback_force_rate_limit;
  double osc_wrench_feedback_allocation_damping;
  double osc_wrench_feedback_min_commanded_yaw_moment;
  double osc_wrench_feedback_min_resolved_yaw_moment;
  double osc_wrench_feedback_authority_timeout;
  bool osc_wrench_feedback_log;
  bool fk_target;
  int N;
  double c3_dt;
  double mu;
  int num_friction_directions;
  std::string contact_model;
  double input_scale;
  bool relinearize;
  bool warm_start_admm;
  int c3_period_steps;
  int relin_period_steps;
  int track_ik_period_steps;
  double osqp_eps;
  int admm_iter;
  double rho_scale;
  bool warm_start;
  bool release_middle;
  std::string release_finger;
  double release_middle_t;
  double release_middle_offset;
  double regrasp_settle_time;
  double regrasp_duration;
  double regrasp_touch_tol;
  double regrasp_arc_clearance;
  double release_middle_tri_spread;
  double release_middle_tri_base_z;
  double release_middle_tri_apex_z;
  bool gait;
  double gait_delta;
  int gait_cycles;
  double gait_rotate_duration;
  std::string gait_scheme;
  double spider_yaw_delta;
  double spider_yaw_duration;
  // Diagnostic mode: perform the isolated spider yaw and retain the
  // established index-middle-thumb C3 hold without starting ring placement.
  bool spider_yaw_only;
  double spider_ring_placement_duration;
  // Nominal 60 mm-cube coordinates for the three-finger triangle on yellow.
  // They scale with cube_size_scale before being used as contact points.
  double spider_triangle_half_width;
  double spider_triangle_base_z;
  double spider_triangle_apex_z;
  double spider_ring_red_y;
  double spider_ring_hold_z;
  double spider_index_crawl;
  double spider_index_arc_clearance;
  double spider_index_duration;
  // Keep the current maneuver at a four-contact hold after the selected ring
  // placement unless an explicit later experiment enables the index crawl.
  bool spider_index_crawl_after_ring;
  double spider_support_normal_margin;
  double spider_support_verify_time;
  // Before changing the C3 contact topology, keep the four-contact grasp
  // until the measured cube pose and velocity have remained within these
  // settling bounds for spider_support_settle_time.
  double spider_support_settle_time;
  double spider_support_settle_yaw_error;
  double spider_support_settle_translation_error;
  double spider_support_settle_linear_speed;
  double spider_support_settle_angular_speed;
  // Before adding ring to C3, require the actual SAP fingertip contact
  // force on the cube to support its weight within this allowed deficit.
  // This is a measured-force gate, not a C3 normal-force constraint.
  double spider_ring_handoff_vertical_force_deficit;
  // Maximum total three-dimensional position error accepted by the gate.
  double spider_support_max_translation_error;
  double spider_support_max_orientation_error;
  double spider_support_max_linear_speed;
  double spider_support_max_angular_speed;
  // Diagnostic-only override: retain the solved-force and normal-margin
  // checks, but permit the index lift despite a predicted-motion violation.
  bool spider_allow_unverified_index_lift;
  // Diagnostic-only virtual placement sweep.  After the small yaw, retain
  // the real index-middle-thumb hold, leave ring parked, and solve temporary
  // middle-thumb-ring C3 problems on this grid of hypothetical red-face
  // ring points.  It never commands any of those points.
  bool spider_virtual_ring_search;
  int spider_ring_search_rows;
  int spider_ring_search_cols;
  double spider_ring_search_face_margin;
  double relay_ring_hold_z;
  double relay_ring_press;
  double relay_ring_retract;
  bool gait_realign;
  bool gait_log;
  bool gait_handoff_debug;
  double gait_log_period;
  bool legacy_log;
  int gait_seek_period_steps;
  double gait_leg_gap;
  double gait_touch_offset;
  double gait_touch_duration;
  int gait_rebuild_solve_passes;
  double gait_force_ramp_time;
  double gait_torque_ramp_time;
  double gait_leg_timeout;
  double gait_hold_time;
  double cube_start_x;
  double cube_start_y;
  double cube_start_z;
  // Uniform geometric scale relative to the nominal 60 mm cube.  The
  // currently supplied models support 1.0, 0.8, and 0.6.
  double cube_size_scale;
  bool preview;
  double sim_time;
};

SqueezeConfig LoadSqueezeConfig();

}  // namespace dairlib::allegro_grasp_c3
