#include "allegro_grasp_c3_squeeze_config.h"

#include "allegro_grasp_c3_squeeze_flags.h"

namespace dairlib::allegro_grasp_c3 {

SqueezeConfig LoadSqueezeConfig() {
  SqueezeConfig result;
  result.kp = FLAGS_kp;
  result.kd = FLAGS_kd;
  result.tau_max = FLAGS_tau_max;
  result.contact_force_thresh = FLAGS_contact_force_thresh;
  result.penetration_index_middle = FLAGS_penetration_index_middle;
  result.penetration_thumb = FLAGS_penetration_thumb;
  result.tip_surface_offset_z = FLAGS_tip_surface_offset_z;
  result.ring_tip_surface_offset_z = FLAGS_ring_tip_surface_offset_z;
  result.ring_tip_surface_offset_y = FLAGS_ring_tip_surface_offset_y;
  result.track_cube_contact = FLAGS_track_cube_contact;
  result.contact_ik_pose_source = FLAGS_contact_ik_pose_source;
  result.contact_force_log = FLAGS_contact_force_log;
  result.contact_force_log_hz = FLAGS_contact_force_log_hz;
  result.lcm_publish = FLAGS_lcm_publish;
  result.lcm_publish_hz = FLAGS_lcm_publish_hz;
  result.isolate_cube = FLAGS_isolate_cube;
  result.contact_enable_t = FLAGS_contact_enable_t;
  result.t_contact = FLAGS_t_contact;
  result.handoff_settle_time = FLAGS_handoff_settle_time;
  result.k_hold = FLAGS_k_hold;
  result.w_cube = FLAGS_w_cube;
  result.w_cube_vel = FLAGS_w_cube_vel;
  result.w_vel = FLAGS_w_vel;
  result.w_R = FLAGS_w_R;
  result.w_G = FLAGS_w_G;
  result.w_U = FLAGS_w_U;
  result.w_lambda = FLAGS_w_lambda;
  result.alpha_m = FLAGS_alpha_m;
  result.cube_motion_mode = FLAGS_cube_motion_mode;
  result.cube_move_dx = FLAGS_cube_move_dx;
  result.cube_move_dy = FLAGS_cube_move_dy;
  result.cube_move_dz = FLAGS_cube_move_dz;
  result.cube_move_roll = FLAGS_cube_move_roll;
  result.cube_move_pitch = FLAGS_cube_move_pitch;
  result.cube_move_yaw = FLAGS_cube_move_yaw;
  result.cube_move_period = FLAGS_cube_move_period;
  result.cube_move_duration = FLAGS_cube_move_duration;
  result.cube_ik_lead_pos_max = FLAGS_cube_ik_lead_pos_max;
  result.cube_ik_lead_rot_max = FLAGS_cube_ik_lead_rot_max;
  result.show_cube_target = FLAGS_show_cube_target;
  result.show_cube_start = FLAGS_show_cube_start;
  result.rot_log = FLAGS_rot_log;
  result.rot_log_period = FLAGS_rot_log_period;
  result.lambda_map_debug = FLAGS_lambda_map_debug;
  result.plan_debug = FLAGS_plan_debug;
  result.c3_joint_plan_log = FLAGS_c3_joint_plan_log;
  result.osc_torque_split_log = FLAGS_osc_torque_split_log;
  result.task_kp = FLAGS_task_kp;
  result.task_kd = FLAGS_task_kd;
  result.exec_mode = FLAGS_exec_mode;
  result.osc_target_source = FLAGS_osc_target_source;
  result.osc_kp = FLAGS_osc_kp;
  result.osc_kd = FLAGS_osc_kd;
  result.osc_qd_filter_tau = FLAGS_osc_qd_filter_tau;
  result.exec_grav_comp = FLAGS_exec_grav_comp;
  result.force_floor = FLAGS_force_floor;
  result.lambda_torque_scale = FLAGS_lambda_torque_scale;
  result.osc_full_contact_force = FLAGS_osc_full_contact_force;
  result.osc_wrench_feedback = FLAGS_osc_wrench_feedback;
  result.osc_wrench_feedback_period = FLAGS_osc_wrench_feedback_period;
  result.osc_wrench_feedback_yaw_kp = FLAGS_osc_wrench_feedback_yaw_kp;
  result.osc_wrench_feedback_yaw_ki = FLAGS_osc_wrench_feedback_yaw_ki;
  result.osc_wrench_feedback_max_yaw_moment =
      FLAGS_osc_wrench_feedback_max_yaw_moment;
  result.osc_wrench_feedback_max_force_per_contact =
      FLAGS_osc_wrench_feedback_max_force_per_contact;
  result.osc_wrench_feedback_force_rate_limit =
      FLAGS_osc_wrench_feedback_force_rate_limit;
  result.osc_wrench_feedback_allocation_damping =
      FLAGS_osc_wrench_feedback_allocation_damping;
  result.osc_wrench_feedback_min_commanded_yaw_moment =
      FLAGS_osc_wrench_feedback_min_commanded_yaw_moment;
  result.osc_wrench_feedback_min_resolved_yaw_moment =
      FLAGS_osc_wrench_feedback_min_resolved_yaw_moment;
  result.osc_wrench_feedback_authority_timeout =
      FLAGS_osc_wrench_feedback_authority_timeout;
  result.osc_wrench_feedback_log = FLAGS_osc_wrench_feedback_log;
  result.fk_target = FLAGS_fk_target;
  result.N = FLAGS_N;
  result.c3_dt = FLAGS_c3_dt;
  result.mu = FLAGS_mu;
  result.num_friction_directions = FLAGS_num_friction_directions;
  result.contact_model = FLAGS_contact_model;
  result.input_scale = FLAGS_input_scale;
  result.relinearize = FLAGS_relinearize;
  result.warm_start_admm = FLAGS_warm_start_admm;
  result.c3_period_steps = FLAGS_c3_period_steps;
  result.relin_period_steps = FLAGS_relin_period_steps;
  result.track_ik_period_steps = FLAGS_track_ik_period_steps;
  result.osqp_eps = FLAGS_osqp_eps;
  result.admm_iter = FLAGS_admm_iter;
  result.rho_scale = FLAGS_rho_scale;
  result.warm_start = FLAGS_warm_start;
  result.release_middle = FLAGS_release_middle;
  result.release_finger = FLAGS_release_finger;
  result.release_middle_t = FLAGS_release_middle_t;
  result.release_middle_offset = FLAGS_release_middle_offset;
  result.regrasp_settle_time = FLAGS_regrasp_settle_time;
  result.regrasp_duration = FLAGS_regrasp_duration;
  result.regrasp_touch_tol = FLAGS_regrasp_touch_tol;
  result.regrasp_arc_clearance = FLAGS_regrasp_arc_clearance;
  result.release_middle_tri_spread = FLAGS_release_middle_tri_spread;
  result.release_middle_tri_base_z = FLAGS_release_middle_tri_base_z;
  result.release_middle_tri_apex_z = FLAGS_release_middle_tri_apex_z;
  result.gait = FLAGS_gait;
  result.gait_delta = FLAGS_gait_delta;
  result.gait_cycles = FLAGS_gait_cycles;
  result.gait_rotate_duration = FLAGS_gait_rotate_duration;
  result.gait_scheme = FLAGS_gait_scheme;
  result.spider_yaw_delta = FLAGS_spider_yaw_delta;
  result.spider_yaw_duration = FLAGS_spider_yaw_duration;
  result.spider_yaw_only = FLAGS_spider_yaw_only;
  result.spider_ring_placement_duration =
      FLAGS_spider_ring_placement_duration;
  result.spider_triangle_half_width = FLAGS_spider_triangle_half_width;
  result.spider_triangle_base_z = FLAGS_spider_triangle_base_z;
  result.spider_triangle_apex_z = FLAGS_spider_triangle_apex_z;
  result.spider_ring_red_y = FLAGS_spider_ring_red_y;
  result.spider_ring_hold_z = FLAGS_spider_ring_hold_z;
  result.spider_index_crawl = FLAGS_spider_index_crawl;
  result.spider_index_arc_clearance = FLAGS_spider_index_arc_clearance;
  result.spider_index_duration = FLAGS_spider_index_duration;
  result.spider_index_crawl_after_ring =
      FLAGS_spider_index_crawl_after_ring;
  result.spider_support_normal_margin = FLAGS_spider_support_normal_margin;
  result.spider_support_verify_time = FLAGS_spider_support_verify_time;
  result.spider_support_settle_time = FLAGS_spider_support_settle_time;
  result.spider_support_settle_yaw_error =
      FLAGS_spider_support_settle_yaw_error;
  result.spider_support_settle_translation_error =
      FLAGS_spider_support_settle_translation_error;
  result.spider_support_settle_linear_speed =
      FLAGS_spider_support_settle_linear_speed;
  result.spider_support_settle_angular_speed =
      FLAGS_spider_support_settle_angular_speed;
  result.spider_ring_handoff_vertical_force_deficit =
      FLAGS_spider_ring_handoff_vertical_force_deficit;
  result.spider_support_max_translation_error =
      FLAGS_spider_support_max_translation_error;
  result.spider_support_max_orientation_error =
      FLAGS_spider_support_max_orientation_error;
  result.spider_support_max_linear_speed =
      FLAGS_spider_support_max_linear_speed;
  result.spider_support_max_angular_speed =
      FLAGS_spider_support_max_angular_speed;
  result.spider_allow_unverified_index_lift =
      FLAGS_spider_allow_unverified_index_lift;
  result.spider_virtual_ring_search = FLAGS_spider_virtual_ring_search;
  result.spider_ring_search_rows = FLAGS_spider_ring_search_rows;
  result.spider_ring_search_cols = FLAGS_spider_ring_search_cols;
  result.spider_ring_search_face_margin =
      FLAGS_spider_ring_search_face_margin;
  result.relay_ring_hold_z = FLAGS_relay_ring_hold_z;
  result.relay_ring_press = FLAGS_relay_ring_press;
  result.relay_ring_retract = FLAGS_relay_ring_retract;
  result.gait_realign = FLAGS_gait_realign;
  result.gait_log = FLAGS_gait_log;
  result.gait_handoff_debug = FLAGS_gait_handoff_debug;
  result.gait_log_period = FLAGS_gait_log_period;
  result.legacy_log = FLAGS_legacy_log;
  result.gait_seek_period_steps = FLAGS_gait_seek_period_steps;
  result.gait_leg_gap = FLAGS_gait_leg_gap;
  result.gait_touch_offset = FLAGS_gait_touch_offset;
  result.gait_touch_duration = FLAGS_gait_touch_duration;
  result.gait_rebuild_solve_passes = FLAGS_gait_rebuild_solve_passes;
  result.gait_force_ramp_time = FLAGS_gait_force_ramp_time;
  result.gait_torque_ramp_time = FLAGS_gait_torque_ramp_time;
  result.gait_leg_timeout = FLAGS_gait_leg_timeout;
  result.gait_hold_time = FLAGS_gait_hold_time;
  result.cube_start_x = FLAGS_cube_start_x;
  result.cube_start_y = FLAGS_cube_start_y;
  result.cube_start_z = FLAGS_cube_start_z;
  result.cube_size_scale = FLAGS_cube_size_scale;
  result.preview = FLAGS_preview;
  result.sim_time = FLAGS_sim_time;
  return result;
}

}  // namespace dairlib::allegro_grasp_c3
