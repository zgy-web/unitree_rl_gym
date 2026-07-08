from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# ======================================================================================
# tinymal 平地 Trot 训练配置
# --------------------------------------------------------------------------------------
# 由 TINYMAL_Trot_config.py 移植，仅保留原版 LeggedRobot / rsl_rl 能实际生效的参数；
# 依赖自定义 env / 自定义 rsl_rl 的项（frame_stack、特权 obs 堆叠、对称性损失 sym_loss、
# trot/gallop/energy_consumption/feet_clearance 等步态奖励、link_mass/com/pd_gains/latency
# 等高级 domain_rand）已剔除——它们在原版框架下不会被读取或会触发 AttributeError。
#
# tinymal 物理量(URDF)：整机 5.664 kg，base 2.266 kg，腿长(thigh0.1196+calf0.1513)=0.271 m，
# 全关节 effort=12 N·m，velocity=20 rad/s。
# 关键差异：calf 限位 [0,2.7] 为正区间（go2 为负区间 [-2.72,-0.84]），故默认 calf 角取 +1.5。
#
# 用户硬约束：平地(plane)、固定速度范围 [-1,1] m/s、不要 gallop。
# ======================================================================================
class TinymalRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 原版 LeggedRobot.compute_observations 固定返回 48 维：
        # base_lin_vel(3)+base_ang_vel(3)+projected_gravity(3)+commands(3)+dof_pos(12)+dof_vel(12)+actions(12)
        # Trot 配置的 frame_stack*47=470 需自定义 env，原版不支持，故保持 48。
        num_observations = 48
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 24      # Trot 原值
        env_spacing = 3.
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'        # 【用户约束】平地训练
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True          # plane 模式下地形课程不生效，保留无碍
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False    # 平地不需要高度图（原版 obs 不含高度，此项仅影响随机化等）
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # proportions 语义（cumsum 后按 choice 区间划分地形类型）：
        #   [0]=平地(terrain.py 已改为生成平地), [1]=粗糙坡, [2]/[3]=楼梯(下/上), [4]=离散障碍
        # [0.5, 0.0, 0.25, 0.25, 0.0] → 50% 平地 + 25% 下楼梯 + 25% 上楼梯，无坡无障碍
        terrain_proportions = [0.5, 0.0, 0.25, 0.25, 0.0]
        slope_treshold = 0.75

    class commands(LeggedRobotCfg.commands):
        curriculum = False         # 【用户约束】关闭速度课程
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 5.
        heading_command = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]    # 【用户约束】最高 1 m/s
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        # pos_z=0.28 略高于站立高度 0.20，给策略一个落地缓冲学习窗口
        pos = [0.0, 0.0, 0.28]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        # 【默认站立姿态】tinymal calf 限位 [0,2.7] 为正区间（与 go2 负区间相反），
        # 故 calf=+1.5（go2 用 -1.5）。thigh=1.0、hip=0.0，前后对称、四脚同高 ≈0.195m。
        # 限位核验：hip0∈[-0.6,0.6], thigh1.0∈[-1.57,1.57], calf1.5∈[0,2.7]，全部在硬限位内。
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,

            'FL_thigh_joint': 1.0,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 1.0,
            'RR_thigh_joint': 1.0,

            'FL_calf_joint': 1.5,
            'RL_calf_joint': 1.5,
            'FR_calf_joint': 1.5,
            'RR_calf_joint': 1.5,
        }

    class control(LeggedRobotCfg.control):
        # PD 参数：stiffness=12 满足 stiffness≈effort 缩放律（go2 25≈effort23.7；tinymal 12=effort12）。
        # damping per-joint：hip 承担整条腿反射惯量最大、calf 远端最小，按关节取不同阻尼更稳。
        # key 用 '_hip'/'_thigh'/'_calf'（带下划线）：env 用 'if key in dof_name' 子串匹配，
        # 'hip' 是 'thigh' 的子串会误匹配，加下划线消除歧义。
        control_type = 'P'
        stiffness = {'_hip': 12., '_thigh': 12., '_calf': 12.}
        damping = {'_hip': 1.0, '_thigh': 0.8, '_calf': 0.3}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tinymal/urdf/tinymal.urdf'
        name = "tinymal"
        foot_name = "foot"                # FL/FR/RL/RR_foot
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = False     # 保留 foot 独立 body，feet_indices 才能找到
        fix_base_link = False
        default_dof_drive_mode = 3        # effort 模式，env 用 PD 算力矩后写入
        self_collisions = 0               # 0: 启用碰撞过滤
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False    # STL 已是 Z-up，翻转会破坏装配（与 h1/h1_2/g1 一致）
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01                   # 小电机反射惯量占比大，提升仿真保真、缩 sim2real gap
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 仅保留原版 LeggedRobot 实际读取的随机化项；
        # Trot 里的 link_mass/base_com/pd_gains/motor_offset/latency 等需自定义 env，已剔除。
        randomize_friction = True
        friction_range = [0.2, 1.2]

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4

        # 按整机质量比例缩放(5.664/15.019≈0.377)。原版字段名为 added_mass_range。
        randomize_base_mass = True
        added_mass_range = [-0.4, 0.8]

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            # 主要奖励（轻小机器人加强跟踪与姿态稳定）
            tracking_lin_vel = 3.5
            tracking_ang_vel = 1.5
            # 轻机器人转动惯量小，同扰动影响更大，加强姿态稳定惩罚
            lin_vel_z = -1.5
            ang_vel_xy = -0.4
            orientation = -2.
            # 质量轻 + 弱电机，净力矩余量适中，惩罚略放松鼓励用矩
            torques = -0.0003
            dof_vel = -1e-6
            dof_acc = -2.5e-7
            collision = -0.5               # 腿更短更细，碰撞危害小，放松
            action_rate = -0.02
            stand_still = -1.0
            base_height = -1.
            # 鼓励抬脚形成 trot 步态（原版支持；Trot 配置用自定义 trot 奖励替代，此处用 feet_air_time 补位）
            feet_air_time = 1.0
            # 删除项（原版无对应 _reward_ 函数，配置会报错）：
            #   trot / gallop / energy_consumption / feet_clearance /
            #   default_hip_pos / default_pos / contact_without_command / alive

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.20         # = FK 站立高度 0.195，惩罚设在自然站立点
        max_contact_force = 40.           # 按质量缩放：100×0.377≈38
        # cycle_time / target_foot_height 仅供自定义步态奖励使用，原版不读，已剔除

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class TinymalRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3.e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        # sym_loss / obs_permutation / act_permutation / frame_stack / sym_coef
        # 需自定义 rsl_rl 才生效，原版不读，已剔除

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        num_steps_per_env = 24
        max_iterations = 2000
        save_interval = 100
        experiment_name = 'tinymal_trot'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
