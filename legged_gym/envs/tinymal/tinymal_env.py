"""
Tinymal机器人环境类
继承自基础LeggedRobot类，提供tinymal特定的环境实现
"""

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.tinymal.tinymal_config import TinymalRoughCfg


class TinymalRobot(LeggedRobot):
    """
    Tinymal四足机器人环境类

    这个类继承自基础的LeggedRobot类，提供了tinymal机器人的特定实现。
    如果需要添加tinymal特有的功能（如特殊的奖励函数、观测计算等），
    可以在这个类中重写相应的方法。
    """

    def __init__(self, cfg: TinymalRoughCfg, sim_params, physics_engine, sim_device, headless):
        """
        初始化Tinymal机器人环境

        Args:
            cfg: Tinymal配置对象
            sim_params: 仿真参数
            physics_engine: 物理引擎类型
            sim_device: 仿真设备 (cuda/cpu)
            headless: 是否无头模式 (不显示图形界面)
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # 如果需要添加tinymal特定的功能，可以重写以下方法：
    # def _post_physics_step_callback(self):
    #     """在物理步进后调用的回调函数"""
    #     pass

    # def _compute_reward(self):
    #     """计算自定义奖励函数"""
    #     pass

    # def _compute_obs(self):
    #     """计算自定义观测"""
    #     pass

    # def reset_idx(self, env_ids):
    #     """重置指定环境"""
    #     pass