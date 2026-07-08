"""
Tinymal机器人环境模块

这个模块包含了tinymal四足机器人的配置和环境类。
"""

from legged_gym.envs.tinymal.tinymal_config import TinymalRoughCfg, TinymalRoughCfgPPO
from legged_gym.envs.tinymal.tinymal_env import TinymalRobot

__all__ = ['TinymalRoughCfg', 'TinymalRoughCfgPPO', 'TinymalRobot']