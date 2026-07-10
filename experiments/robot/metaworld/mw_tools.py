import numpy as np
from PIL import Image
import gymnasium
from gymnasium.core import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import metaworld
# 删除了 from metaworld.policies import *，因为评估不需要生成专家轨迹

# import os
# os.environ["MUJOCO_GL"] = "osmesa" # 设置 MuJoCo 渲染后端

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,  # 绕向上(up)向量旋转相机
    "elevation": -25.0,  # 绕向右(right)向量旋转相机
    "lookat": np.array([0.0, 0.65, 0.0]), # 相机注视的中心点坐标
    }

DEFAULT_SIZE=256
# EPISODE_LENGTH = 500 * 100 

# ⚠️ 已删除庞大且无用的 POLICIES 字典，避免 V2/V3 策略名称冲突

class CameraWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, seed: int):
        super().__init__(env)

        # 设置 MuJoCo 离线渲染器的宽高
        self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
        # 初始化带自定义视角的 MuJoCo 渲染器
        self.unwrapped.mujoco_renderer = MujocoRenderer(env.model, env.data, DEFAULT_CAMERA_CONFIG, DEFAULT_SIZE, DEFAULT_SIZE)

        # 技巧：启用随机重置
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)

    def reset(self):
        obs, info = super().reset()
        return obs, info

    def step(self, action):
        next_obs, reward, done, truncate, info = self.env.step(action)
        return next_obs, reward, done, truncate, info

def setup_metaworld_env(task_name: str, seed: int = 42):
    # 移除旧版后缀，适配新版 API
    base_name = task_name.replace('-goal-observable', '')
    mt50 = metaworld.MT50()
    env_cls = mt50.train_classes[base_name]
    
    # 1. 实例化基础的无包装环境
    unwrapped_env = env_cls(render_mode="rgb_array")
    
    # 2. 【核心修复】从 MT50 任务库中筛选出当前环境对应的具体任务配置，并设置
    # 这会初始化物体生成位置、目标点坐标等，从而满足 env.step 的前置要求
    tasks = [task for task in mt50.train_tasks if task.env_name == base_name]
    if tasks:
        unwrapped_env.set_task(tasks[0]) # 默认使用该环境的第一个任务变体
        
    # 3. 加上相机包装器
    env = CameraWrapper(unwrapped_env, seed)
    return env
