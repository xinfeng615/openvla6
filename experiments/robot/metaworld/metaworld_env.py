import gymnasium as gym  # 修复 gym 弃用警告并解决 Numpy 兼容性
import numpy as np

import metaworld
from .mw_tools import setup_metaworld_env

class MetaworldEnv(gym.Env):
    def __init__(
        self,
        env_name: str,
        im_size: int = 256,
        seed: int = 42,
    ):
        self._env_name = env_name
        self._env = setup_metaworld_env(env_name + '-goal-observable')
        self._env._partially_observable = False
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        
        # 定义观测空间，对齐 OpenVLA 的输入格式
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((39,)) * -1, high=np.ones((39,)), dtype=np.float32
                ),
            }
        )
        
        # 定义动作空间
        self.action_space = gym.spaces.Box(
            low=np.ones((4,)) * -1, high=np.ones((4,)), dtype=np.float32
        )
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        state, reward, done, truncate, info = self._env.step(action)
        images = self._env.render() # 获取当前帧渲染图像
        
        info.update({"state": state})
        # 将原始环境输出打包成字典
        obs = {
            "image_primary": images,
            "proprio": np.asarray(state[:4], dtype=np.float32)
        }
        
        # 如果任务成功，标记 episode_is_success 并将 done 设为 True 结束当前回合
        if info['success']: 
            self._episode_is_success = 1
            done = True
        if self._env.env.curr_path_length == self._env.env.max_path_length:
            truncate = True
        
        return obs, reward, done, truncate, info
    
    def reset(self, **kwargs):
        state, info = self._env.reset(**kwargs)
        images = self._env.render()
        
        info.update({"state": state})
        obs = {
            "image_primary": images,
            "proprio": np.asarray(state[:4], dtype=np.float32)
        }
        
        return obs, info

    def get_task(self):
        # 提取语言指令（移除任务名称中的 -v2 等后缀，并用空格连接）
        return {
            "language_instruction": [" ".join(self._env_name.split('-')[:-1])],
        }

    def get_episode_metrics(self):
        # 获取回合指标，主要用于统计成功率
        return {
            "success_rate": self._episode_is_success,
        }

# 自动向 gym 注册所有 Meta-World 训练环境
benchmark = metaworld.MT50().train_classes
for name in benchmark.keys():
    gym.register(
        name,
        entry_point=lambda n=name: MetaworldEnv(n),
    )
