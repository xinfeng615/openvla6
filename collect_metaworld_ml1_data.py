import os
import numpy as np
import metaworld

# 设置 MuJoCo 使用 EGL 渲染（无头服务器环境必需）
os.environ["MUJOCO_GL"] = "egl"

import h5py
import metaworld.policies as policies
from experiments.robot.metaworld.metaworld_env import MetaworldEnv

# ML1 单任务: Basketball 专家策略
POLICY_MAP = {
    'basketball-v3': 'SawyerBasketballV3Policy',
}

def collect_ml1_data(episodes_per_task=50, max_steps=500):
    """
    收集 ML1 (MetaWorld-1) Basketball 单任务数据

    Args:
        episodes_per_task: 每个任务采集的成功轨迹数量
        max_steps: 每回合最大步数
    """
    # 将生成的 HDF5 文件保存在数据盘
    save_dir = "/root/autodl-tmp/metaworld_ml1_hdf5"
    os.makedirs(save_dir, exist_ok=True)

    # ML1: Basketball 单任务
    task_name = "basketball-v3"
    print(f"\n🚀 开始采集任务: {task_name}")

    # 创建环境
    env = MetaworldEnv(task_name)

    # 获取对应的专家策略
    policy_class_name = POLICY_MAP.get(task_name)
    if not policy_class_name:
        print(f"❌ 字典中未定义 {task_name} 的策略，退出...")
        return

    try:
        policy_cls = getattr(policies, policy_class_name)
        policy = policy_cls()
    except AttributeError:
        print(f"❌ 找不到策略 {policy_class_name}，退出")
        return

    # 获取语言指令
    task_instruction = env.get_task()["language_instruction"][0]

    # 保存为 HDF5 格式
    h5_path = os.path.join(save_dir, f"{task_name}.hdf5")
    with h5py.File(h5_path, "w") as f:
        data_grp = f.create_group("data")

        success_count = 0
        total_attempts = 0
        max_attempts = episodes_per_task * 10  # 防止无限循环

        while success_count < episodes_per_task and total_attempts < max_attempts:
            total_attempts += 1

            obs_dict, info = env.reset()
            images, proprios, actions = [], [], []
            episode_success = False

            for step in range(max_steps):
                # 1. 记录相机图像和本体状态
                images.append(obs_dict["image_primary"])
                proprios.append(obs_dict["proprio"])

                # 2. 从环境的 info 字典中提取专家需要的原始 39 维物理状态，并获取完美动作
                raw_state = info["state"]
                action = policy.get_action(raw_state)
                actions.append(action)

                # 3. 步进环境
                obs_dict, reward, done, trunc, info = env.step(action)

                if info.get('success', False):
                    episode_success = True

                if done or trunc:
                    break

            # 只有当任务真正成功时，才将这段轨迹写入文件
            if episode_success:
                ep_grp = data_grp.create_group(f"demo_{success_count}")
                ep_grp.create_dataset("image_primary", data=np.array(images, dtype=np.uint8))
                ep_grp.create_dataset("proprio", data=np.array(proprios, dtype=np.float32))
                ep_grp.create_dataset("action", data=np.array(actions, dtype=np.float32))
                ep_grp.attrs["language_instruction"] = task_instruction

                success_count += 1
                print(f"  ✅ 成功收集轨迹: {success_count}/{episodes_per_task} (耗时步数: {len(actions)})")
            else:
                # 专家策略在某些随机初始化下也会失误，失败的直接丢弃
                if total_attempts % 10 == 0:
                    print(f"  🔄 专家策略失误，已尝试 {total_attempts} 次，成功 {success_count} 次...")

        print(f"\n📊 采集统计: 共尝试 {total_attempts} 次，成功 {success_count} 次")

    print(f"🎉 任务 {task_name} 数据已保存至 {h5_path}")

if __name__ == "__main__":
    collect_ml1_data(episodes_per_task=50, max_steps=500)
