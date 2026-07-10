import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ========== 核心修复：彻底禁用 Google Cloud 联网检查 ==========
tfds.core.utils.gcs_utils._is_gcs_disabled = True

# ========== 类名显式加上下划线，强制 TFDS 保留 ==========
class MetaworldMl1_50e(tfds.core.GeneratorBasedBuilder):
    """
    ML1 Basketball 单任务数据集构建器
    将 HDF5 格式的专家轨迹数据转换为 RLDS TensorFlow Dataset 格式
    """
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        # 严格按照 OpenVLA RLDS 标准定义数据结构
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_primary': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
                        'state': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    'discount': tfds.features.Scalar(dtype=tf.float32),
                    'reward': tfds.features.Scalar(dtype=tf.float32),
                    'is_first': tfds.features.Scalar(dtype=tf.bool),
                    'is_last': tfds.features.Scalar(dtype=tf.bool),
                    'is_terminal': tfds.features.Scalar(dtype=tf.bool),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        # 告诉构建器去哪里读取我们采集的 HDF5 文件
        return {'train': self._generate_examples(path='/root/autodl-tmp/metaworld_ml1_hdf5')}

    def _generate_examples(self, path):
        """
        从 HDF5 文件生成 RLDS 格式的样本
        """
        episode_id = 0
        
        # 遍历 HDF5 文件目录
        for h5_file in os.listdir(path):
            if not h5_file.endswith('.hdf5'):
                continue
                
            file_path = os.path.join(path, h5_file)
            print(f"📂 处理文件: {file_path}")
            
            with h5py.File(file_path, 'r') as f:
                data_grp = f['data']
                
                for ep_key in data_grp.keys():
                    ep_grp = data_grp[ep_key]
                    images = ep_grp['image_primary'][:]
                    proprios = ep_grp['proprio'][:]
                    actions = ep_grp['action'][:]
                    instruction = ep_grp.attrs['language_instruction']
                    
                    episode_length = len(actions)
                    steps = []
                    
                    for i in range(episode_length):
                        # 【核心逻辑】：4维补零扩充至7维
                        # 状态：[x, y, z] + [0, 0, 0] + [gripper]
                        padded_state = np.concatenate((proprios[i][:3], np.zeros(3), proprios[i][3:4]))
                        # 动作：[x, y, z] + [0, 0, 0] + [gripper]
                        padded_action = np.concatenate((actions[i][:3], np.zeros(3), actions[i][3:4]))
                        
                        steps.append({
                            'observation': {
                                'image_primary': images[i],
                                'state': padded_state.astype(np.float32),
                            },
                            'action': padded_action.astype(np.float32),
                            'discount': 1.0,
                            'reward': float(i == episode_length - 1),
                            'is_first': (i == 0),
                            'is_last': (i == episode_length - 1),
                            'is_terminal': (i == episode_length - 1),
                            'language_instruction': instruction,
                        })
                    
                    yield str(episode_id), {
                        'steps': steps,
                        'episode_metadata': {'file_path': file_path}
                    }
                    episode_id += 1
        
        print(f"✅ 共处理 {episode_id} 条轨迹")


if __name__ == '__main__':
    # 手动实例化 Builder，注意这里调用的是新类名 MetaworldMl1_50e
    builder = MetaworldMl1_50e(data_dir='/root/autodl-tmp/tensorflow_datasets')
    
    print("=" * 60)
    print("🚀 开始构建 ML1 Basketball RLDS 数据集")
    print("=" * 60)
    
    # 启动转换与预处理过程
    builder.download_and_prepare()
    
    print("=" * 60)
    print("🎉 数据集构建完成！")
    print("   保存路径: /root/autodl-tmp/tensorflow_datasets/metaworld_ml1_50e")
    print("=" * 60)
