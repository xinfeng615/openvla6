import tensorflow as tf


def resize_image(img, resize_size):
    """
    接收对应于单张图像的 numpy 数组，并返回调整大小后的图像 numpy 数组。

    注意（Moo Jin）：为了使输入图像与训练时看到的输入分布一致，我们遵循 Octo 数据加载器中使用的相同调整大小方案，
                    OpenVLA 在训练中使用该方案。
    """
    assert isinstance(resize_size, tuple)
    # 调整图像大小至模型期望的尺寸
    img = tf.image.encode_jpeg(img)  # 编码为 JPEG，与 RLDS 数据集构建器中的操作一致
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # 立即解码回来
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img
