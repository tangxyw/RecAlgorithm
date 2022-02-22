import tensorflow as tf


def cross_layer(x0, xl, index):
    """
        dcn模型的cross layer
    Args:
        x0 (tensor): cross layer最原始输入
        xl (tensor): cross layer上一层输出
        index (int): cross layer序号

    Returns:
        tensor, 维度与x0一致
    """

    dimension = int(x0.get_shape()[-1])
    # wl，bl为cross_layer参数
    wl = tf.get_variable(name=f"wl_{index}", shape=(dimension, 1), dtype=tf.float32)  # (d, 1)
    bl = tf.get_variable(name=f"bl_{index}", shape=(dimension, 1), dtype=tf.float32)  # (d, 1)

    xl_wl = tf.matmul(xl, wl)  # (batch, d) * (d, 1) = (batch, 1)
    x0_xl_wl = tf.multiply(x0, xl_wl)   # (batch, d) multiply (batch, 1) = (batch, d)
    output = tf.add(x0_xl_wl, tf.transpose(bl))
    output = tf.add(output, xl)

    return output
