import tensorflow as tf


def senet(input, embedding_dim, reduction_ratio):
    """
        SENET实现
    Args:
        input (Tensor): 输入, shape=(batch, F, K)
        embedding_dim (int): embedding维度K
        reduction_ratio (int): 缩减比率

    Returns:
        tf.Tensor, shape=(batch, F, K)
    """

    # 维度信息
    F = input.get_shape()[1]
    reduction_dim = embedding_dim // reduction_ratio
    assert reduction_dim < embedding_dim, "reduction_dim must be less than embedding_dim"

    # 定义模型参数
    w1 = tf.get_variable(name="senet_w1", shape=(F, reduction_dim), dtype=tf.float32)
    w2 = tf.get_variable(name="senet_w2", shape=(reduction_dim, F), dtype=tf.float32)

    # 计算权重向量a
    z = tf.reduce_mean(input, axis=-1)  # (batch, F)
    a = tf.matmul(z, w1)    # (batch, reduction_dim)
    a = tf.nn.relu(a)   # (batch, reduction_dim)
    a = tf.matmul(a, w2)  # (batch, F)
    a = tf.nn.relu(a)  # (batch, F)
    a = tf.expand_dims(a, axis=-1)  # (batch, F, 1)

    # reweight
    v = input * a   # (batch, F, K) * (batch, F, 1) = (batch, F, K)

    return v
