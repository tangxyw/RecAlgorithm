import tensorflow as tf
import itertools


def bilinear_interaction_layer(input, embedding_dim, type, name):
    """
        FiBinet模型中的Bilinear Interaction Layer实现
    Args:
        input (Tensor): 输入, shape=(batch, F, K)
        embedding_dim (int): embedding维度K
        type (str): Bilinear interaction类型, "all", "each", 或者 "interaction"
        name (str): 用来区分不同的Bilinear Interaction Layer

    Returns:
        tf.Tensor, shape=(batch, F*(F-1)/2, K)
    """

    # 维度信息
    F = input.get_shape()[1]

    if type == "all":
        w = tf.get_variable(name=f"{name}_w_all", shape=(embedding_dim, embedding_dim), dtype=tf.float32)   # (K, K)
        v_w = tf.matmul(input, w)   # (batch, F, K) @ (K, K) = (batch, F, K)
        p = [tf.multiply(v_w[:, i, :], input[:, j, :]) for i, j in itertools.combinations(range(F-1), 2)]   # (batch, K) * F*(F-1)/2

    elif type == "each":
        w = tf.get_variable(name=f"{name}_w_each", shape=(F-1, embedding_dim, embedding_dim), dtype=tf.float32)  # (F-1, K, K)
        v_w = [tf.matmul(input[:, i, :], w[i, :, :]) for i in range(F-1)]   # (batch, K) * F-1
        p = [tf.multiply(v_w[i], input[:, j, :]) for i, j in itertools.combinations(range(F-1), 2)] # (batch, K) * F*(F-1)/2

    elif type == "interaction":
        w = tf.get_variable(name=f"{name}_w_interaction", shape=(F*(F-1)//2, embedding_dim, embedding_dim), dtype=tf.float32)    # (F(F-1)/2, K, K)
        p = [tf.multiply(tf.matmul(input[:, i, :], w[k, :, :]), input[:, j, :])
             for (i, j), k in zip(itertools.combinations(range(F-1), 2), range(F*(F-1)//2))]   # (batch, K) * F*(F-1)/2

    else:
        raise ValueError(
                f"Bilinear Interaction type must be in ['all','each','interaction'], got '{type}'")

    p = tf.stack(p, axis=1)  # (batch, F*F(F-1)/2, K)

    return p
