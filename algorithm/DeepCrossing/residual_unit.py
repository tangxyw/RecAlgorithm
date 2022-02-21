import tensorflow as tf


def residual_unit(input, internal_dim=None, index=None):
    """
        deepcrossing模型的残差网络部分
    Args:
        input (tensor): 输入tensor
        internal_dim (int): 网络内部维度
        index (int): 序号

    Returns:
        tensor, 维度与输入tensor一致
    """

    output_dim = int(input.get_shape()[-1])
    x = input
    x = tf.layers.dense(x, internal_dim, name=f"dense_{index}_0", activation=tf.nn.relu)
    x = tf.layers.dense(x, output_dim, name=f"dense_{index}_1", activation=None)
    output = tf.nn.relu(input + x)

    return output