import tensorflow as tf


def prelu(x, name=""):
    """
    Args:
        x (tf.Tensor): 输入tensor
        name (str): alpha的变量名后缀
    Returns:
        经过prelu激活后的tensor
    """

    alpha = tf.get_variable(name=f"prelu_alpha_{name}",
                            shape=x.shape[-1],
                            initializer=tf.constant_initializer(1.0),
                            dtype=x.dtype)
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def dice(x, name=""):
    """
    Args:
        x (tf.Tensor): 输入tensor
        name (str): alpha, beta的变量名后缀
    Returns:
        经过dice激活后的tensor
    """

    alpha = tf.get_variable(name=f"dice_alpha_{name}",
                            shape=x.shape[-1],
                            initializer=tf.constant_initializer(1.0),
                            dtype=x.dtype)
    # 利用batch_normalization的API, 无需可训练参数beta和gamma
    x_norm = tf.layers.batch_normalization(x, center=False, scale=False, name=f"dice_bn_{name}")
    px = tf.sigmoid(x_norm)

    return x * px + alpha * x * (1 - px)
