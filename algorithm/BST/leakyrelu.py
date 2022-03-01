import tensorflow as tf


def leakyrelu(x, leak=0.01):
    """
        leakyrelu激活函数
    Args:
        x (Tensor): input
        leak (int): x<0时的斜率

    Returns:
        Tensor
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)

