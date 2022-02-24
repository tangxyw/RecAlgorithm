import tensorflow as tf


def cin_layer(x0, xk, hk_1, index):
    """
        xdeepfm模型的Compressed Interaction Network部分
    Args:
        x0 (tensor): 原始输入tensor, shape=(batch, m, D)
        xk (tensor): 上一CIN层输出tensor, shape=(batch, hk, D)
        hk_1 (int): 本层输出维度, 也是feature map个数
        index (int): 序号

    Returns:
        tensor, 维度与输入tensor一致
    """

    D = int(x0.get_shape()[-1])
    m = int(x0.get_shape()[1])
    hk = int(xk.get_shape()[1])

    outer = tf.einsum('...ik,...jk -> ...kij', xk, x0)  # (batch, D, hk, m)
    outer = tf.reshape(outer, shape=(-1, D, hk*m))  # (batch, D, hk*m)

    # 卷积核, 相当于论文中的wk
    filters = tf.get_variable(name=f"cin_layer_{index}_filter", shape=(1, hk*m, hk_1))
    # 一维卷积操作
    xk_1 = tf.nn.conv1d(outer, filters=filters, stride=1, padding="VALID")  # (batch, D, hk_1)
    xk_1 = tf.transpose(xk_1, perm=[0, 2, 1])   # (batch, hk_1, D)

    return xk_1
