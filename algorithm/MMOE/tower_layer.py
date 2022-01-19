import tensorflow as tf


def tower_layer(x, hidden_units, mode, batch_norm=True, dropout_rate=0.1, name=""):
    """
        任务塔网络
    Args:
        x (tf.Tensor): 输入tensor, 专家网络的输出
        hidden_units (list): 隐藏层维度
        mode (str): 当前模式, 外部作用域直接传入
        batch_norm (bool): 是否使用bn
        dropout_rate (float): dropout_rate
        name (str): 变量名后缀
    Returns:
        任务logit
    """

    net = x
    for i, unit in enumerate(hidden_units):
        net = tf.layers.dense(net, unit, activation=tf.nn.relu)
        if 0.0 < dropout_rate < 1.0:
            net = tf.layers.dropout(net, dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))
        if batch_norm:
            net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logit = tf.layers.dense(net, 1, name=f"tower_{name}_logit")

    return logit
