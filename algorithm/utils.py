import tensorflow as tf


def train_input_fn(filepath, example_parser, batch_size, num_epochs, shuffle_buffer_size):
    """
        模型的训练阶段input_fn
    Args:
        filepath (str): 训练集/验证集的路径
        example_parser (function): 解析example的函数
        batch_size (int): 每个batch样本大小
        num_epochs (int): 训练轮数
        shuffle_buffer_size (inr): shuffle时buffer的大小

    Returns:
        dataset
    """

    dataset = tf.data.TFRecordDataset(filepath)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(1)

    return dataset


def eval_input_fn(filepath, example_parser, batch_size):
    """
        模型的eval阶段input_fn
    Args:
        filepath (str): 训练集/验证集的路径
        example_parser (function): 解析example的函数
        batch_size (int): 每个batch样本大小

    Returns:
        dataset
    """

    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(1)

    return dataset


def to_sparse_tensor(one_hot_tensor):
    """
        将one-hot/multi-hot输入转化成稀疏张量, 作为tf.nn.safe_embedding_lookup_sparse的输入
    Args:
        one_hot_tensor (tensor): one-hot/multi-hot输入

    Returns:
        tf.SparseTensor
    """
    one_hot_indices = tf.where(tf.not_equal(one_hot_tensor, 0))
    one_hot_values = one_hot_indices[:, 1]

    return tf.SparseTensor(
        indices=one_hot_indices,
        values=one_hot_values,
        dense_shape=tf.shape(one_hot_tensor, out_type=tf.int64))

