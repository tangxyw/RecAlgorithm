<<<<<<< HEAD
import tensorflow as tf


def din_attention(query, keys, keys_length, is_softmax=False):
    """
        实现DIN模型中的attention模块
    Args:
        query (tf.Tensor):  目标  shape=(B, H)
        keys (tf.Tensor):   历史行为序列, shape=(B, T, H)
        keys_length (tf.Tensor):    历史行为队列长度, 目的为生成mask, shape=(B, )
        is_softmax (bool):  attention权重是否使用softmax激活

    Returns:
        tf.Tensor, weighted sum pooling结果
    """

    embedding_dim = query.shape[-1].value
    query = tf.tile(query, multiples=[1, tf.shape(keys)[1]])  # (B, H*T)
    query = tf.reshape(query, shape=(-1, tf.shape(keys)[1], embedding_dim))  # (B, T, H)
    cross_all = tf.concat([query, keys, query - keys, query * keys], axis=-1)  # (B, T, 4*H)
    d_layer_1_all = tf.layers.dense(cross_all, 64, activation=tf.nn.relu, name='f1_att', reuse=tf.AUTO_REUSE)  # (B, T, 64)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 32, activation=tf.nn.relu, name='f2_att', reuse=tf.AUTO_REUSE)  # (B, T, 32)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)  # (B, T, 1)
    output_weight = d_layer_3_all  # (B, T, 1)

    # mask
    keys_mask = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (B, T)
    keys_mask = tf.expand_dims(keys_mask, -1)  # 与output_weight对齐, (B, T, 1)

    if is_softmax:
        paddings = tf.ones_like(output_weight) * (-2 ** 32 + 1)  # (B, T, 1)
        output_weight = tf.where(keys_mask, output_weight, paddings)  # (B, T, 1)
        # scale, 防止梯度消失
        output_weight = output_weight / (embedding_dim ** 0.5)  # (B, T, 1)
        output_weight = tf.nn.softmax(output_weight, axis=1)  # (B, T, 1)
    else:  # 按论文原文, 不使用softmax激活
        output_weight = tf.cast(keys_mask, tf.float32)  # (B, T, 1)

    outputs = tf.matmul(output_weight, keys, transpose_a=True)  # (B, 1, T) * (B, T, H) = (B, 1, H)
    outputs = tf.squeeze(outputs, 1)    # (B, H)

    return outputs


if __name__ == "__main__":
    # Test
    # B=2, T=3, H=4
    # fake_keys = tf.zeros(shape=(2, 3, 4))
    fake_keys = tf.random_normal(shape=(2, 3, 4))
    fake_query = tf.random_normal(shape=(2, 4))
    fake_keys_length = tf.constant([0, 1], 3)
    attention_out1 = din_attention(fake_query, fake_keys, fake_keys_length, is_softmax=False)
    attention_out2 = din_attention(fake_query, fake_keys, fake_keys_length, is_softmax=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("不使用softmax激活:")
        print(sess.run(attention_out1))
        print("使用softmax激活:")
        print(sess.run(attention_out2))
=======
import tensorflow as tf


def din_attention(query, keys, keys_length, is_softmax=False):
    """
        实现DIN模型中的attention模块
    Args:
        query (tf.Tensor):  目标  shape=(B, H)
        keys (tf.Tensor):   历史行为序列, shape=(B, T, H)
        keys_length (tf.Tensor):    历史行为队列长度, 目的为生成mask, shape=(B, )
        is_softmax (bool):  attention权重是否使用softmax激活

    Returns:
        tf.Tensor, weighted sum pooling结果
    """

    embedding_dim = query.shape[-1].value
    query = tf.tile(query, multiples=[1, tf.shape(keys)[1]])  # (B, H*T)
    query = tf.reshape(query, shape=(-1, tf.shape(keys)[1], embedding_dim))  # (B, T, H)
    cross_all = tf.concat([query, keys, query - keys, query * keys], axis=-1)  # (B, T, 4*H)
    d_layer_1_all = tf.layers.dense(cross_all, 64, activation=tf.nn.relu, name='f1_att', reuse=tf.AUTO_REUSE)  # (B, T, 64)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 32, activation=tf.nn.relu, name='f2_att', reuse=tf.AUTO_REUSE)  # (B, T, 32)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)  # (B, T, 1)
    output_weight = d_layer_3_all  # (B, T, 1)

    # mask
    keys_mask = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (B, T)
    keys_mask = tf.expand_dims(keys_mask, -1)  # 与output_weight对齐, (B, T, 1)

    if is_softmax:
        paddings = tf.ones_like(output_weight) * (-2 ** 32 + 1)  # (B, T, 1)
        output_weight = tf.where(keys_mask, output_weight, paddings)  # (B, T, 1)
        # scale, 防止梯度消失
        output_weight = output_weight / (embedding_dim ** 0.5)  # (B, T, 1)
        output_weight = tf.nn.softmax(output_weight, axis=1)  # (B, T, 1)
    else:  # 按论文原文, 不使用softmax激活
        output_weight = tf.cast(keys_mask, tf.float32)  # (B, T, 1)

    outputs = tf.matmul(output_weight, keys, transpose_a=True)  # (B, 1, T) * (B, T, H) = (B, 1, H)
    outputs = tf.squeeze(outputs, 1)    # (B, H)

    return outputs


if __name__ == "__main__":
    # Test
    # B=2, T=3, H=4
    # fake_keys = tf.zeros(shape=(2, 3, 4))
    fake_keys = tf.random_normal(shape=(2, 3, 4))
    fake_query = tf.random_normal(shape=(2, 4))
    fake_keys_length = tf.constant([0, 1], 3)
    attention_out1 = din_attention(fake_query, fake_keys, fake_keys_length, is_softmax=False)
    attention_out2 = din_attention(fake_query, fake_keys, fake_keys_length, is_softmax=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("不使用softmax激活:")
        print(sess.run(attention_out1))
        print("使用softmax激活:")
        print(sess.run(attention_out2))
>>>>>>> 734986b93a9246f05fb1b15f98977242f436de04
