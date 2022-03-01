import math
import tensorflow as tf
from leakyrelu import leakyrelu


def bst_transformer(queries, keys, values, keys_length, heads, index, max_length, use_position_embedding=True):
    """
        实现transformer层
    Args:
        queries (tf.Tensor): 目标feed+历史行为序列, shape=(B, T, dk)
        keys (tf.Tensor): 在bst中, 同queries
        values (tf.Tensor): 在bst中, 同queries
        keys_length (tf.Tensor): 历史行为队列长度+1, 目的为生成mask, shape=(B, )
        heads (int): head数量, 最少为1
        index (int): transformer层的序号
        max_length: 序列长度T的最大值, 应为最大历史序列长度+1
        use_position_embedding (bool): 是否使用位置编码embedding (sinusoid形式的实现请参考https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer.py)

    Returns:
        tf.Tensor
    """

    # 静态维度信息
    d_k = int(queries.get_shape()[-1])
    d_model = d_k

    # position embedding
    if use_position_embedding:
        # 第0个位置编码为目标feed的, 从1开始是真正的位置编码
        position_embedding_table = tf.get_variable(name="position_embedding", shape=(max_length, d_k))
        # 注意这里queries的第1个维度是动态的, 只能tf.shape得到, 不能用queries.get_shape()
        # refer: https://blog.csdn.net/Hk_john/article/details/78213933
        position_ids = tf.expand_dims(tf.range(tf.shape(queries)[1]), 0)   # (1, T), [[0,1,2,3,...,T-1]]
        position_emb = tf.nn.embedding_lookup(position_embedding_table, position_ids)    # (1, T, d_k)
        # value不加位置编码
        queries += position_emb
        keys += position_emb

    # project matrix
    w_q = tf.get_variable(name=f"w_q_{index}", shape=(heads, d_k, d_model))
    w_k = tf.get_variable(name=f"w_k_{index}", shape=(heads, d_k, d_model))
    w_v = tf.get_variable(name=f"w_v_{index}", shape=(heads, d_k, d_model))
    w_o = tf.get_variable(name=f"w_o_{index}", shape=(heads*d_model, d_model))

    # self attention
    Q = tf.einsum("...bik,...hkj -> ...bhij", queries, w_q)  # (batch, heads, T, d_model)
    K = tf.einsum("...bik,...hkj -> ...bhij", keys, w_k)  # (batch, heads, T, d_model)
    V = tf.einsum("...bik,...hkj -> ...bhij", values, w_v)  # (batch, heads, T, d_model)

    K_T = tf.transpose(K, perm=[0, 1, 3, 2])    # (batch, heads, d_model, T)

    # mask
    keys_mask = tf.sequence_mask(keys_length, tf.shape(keys)[1], dtype=tf.float32)  # (batch, T)
    keys_mask = (keys_mask - 1) * (-1)
    keys_mask = keys_mask * (-2 ** 32 + 1)  # (batch, T)
    keys_mask = tf.expand_dims(keys_mask, axis=-1)  # (batch, T, 1)
    keys_mask = tf.expand_dims(keys_mask, axis=1)  # (batch, 1, T, 1)

    # 计算softmax和score
    soft_max = tf.matmul(Q, K_T) / math.sqrt(d_k)    # (batch, heads, T, T)
    soft_max = soft_max + keys_mask  # (batch, 1, T, 1) + (batch, heads, T, T) = (batch, heads, T, T)
    soft_max = tf.nn.softmax(soft_max, axis=-1)  # (batch, heads, T, T)
    score = tf.matmul(soft_max, V)  # (batch, heads, T, d_model)

    # concat所有head
    all_heads = tf.transpose(score, perm=[0, 2, 1, 3])  # (batch, T, heads, d_model)
    all_heads = tf.reshape(all_heads, shape=(-1, tf.shape(queries)[1], heads*d_model))  # (batch, T, heads*d_model)
    all_heads = tf.matmul(all_heads, w_o)   # (batch, T, heads*d_model) @ (heads*d_model, d_model) = (batch, T, d_model)

    # add & layer norm
    net = all_heads + queries  # (batch, T, d_model)
    net = tf.contrib.layers.layer_norm(net)  # (batch, T, d_model)

    # FFN
    ffn = tf.layers.dense(net, units=d_model, activation=None)   # (batch, T, d_model)
    ffn = leakyrelu(ffn)
    # add & layer norm
    net = ffn + net
    net = tf.contrib.layers.layer_norm(net)  # (batch, T, d_model)

    return net


if __name__ == "__main__":
    # Test
    # B=2, T=3, d_k=4
    fake_queries = tf.random_normal(shape=(2, 3, 4))
    fake_keys_length = tf.constant([0, 1])
    out = bst_transformer(queries=fake_queries,
                          keys=fake_queries,
                          values=fake_queries,
                          keys_length=fake_keys_length,
                          heads=3,
                          index=0,
                          max_length=5,
                          use_position_embedding=True)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(out))
