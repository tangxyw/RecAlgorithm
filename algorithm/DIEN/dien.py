"""
    [1] Zhou, Guorui & Mou, Na & Fan, Ying & Pi, Qi & Bian, Weijie & Zhou, Chang & Zhu, Xiaoqiang & Gai, Kun. (2018).
    Deep Interest Evolution Network for Click-Through Rate Prediction.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from typing import List, Tuple, Any
import pandas as pd
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import feature_column as fc
from utils import train_input_fn, eval_input_fn
from rnn import dynamic_rnn
from custom_grucell import AGRUCell, AUGRUCell
from DIN.activations import prelu, dice

# 定义输入参数
flags = tf.app.flags

# 训练参数
flags.DEFINE_string("model_dir", "./model_dir", "Directory where model parameters, graph, etc are saved")
flags.DEFINE_string("output_dir", "./output_dir", "Directory where pb file are saved")

# flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "../../dataset/wechat_algo_data1/tfrecord/train.tfrecord", "Path to the train data")
flags.DEFINE_string("eval_data", "../../dataset/wechat_algo_data1/tfrecord/test.tfrecord",
                    "Path to the evaluation data")
flags.DEFINE_string("vocabulary_dir", "../../dataset/wechat_algo_data1/vocabulary/",
                    "Folder where the vocabulary file is stored")
flags.DEFINE_integer("num_epochs", 1, "Epoch of training phase")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "Dataset shuffle buffer size")
flags.DEFINE_integer("num_parallel_readers", -1, "Number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "Save checkpoints every this many steps")

# 模型参数
flags.DEFINE_integer("batch_size", 1024, "Training batch size")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")
flags.DEFINE_string("activation", "dice", "Dense layer activation, supported strings are in {'prelu', dice'}")
# flags.DEFINE_boolean("mini_batch_aware_regularization", True, "Whether to use mini_batch_aware_regularization")
# flags.DEFINE_float("l2_lambda", 0.2, "Coefficient when using mini_batch_aware_regularization")
# flags.DEFINE_boolean("use_softmax", False, "Whether to use softmax on attention score")
flags.DEFINE_boolean("use_auxiliary_loss", False, "Whether to use auxiliary loss.")
flags.DEFINE_integer("negative_sample_number", 3, "The number of negative samples per positive sample.")
flags.DEFINE_string("custom_gru_type", "AGRU", "The Type of custom GRU cell, supported strings are in {'AGRU', AUGRU'}")
flags.DEFINE_integer("gru_output_units", 8, "output dimension of gru cell.")

FLAGS = flags.FLAGS


def create_feature_columns() -> Tuple[list, list, list, list, list, list]:
    """

    Returns:
        dense_feature_columns (list): 连续特征的feature_columns
        category_feature_columns (list): 类别特征的feature_columns
        target_feedid_feature_columns (list): 目标feed的feature_columns
        sequence_feature_columns (list): 历史行为队列的feature_columns
        negative_sequence_feature_columns (list): 负采样队列的feature_columns
        label_feature_columns (list): 因变量的feature_columns
    """

    category_feature_columns, dense_feature_columns = [], []
    target_feedid_feature_columns, sequence_feature_columns, negative_sequence_feature_columns = [], [], []
    label_feature_columns = []

    # 连续特征
    videoplayseconds = fc.numeric_column('videoplayseconds', default_value=0.0)
    u_read_comment_7d_sum = fc.numeric_column('u_read_comment_7d_sum', default_value=0.0)
    u_like_7d_sum = fc.numeric_column('u_like_7d_sum', default_value=0.0)
    u_click_avatar_7d_sum = fc.numeric_column('u_click_avatar_7d_sum', default_value=0.0)
    u_forward_7d_sum = fc.numeric_column('u_forward_7d_sum', default_value=0.0)
    u_comment_7d_sum = fc.numeric_column('u_comment_7d_sum', default_value=0.0)
    u_follow_7d_sum = fc.numeric_column('u_follow_7d_sum', default_value=0.0)
    u_favorite_7d_sum = fc.numeric_column('u_favorite_7d_sum', default_value=0.0)

    i_read_comment_7d_sum = fc.numeric_column('i_read_comment_7d_sum', default_value=0.0)
    i_like_7d_sum = fc.numeric_column('i_like_7d_sum', default_value=0.0)
    i_click_avatar_7d_sum = fc.numeric_column('i_click_avatar_7d_sum', default_value=0.0)
    i_forward_7d_sum = fc.numeric_column('i_forward_7d_sum', default_value=0.0)
    i_comment_7d_sum = fc.numeric_column('i_comment_7d_sum', default_value=0.0)
    i_follow_7d_sum = fc.numeric_column('i_follow_7d_sum', default_value=0.0)
    i_favorite_7d_sum = fc.numeric_column('i_favorite_7d_sum', default_value=0.0)

    c_user_author_read_comment_7d_sum = fc.numeric_column('c_user_author_read_comment_7d_sum', default_value=0.0)

    dense_feature_columns += [videoplayseconds, u_read_comment_7d_sum, u_like_7d_sum, u_click_avatar_7d_sum,
                              u_forward_7d_sum, u_comment_7d_sum, u_follow_7d_sum, u_favorite_7d_sum,
                              i_read_comment_7d_sum, i_like_7d_sum, i_click_avatar_7d_sum, i_forward_7d_sum,
                              i_comment_7d_sum, i_follow_7d_sum, i_favorite_7d_sum,
                              c_user_author_read_comment_7d_sum]

    # 类别特征
    userid = fc.categorical_column_with_vocabulary_file('userid', os.path.join(FLAGS.vocabulary_dir, 'userid.txt'))
    feedid = fc.sequence_categorical_column_with_vocabulary_file('feedid',
                                                                 os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))
    device = fc.categorical_column_with_vocabulary_file('device', os.path.join(FLAGS.vocabulary_dir, 'device.txt'))
    authorid = fc.categorical_column_with_vocabulary_file('authorid',
                                                          os.path.join(FLAGS.vocabulary_dir, 'authorid.txt'))
    bgm_song_id = fc.categorical_column_with_vocabulary_file('bgm_song_id',
                                                             os.path.join(FLAGS.vocabulary_dir, 'bgm_song_id.txt'))
    bgm_singer_id = fc.categorical_column_with_vocabulary_file('bgm_singer_id',
                                                               os.path.join(FLAGS.vocabulary_dir, 'bgm_singer_id.txt'))

    manual_tag_list = fc.categorical_column_with_vocabulary_file('manual_tag_list', os.path.join(FLAGS.vocabulary_dir,
                                                                                                 'manual_tag_id.txt'))
    his_read_comment_7d_seq = fc.sequence_categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
                                                                                  os.path.join(FLAGS.vocabulary_dir,
                                                                                               'feedid.txt'))
    ## 负采样序列, 输入shape为(B, (T-1)*T_neg), T_neg为为每个正样本采样的负样本个数
    ## 因为tf.contrib.feature_column.sequence_input_layer的输出是恒定三维，所以要做上述处理
    ## 原始论文的实现中, 负采样数量T_neg为3，但是后续为了简单，只用了第0个负样本, ref: https://github.com/mouna99/dien/blob/1f314d16aa1700ee02777e6163fb8ca94e3d2810/script/model.py#L52
    ## 目前生成训练集时尚未实现负采样逻辑
    neg_read_comment_seq = fc.sequence_categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
                                                                               os.path.join(FLAGS.vocabulary_dir,
                                                                                            'feedid.txt'))

    userid_emb = fc.embedding_column(userid, 16)
    feedid_emb = fc.shared_embedding_columns([feedid, his_read_comment_7d_seq, neg_read_comment_seq], 16,
                                             combiner='mean')
    device_emb = fc.embedding_column(device, 2)
    authorid_emb = fc.embedding_column(authorid, 4)
    bgm_song_id_emb = fc.embedding_column(bgm_song_id, 4)
    bgm_singer_id_emb = fc.embedding_column(bgm_singer_id, 4)
    manual_tag_id_emb = fc.embedding_column(manual_tag_list, 4, combiner='mean')

    category_feature_columns += [userid_emb, device_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb,
                                 manual_tag_id_emb]

    target_feedid_feature_columns += [feedid_emb[0]]
    sequence_feature_columns += [feedid_emb[1]]
    negative_sequence_feature_columns += [feedid_emb[2]]

    # label
    read_comment = fc.numeric_column("read_comment", default_value=0.0)
    label_feature_columns += [read_comment]

    return dense_feature_columns, category_feature_columns, target_feedid_feature_columns, sequence_feature_columns, negative_sequence_feature_columns, label_feature_columns


def example_parser(serialized_example):
    """
        批量解析Example
    Args:
        serialized_example:

    Returns:
        features, labels
    """
    fea_columns = total_feature_columns
    label_columns = label_feature_columns

    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns + label_columns)
    features = tf.parse_example(serialized_example, features=feature_spec)
    read_comment = features.pop("read_comment")
    return features, {"read_comment": read_comment}


def dien_model_fn(features, labels, mode, params):
    """
        dien模型的model_fn
    Args:
        features (dict): input_fn的第一个返回值, 模型输入样本特征
        labels (dict): input_fn的第二个返回值, 样本标签
        mode: tf.estimator.ModeKeys
        params (dict): 模型超参数

    Returns:
        tf.estimator.EstimatorSpec
    """

    # 连续特征
    with tf.variable_scope("dense_input"):
        dense_input = fc.input_layer(features, params["dense_feature_columns"])

    # 类别特征
    with tf.variable_scope("category_input"):
        category_input = fc.input_layer(features, params["category_feature_columns"])

    # 目标feed
    with tf.variable_scope("target_input"):
        target_input, _ = tf.contrib.feature_column.sequence_input_layer(features, params[
            "target_feedid_feature_columns"])  # (B, 1, H)
        target_input = tf.squeeze(target_input, axis=1)  # (B, H)

    # 历史行为序列
    with tf.variable_scope("his_seq_input"):
        sequnence_input, sequnence_length = tf.contrib.feature_column.sequence_input_layer(features, params[
            "sequence_feature_columns"])  # (B, T, H), (B,)

    # 历史行为序列编码
    with tf.variable_scope("seq_encoder"):
        # 此处有bug, 在Graph构建完成后，训练过程中报错tensorflow.python.framework.errors_impl.InvalidArgumentError: Retval[0] does not have value
        # 可能与dynamic_rnn中使用了tf.cond有关
        h, _ = dynamic_rnn(cell=tf.nn.rnn_cell.GRUCell(num_units=params["gru_output_units"]),
                           inputs=sequnence_input,
                           dtype=tf.float32)    # (B, T, nh)
        # Attention, 论文作者的实现和原始论文不一样, 论文作者使用的是DIN的Attention, 此处与原始论文保持一致
        nh = params["gru_output_units"]
        na = int(sequnence_input.get_shape()[-1])  # na就是H, 这里用na为了与原始论文保持一致
        w = tf.get_variable(name="attention_project_matrix", shape=(nh, na), dtype=tf.float32)
        w_ea = tf.matmul(w, tf.expand_dims(target_input, -1))  # (nh, na) matmul (B, na, 1) = (B, nh, 1)
        h_w_ea = tf.matmul(h, w_ea)  # (B, T, nh) matmul (B, nh, 1) = (B, T, 1)

        # seq mask
        seq_mask = tf.sequence_mask(sequnence_length, tf.shape(sequnence_input)[1])  # (B, T)
        seq_mask = tf.expand_dims(seq_mask, -1)  # 与output_weight对齐, (B, T, 1)
        # 计算attention得分
        paddings = tf.ones_like(h_w_ea) * (-2 ** 32 + 1)  # (B, T, 1)
        attention_scores = tf.where(seq_mask, h_w_ea, paddings)  # (B, T, 1)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)  # (B, T, 1)

        if params["custom_gru_type"] == "AUGRU":
            custom_gru_cell = AUGRUCell(nh)
        else:
            custom_gru_cell = AGRUCell(nh)

        _, final_state = dynamic_rnn(custom_gru_cell,
                                     h,
                                     attention_scores,
                                     sequence_length=sequnence_length,
                                     dtype=tf.float32)  # (B, nh)

        # gather序列最后一个状态输出
        # sequnence_length = tf.expand_dims(sequnence_length, -1)  # (B, 1)
        # final_state = tf.gather(custom_gru_output, indices=sequnence_length, batch_dims=1)  # (B, 1, nh)
        # final_state = tf.squeeze(final_state, axis=1)  # (B, nh)

        # concat all
        concat_all = tf.concat([dense_input, category_input, target_input, final_state], axis=-1)

        # 全连接层
        with tf.variable_scope("fcn"):
            net = concat_all
            for i, unit in enumerate(params["hidden_units"]):
                layer_index = i + 1
                net = tf.layers.dense(net, unit, activation=None)
                if params["activation"] == "dice":
                    net = dice(net, name=layer_index)
                else:
                    net = prelu(net, name=layer_index)
                if params["batch_norm"]:
                    net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
                if "dropout_rate" in params and 0.0 < params["dropout_rate"] < 1.0:
                    net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))

            logit = tf.layers.dense(net, 1)

    # 使用辅助任务
    # 需要用到历史行为序列的负采样序列, shape=(B, (T-1)*T_neg, H)
    # 要保证每个batch内min(T) >= 2, 一般batch大小为1024的话，不会有问题; todo：以防万一还是要考虑这一点
    # P.S: 数据原因, 以下代码无法运行, 仅做参考, 若有问题, 请联系作者.
    # 可能的原因之一: 经过tf.log后输出变成了-inf或者nan
    if FLAGS.use_auxiliary_loss:
        with tf.variable_scope("neg_seq_input"):
            neg_sequnence_input, _ = tf.contrib.feature_column.sequence_input_layer(features, params["negative_sequence_feature_columns"])  # (B, (T-1)*T_neg, H)
        with tf.variable_scope("aux_loss"):
            pos_sequnence_input = sequnence_length[:, 1:, :]    # (B, T-1, na)
            h_ = h[:, :-1, :]    # (B, T-1, nh)
            # 将pos和neg投影到h_的空间, 以便可以做内积运算
            w_aux = tf.get_variable(name="aux_project_matrix", shape=(na, nh))
            pos = tf.matmul(pos_sequnence_input, w_aux)     # (B, T-1, na) matmul (na, nh) = (B, T-1, nh)
            neg = tf.matmul(neg_sequnence_input, w_aux)     # (B, (T-1)*T_neg, na) matmul (na, nh) = (B, (T-1)*T_neg, nh)
            # 内积计算
            # 正样本部分
            pos_part = tf.einsum("bth,bth->bt", h_, pos)    # (B, T-1)
            pos_part = tf.log(tf.sigmoid(pos_part))    # (B, T-1)
            # 负样本部分
            # 要先变换shape
            T = tf.shape(sequnence_input)[1]
            T_neg = params["negative_sample_number"]
            neg = tf.reshape(neg, shape=(-1, T-1, T_neg, nh))
            neg_part = tf.einsum("bth,btnh->btn", h_, neg)    # (B, T-1, T_neg)
            neg_part = tf.log(1 - tf.sigmoid(neg_part))    # (B, T-1, T_neg)
            # 再变回去
            neg_part = tf.reshape(neg_part, shape=(-1, (T-1)*T_neg))    # (B, (T-1)*T_neg)

            # 生成mask
            aux_mask = tf.sequence_mask(sequnence_length-1, T-1)  # (B, T-1)
            neg_mask = tf.expand_dims(aux_mask, axis=-1)    # (B, T-1, 1)
            neg_mask = tf.tile(neg_mask, multiples=(1, 1, T_neg))   # (B, T-1, T_neg)
            aux_mask = tf.cast(aux_mask, tf.float32)
            neg_mask = tf.cast(neg_mask, tf.float32)

            # mask操作
            pos_part = pos_part * aux_mask
            neg_part = neg_part * neg_mask

            # 计算loss
            pos_part = tf.reduce_sum(pos_part, axis=1, keepdims=True)  # (B, 1)
            neg_part = tf.reduce_sum(neg_part, axis=[1, 2])  # (B, )
            neg_part = tf.expand_dims(neg_part, axis=-1)    # (B, 1)
            aux_loss = tf.reduce_mean(pos_part+neg_part)

    # -----定义PREDICT阶段行为-----
    prediction = tf.sigmoid(logit, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': prediction
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    # -----定义完毕-----

    y = labels["read_comment"]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit), name="ce_loss")
    if FLAGS.use_auxiliary_loss:
        loss += aux_loss

    accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
    auc = tf.metrics.auc(labels=y, predictions=prediction)

    # -----定义EVAL阶段行为-----
    metrics = {"eval_accuracy": accuracy, "eval_auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # -----定义完毕-----

    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=params["learning_rate"], beta1=0.9,
                                                 beta2=0.999, epsilon=1e-8)
    # optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=0.9,
    #                                    beta2=0.999, epsilon=1e-8)
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # -----定义TRAIN阶段行为-----
    assert mode == tf.estimator.ModeKeys.TRAIN

    # tensorboard收集
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_auc", auc[1])

    # 训练log打印
    log_hook = tf.train.LoggingTensorHook(
        {
            "train_loss": loss,
            "train_auc": auc[1],
            "attention_weights": attention_scores,
        },
        every_n_iter=100
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
    # -----定义完毕-----


def main(unused_argv):
    """训练入口"""

    global total_feature_columns, label_feature_columns
    dense_feature_columns, category_feature_columns, target_feedid_feature_columns, sequence_feature_columns, negative_sequence_feature_columns, label_feature_columns = create_feature_columns()
    total_feature_columns = dense_feature_columns + category_feature_columns + target_feedid_feature_columns + sequence_feature_columns \
                            + negative_sequence_feature_columns

    params = {
        "dense_feature_columns": dense_feature_columns,
        "category_feature_columns": category_feature_columns,
        "sequence_feature_columns": sequence_feature_columns,
        "negative_sequence_feature_columns": negative_sequence_feature_columns,
        "target_feedid_feature_columns": target_feedid_feature_columns,
        "hidden_units": FLAGS.hidden_units.split(','),
        "dropout_rate": FLAGS.dropout_rate,
        "batch_norm": FLAGS.batch_norm,
        "learning_rate": FLAGS.learning_rate,
        "activation": FLAGS.activation,
        "use_auxiliary_loss": FLAGS.use_auxiliary_loss,
        "negative_sample_number": FLAGS.negative_sample_number,
        "custom_gru_type": FLAGS.custom_gru_type,
        "gru_output_units": FLAGS.gru_output_units,
        # "mini_batch_aware_regularization": FLAGS.mini_batch_aware_regularization,
        # "l2_lambda": FLAGS.l2_lambda,
        # "use_softmax": FLAGS.use_softmax,
    }
    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=dien_model_fn,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(filepath=FLAGS.train_data, example_parser=example_parser,
                                        batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs,
                                        shuffle_buffer_size=FLAGS.shuffle_buffer_size),
        max_steps=FLAGS.train_steps
    )

    feature_spec = tf.feature_column.make_parse_example_spec(total_feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    exporters = [
        tf.estimator.BestExporter(
            name="best_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=5)
    ]
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser,
                                       batch_size=FLAGS.batch_size),
        throttle_secs=600,
        steps=None,
        exporters=exporters
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Evaluate Metrics.
    metrics = estimator.evaluate(input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser,
                                                                batch_size=FLAGS.batch_size))
    for key in sorted(metrics):
        print('%s: %s' % (key, metrics[key]))

    results = estimator.predict(input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser,
                                                               batch_size=FLAGS.batch_size))
    predicts_df = pd.DataFrame.from_dict(results)
    predicts_df['probabilities'] = predicts_df['probabilities'].apply(lambda x: x[0])
    test_df = pd.read_csv("../../dataset/wechat_algo_data1/dataframe/test.csv")
    predicts_df['read_comment'] = test_df['read_comment']
    predicts_df.to_csv("predictions.csv")
    print("after evaluate")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
