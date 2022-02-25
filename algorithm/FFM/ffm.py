"""
    [1] Juan, Yuchin, et al. "Field-aware factorization machines for CTR prediction."
    Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from typing import List, Tuple, Any
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
from utils import train_input_fn, eval_input_fn, to_sparse_tensor


# 定义输入参数
flags = tf.app.flags

# 训练参数
flags.DEFINE_string("model_dir", "./model_dir", "Directory where model parameters, graph, etc are saved")
flags.DEFINE_string("output_dir", "./output_dir", "Directory where pb file are saved")
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
flags.DEFINE_integer("embedding_dim", 8, "Embedding dimension")
# flags.DEFINE_string("hidden_units", "512,256,128",
#                     "Comma-separated list of number of units in each hidden layer of the deep part")
# flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
# flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")

FLAGS = flags.FLAGS


def create_feature_columns() -> Tuple[list, list]:
    """

    Returns:
        one_hot_category_feature_columns (list): 类别特征的feature_columns, 以one-hot形式表示
        label_feature_columns (list): label的feature_columns
    """

    one_hot_category_feature_columns = []
    label_feature_columns = []

    # 类别特征
    userid = fc.categorical_column_with_vocabulary_file('userid', os.path.join(FLAGS.vocabulary_dir, 'userid.txt'))
    feedid = fc.categorical_column_with_vocabulary_file('feedid', os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))
    device = fc.categorical_column_with_vocabulary_file('device', os.path.join(FLAGS.vocabulary_dir, 'device.txt'))
    authorid = fc.categorical_column_with_vocabulary_file('authorid',
                                                          os.path.join(FLAGS.vocabulary_dir, 'authorid.txt'))
    bgm_song_id = fc.categorical_column_with_vocabulary_file('bgm_song_id',
                                                             os.path.join(FLAGS.vocabulary_dir, 'bgm_song_id.txt'))
    bgm_singer_id = fc.categorical_column_with_vocabulary_file('bgm_singer_id',
                                                               os.path.join(FLAGS.vocabulary_dir, 'bgm_singer_id.txt'))

    manual_tag_list = fc.categorical_column_with_vocabulary_file('manual_tag_list',
                                                                 os.path.join(FLAGS.vocabulary_dir, 'manual_tag_id.txt'))
    # his_read_comment_7d_seq = fc.categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
    #                                                                      os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))

    # 转为one-hot特征
    userid_one_hot = fc.indicator_column(userid)
    feedid_one_hot = fc.indicator_column(feedid)
    device_one_hot = fc.indicator_column(device)
    authorid_one_hot = fc.indicator_column(authorid)
    bgm_song_id_one_hot = fc.indicator_column(bgm_song_id)
    bgm_singer_id_one_hot = fc.indicator_column(bgm_singer_id)
    manual_tag_mulit_hot = fc.indicator_column(manual_tag_list)

    one_hot_category_feature_columns += [userid_one_hot, feedid_one_hot, device_one_hot, authorid_one_hot,
                                         bgm_song_id_one_hot, bgm_singer_id_one_hot, manual_tag_mulit_hot]

    # label
    read_comment = fc.numeric_column("read_comment", default_value=0.0)
    label_feature_columns += [read_comment]

    return one_hot_category_feature_columns, label_feature_columns


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


def ffm_model_fn(features, labels, mode, params):
    """
        ffm模型的model_fn
    Args:
        features (dict): input_fn的第一个返回值, 模型输入样本特征
        labels (dict): input_fn的第二个返回值, 样本标签
        mode: tf.estimator.ModeKeys
        params (dict): 模型超参数

    Returns:
        tf.estimator.EstimatorSpec
    """

    # 一阶部分
    with tf.variable_scope("ffm_first_order"):
        ffm_first_order_input = fc.input_layer(features, params["one_hot_category_feature_columns"])
        ffm_first_order_logit = tf.layers.dense(ffm_first_order_input, 1, name="fm_first_order_dense")  # (batch, 1)

    # 创建每个field的embedding变量
    with tf.variable_scope("embedding_variables"):
        # 存放每一个field的embedding变量, 每一个变量shape=(F-1, |Vi|, K)
        embedding_variables = []    # [(F-1, |Vi|, K), ...]
        for name, vocabulary_size in params["fields_vocabulary_size_tuple"]:
            emb = tf.get_variable(name=f"{name}_embedding",
                                  shape=(len(params["one_hot_category_feature_columns"])-1, vocabulary_size, params["embedding_dim"]),
                                  dtype=tf.float32)
            embedding_variables.append(emb)

    # 二阶部分
    with tf.variable_scope("ffm_second_order"):
        field_sparse_ids_list = []
        for one_hot_category_feature_column in params["one_hot_category_feature_columns"]:
            one_hot_input = fc.input_layer(features, [one_hot_category_feature_column])  # (batch, |Vi|)
            field_sparse_ids = to_sparse_tensor(one_hot_input)
            field_sparse_ids_list.append(field_sparse_ids)

        second_order_vec = 0.0
        for i in range(len(params["one_hot_category_feature_columns"])-1):  # 循环F-1次
            for j in range(i+1, len(params["one_hot_category_feature_columns"])):
                # 取出i, j位置的稀疏id
                fi_sparse_ids = field_sparse_ids_list[i]
                fj_sparse_ids = field_sparse_ids_list[j]
                # 因为恒有 i < j
                # 在field i的embedding中取出第一个维度为j-1的子embedding
                # 在field j的embedding中取出第一个维度为i的子embedding
                fi_embedding_variable = embedding_variables[i][j - 1, :, :]   # (|Vi|, K)
                fj_embedding_variable = embedding_variables[j][i, :, :]  # (|Vj|, K)
                # embedding lookup
                vi = tf.nn.safe_embedding_lookup_sparse(fi_embedding_variable, sparse_ids=fi_sparse_ids)    # (batch, K)
                vj = tf.nn.safe_embedding_lookup_sparse(fj_embedding_variable, sparse_ids=fj_sparse_ids)    # (batch, K)
                interaction_vec = tf.multiply(vi, vj)   # (batch, K)
                second_order_vec += tf.reduce_sum(interaction_vec, axis=-1, keepdims=True)  # (batch, 1)

    # 合并
    total_logit = ffm_first_order_logit + second_order_vec

    # -----定义PREDICT阶段行为-----
    prediction = tf.sigmoid(total_logit, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "logit": total_logit,
            'probabilities': prediction,
        }
        saved_model_output = {
            'probabilities': prediction,
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(saved_model_output)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    # -----定义完毕-----

    y = labels["read_comment"]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=total_logit), name="loss")

    accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
    auc = tf.metrics.auc(labels=y, predictions=prediction)

    # -----定义EVAL阶段行为-----
    metrics = {"eval_accuracy": accuracy, "eval_auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # -----定义完毕-----

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=0.9,
                                       beta2=0.999, epsilon=1e-8)
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
            "tain_acc": accuracy[1],
            "train_auc": auc[1],
            "ffm_first_order_logit": ffm_first_order_logit,
            "ffm_second_order_logit": second_order_vec,
        },
        every_n_iter=100
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
    # -----定义完毕-----


def main(unused_argv):
    """训练入口"""

    global total_feature_columns, label_feature_columns
    one_hot_category_feature_columns, label_feature_columns = create_feature_columns()
    total_feature_columns = one_hot_category_feature_columns

    params = {
        "one_hot_category_feature_columns": one_hot_category_feature_columns,
        "learning_rate": FLAGS.learning_rate,
        "embedding_dim": FLAGS.embedding_dim,
        # [(类别变量名, 词典大小), ...]
        "fields_vocabulary_size_tuple": [(feature_column.categorical_column.name, int(feature_column.variable_shape[-1]))
                                         for feature_column in one_hot_category_feature_columns],
    }
    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=ffm_model_fn,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                      save_checkpoints_steps=FLAGS.save_checkpoints_steps)
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
    metrics = estimator.evaluate(
        input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser,
                                       batch_size=FLAGS.batch_size))
    for key in sorted(metrics):
        print('%s: %s' % (key, metrics[key]))

    results = estimator.predict(
        input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser,
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
