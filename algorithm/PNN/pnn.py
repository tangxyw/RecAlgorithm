"""
    [1] Qu, Yanru, et al. "Product-based neural networks for user response prediction." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('..'))
from typing import List, Tuple, Any
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
from utils import train_input_fn, eval_input_fn

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
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")
flags.DEFINE_integer("output_dimension", 1024, "Output dimension of linear part and product part")
flags.DEFINE_string("product_method", "IPNN", "product_method, supported strings are in {'IPNN', 'OPNN'}")

FLAGS = flags.FLAGS


def create_feature_columns() -> Tuple[list, list]:
    """
        生成pnn模型输入特征和label
    Returns:
        category_feature_columns (list): 类别特征的feature_columns(包括序列特征)
        label_feature_columns (list): 因变量的feature_columns
    """

    category_feature_columns, label_feature_columns = [], []

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
                                                                 os.path.join(FLAGS.vocabulary_dir,
                                                                              'manual_tag_id.txt'))
    his_read_comment_7d_seq = fc.categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
                                                                         os.path.join(FLAGS.vocabulary_dir,
                                                                                      'feedid.txt'))

    userid_emb = fc.embedding_column(userid, FLAGS.embedding_dim)
    feedid_emb = fc.shared_embedding_columns([feedid, his_read_comment_7d_seq], FLAGS.embedding_dim, combiner='mean')
    device_emb = fc.embedding_column(device, FLAGS.embedding_dim)
    authorid_emb = fc.embedding_column(authorid, FLAGS.embedding_dim)
    bgm_song_id_emb = fc.embedding_column(bgm_song_id, FLAGS.embedding_dim)
    bgm_singer_id_emb = fc.embedding_column(bgm_singer_id, FLAGS.embedding_dim)
    manual_tag_id_emb = fc.embedding_column(manual_tag_list, FLAGS.embedding_dim, combiner='mean')

    category_feature_columns += [userid_emb, device_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb,
                                 manual_tag_id_emb]
    category_feature_columns += feedid_emb  # feedid_emb是list

    # label
    read_comment = fc.numeric_column("read_comment", default_value=0.0)
    label_feature_columns += [read_comment]

    return category_feature_columns, label_feature_columns


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


def pnn_model_fn(features, labels, mode, params):
    """
        pnn模型的model_fn
    Args:
        features (dict): input_fn的第一个返回值, 模型输入样本特征
        labels (dict): input_fn的第二个返回值, 样本标签
        mode: tf.estimator.ModeKeys
        params (dict): 模型超参数

    Returns:
        tf.estimator.EstimatorSpec
    """

    # 将每个类别特征的embedding取出
    fields_embeddings = []
    for cat_feature_column in params["category_feature_columns"]:
        embed_input = fc.input_layer(features, [cat_feature_column])  # (batch, K)
        fields_embeddings.append(embed_input)  # (batch, K) * F
    fields_embeddings = tf.concat(fields_embeddings, axis=-1)  # (batch, F*K)
    # fields_embeddings = tf.reshape(fields_embeddings, shape=(-1, len(params["category_feature_columns"]), FLAGS.embedding_dim))  # (batch, F, K)

    with tf.variable_scope("linear_part"):
        linear_w = tf.get_variable(name="linear_w",
                                   shape=(len(params["category_feature_columns"]) * FLAGS.embedding_dim,
                                          params["output_dimension"]),
                                   dtype=tf.float32,
                                   regularizer=tf.contrib.layers.l2_regularizer(scale=0.05))  # (F*K, D)
        lz = tf.matmul(fields_embeddings, linear_w)  # (batch, D)

    with tf.variable_scope("product_part"):
        # 变换输入的维度
        fields_embeddings = tf.reshape(fields_embeddings, shape=(-1, len(params["category_feature_columns"]), FLAGS.embedding_dim))  # (batch, F, K)
        # 存放lp的每一个分量
        product_output = []   # (batch, 1) * D
        if params["product_method"] == "IPNN":
            # 每一行对应一个theta, 每个theta的维度是特征数量F
            inner_product_w = tf.get_variable(name="inner_product_w",
                                              shape=(params["output_dimension"], len(params["category_feature_columns"])),
                                              dtype=tf.float32,
                                              regularizer=tf.contrib.layers.l2_regularizer(scale=0.05))  # (D, F)
            for i in range(params["output_dimension"]):  # 遍历每一个theta, 生成lp的每一个分量
                theta = tf.expand_dims(inner_product_w[i], axis=1)  # (F, 1)
                # delta定义见论文
                delta = tf.multiply(fields_embeddings, theta)   # (batch, F, K) multiply (F, 1) = (batch, F, K)，利用广播机制
                delta = tf.reduce_sum(delta, axis=1)    # (batch, K), 沿着特征数维度reduce
                lp_i = tf.reduce_sum(tf.square(delta), axis=1, keepdims=True)   # (batch, 1), 利用tf.square得到l2范数
                product_output.append(lp_i)

        else:   # OPNN
            outer_product_w = tf.get_variable(name="outer_product_w",
                                              shape=(params["output_dimension"], FLAGS.embedding_dim, FLAGS.embedding_dim),
                                              dtype=tf.float32,
                                              regularizer=tf.contrib.layers.l2_regularizer(scale=0.05))  # (D, K, K)
            fields_embeddings_sum = tf.reduce_sum(fields_embeddings, axis=1)    # (batch, K)
            p = tf.matmul(tf.expand_dims(fields_embeddings_sum, axis=2), tf.expand_dims(fields_embeddings_sum, axis=1))  # (batch, K, 1) *  (batch, 1, K) = (batch, K, K)
            for i in range(params["output_dimension"]): # 遍历每一个w, 生成lp的每一个分量
                wi = outer_product_w[i]  # (K, K), 只用上三角的元素, 构造成对称矩阵, 其余参数冗余, 待优化
                upper = tf.matrix_band_part(wi, 0, -1)
                wi = upper + tf.transpose(upper) - tf.matrix_band_part(wi, 0, 0)    # (K, K), 上三角+下三角-对角
                lp_i = tf.multiply(p, wi)  # (batch, K, K) multiply (K, K) = (batch, K, K)
                lp_i = tf.expand_dims(tf.reduce_sum(lp_i, axis=[1, 2]), axis=1)   # (batch, 1)
                product_output.append(lp_i)  # (batch, 1) * D

        lp = tf.concat(product_output, axis=1)  # (batch, D)

    # 偏置
    bias = tf.get_variable(name="bias", shape=(params["output_dimension"]), dtype=tf.float32)   # (D)

    # PNN层输出
    product_final = tf.nn.relu(lz + lp + bias)

    # 全连接层
    with tf.variable_scope("fcn"):
        net = product_final
        for unit in params["hidden_units"]:
            net = tf.layers.dense(net, unit, activation=tf.nn.relu)
            if "dropout_rate" in params and 0.0 < params["dropout_rate"] < 1.0:
                net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))
            if params["batch_norm"]:
                net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))

        logit = tf.layers.dense(net, 1)

    # -----定义PREDICT阶段行为-----
    prediction = tf.sigmoid(logit, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': prediction,
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    # -----定义完毕-----

    y = labels["read_comment"]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit), name="loss")

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
            "train_auc": auc[1],
            "lz": lz,
            "lp": lp,
        },
        every_n_iter=100
    )
    # timeline监控
    profiler_hook = tf.train.ProfilerHook(save_steps=1000, output_dir="./profiler", show_memory=True)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook, profiler_hook])
    # -----定义完毕-----


def main(unused_argv):
    """训练入口"""

    global total_feature_columns, label_feature_columns
    category_feature_columns, label_feature_columns = create_feature_columns()
    total_feature_columns = category_feature_columns


    params = {
            "category_feature_columns": total_feature_columns,
            "hidden_units": FLAGS.hidden_units.split(','),
            "dropout_rate": FLAGS.dropout_rate,
            "batch_norm": FLAGS.batch_norm,
            "learning_rate": FLAGS.learning_rate,
            "output_dimension": FLAGS.output_dimension,
            "product_method": FLAGS.product_method,
        }
    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=pnn_model_fn,
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

