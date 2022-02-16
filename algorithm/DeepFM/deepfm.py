"""
    [1] Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." arXiv preprint arXiv:1703.04247 (2017).

    [2] Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE
"""

import os
from typing import List, Tuple, Any
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc

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
flags.DEFINE_integer("embedding_dim", 8, "Embedding dimension")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")

FLAGS = flags.FLAGS


def create_feature_columns() -> Tuple[list, list, list]:
    """

    Returns:
        first_order_feature_columns (list): fm部分一阶特征的feature_columns
        second_order_feature_columns (list): fm部分二阶特征的feature_columns
        label_feature_columns (list): label的feature_columns
    """

    first_order_feature_columns, second_order_feature_columns, label_feature_columns = [], [], []

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

    # manual_tag_list = fc.categorical_column_with_vocabulary_file('manual_tag_list',
    #                                                              os.path.join(FLAGS.vocabulary_dir, 'manual_tag_id.txt'))
    # his_read_comment_7d_seq = fc.categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
    #                                                                      os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))

    # FM一阶特征
    userid_one_hot = fc.indicator_column(userid)
    feedid_one_hot = fc.indicator_column(feedid)
    device_one_hot = fc.indicator_column(device)
    authorid_one_hot = fc.indicator_column(authorid)
    bgm_song_id_one_hot = fc.indicator_column(bgm_song_id)
    bgm_singer_id_one_hot = fc.indicator_column(bgm_singer_id)

    first_order_feature_columns += [userid_one_hot, feedid_one_hot, device_one_hot, authorid_one_hot,
                                    bgm_song_id_one_hot, bgm_singer_id_one_hot]

    # FM二阶特征&deep部分特征
    userid_emb = fc.embedding_column(userid, FLAGS.embedding_dim)
    feedid_emb = fc.embedding_column(feedid, FLAGS.embedding_dim)
    # feedid_emb = fc.shared_embedding_columns([feedid, his_read_comment_7d_seq], 16, combiner='mean')
    device_emb = fc.embedding_column(device, FLAGS.embedding_dim)
    authorid_emb = fc.embedding_column(authorid, FLAGS.embedding_dim)
    bgm_song_id_emb = fc.embedding_column(bgm_song_id, FLAGS.embedding_dim)
    bgm_singer_id_emb = fc.embedding_column(bgm_singer_id, FLAGS.embedding_dim)
    # manual_tag_id_emb = fc.embedding_column(manual_tag_list, 4, combiner='mean')

    second_order_feature_columns += [userid_emb, feedid_emb, device_emb, authorid_emb, bgm_song_id_emb,
                                     bgm_singer_id_emb]

    # label
    read_comment = fc.numeric_column("read_comment", default_value=0.0)
    label_feature_columns += [read_comment]

    return first_order_feature_columns, second_order_feature_columns, label_feature_columns


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


def train_input_fn(filepath, example_parser, batch_size, num_epochs, shuffle_buffer_size):
    """
        deepfm模型的input_fn
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
        deepfm模型的eval阶段input_fn
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


def deepfm_model_fn(features, labels, mode, params):
    """
        deepfm模型的model_fn
    Args:
        features (dict): input_fn的第一个返回值, 模型输入样本特征
        labels (dict): input_fn的第二个返回值, 样本标签
        mode: tf.estimator.ModeKeys
        params (dict): 模型超参数

    Returns:
        tf.estimator.EstimatorSpec
    """

    # fm一阶部分
    with tf.variable_scope("fm_first_order"):
        fm_first_order_input = fc.input_layer(features, params["first_order_feature_columns"])
        fm_first_order_logit = tf.layers.dense(fm_first_order_input, 1, name="fm_first_order_dense")  # (batch, 1)

    # 将每个类别特征的embedding取出
    fields_embeddings = []
    # 将每个类别特征的embedding取出, 再做element-wise的平方操作
    fields_squared_embeddings = []
    for cat_feature_column in params["second_order_feature_columns"]:
        embed_input = fc.input_layer(features, [cat_feature_column])  # (batch, K)
        fields_embeddings.append(embed_input)
        fields_squared_embeddings.append(tf.square(embed_input))
    # fm二阶部分
    with tf.variable_scope('fm_second_order'):
        # 先加再element-wise平方, 对应FM化简公式的第一项被减数
        sum_embedding_then_square = tf.square(tf.add_n(fields_embeddings))  # (batch, K)
        # 先element-wise平方再加, 对应FM化简公式的第二项减数
        square_embedding_then_sum = tf.add_n(fields_squared_embeddings)  # (batch, K)

        fm_second_order_logit = tf.reduce_sum(0.5 * (sum_embedding_then_square - square_embedding_then_sum),
                                              axis=1,
                                              keepdims=True)  # (batch, 1)

    # deep部分
    with tf.variable_scope('fm_deep'):
        deep_input = tf.concat(fields_embeddings, axis=1)  # (batch, F*K)
        net = deep_input
        for unit in params["hidden_units"]:
            net = tf.layers.dense(net, unit, activation=tf.nn.relu)
            if "dropout_rate" in params and 0.0 < params["dropout_rate"] < 1.0:
                net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))
            if params["batch_norm"]:
                net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        deep_logit = tf.layers.dense(net, 1)  # (batch, 1)

    total_logit = tf.add_n([fm_first_order_logit, fm_second_order_logit, deep_logit])  # (batch, 1)

    # -----定义PREDICT阶段行为-----
    prediction = tf.sigmoid(total_logit, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': prediction,
            'fm_first_order_logit': fm_first_order_logit,
            'fm_second_order_logit': fm_second_order_logit,
            'deep_logit': deep_logit,
            # 'deep_input': deep_input,
            # 'deep_part_final_output': net

        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
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
            "train_auc": auc[1],
            # "fm_first_order_logit": fm_first_order_logit,
            # "fm_second_order_logit": fm_second_order_logit,
            "deep_logit": deep_logit,
            'deep_input': deep_input,
            'deep_part_final_output': net
        },
        every_n_iter=100
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
    # -----定义完毕-----


def main(unused_argv):
    """训练入口"""

    global total_feature_columns, label_feature_columns
    first_order_feature_columns, second_order_feature_columns, label_feature_columns = create_feature_columns()
    total_feature_columns = first_order_feature_columns + second_order_feature_columns #+ label_feature_columns

    params = {
            "first_order_feature_columns": first_order_feature_columns,
            "second_order_feature_columns": second_order_feature_columns,
            'hidden_units': FLAGS.hidden_units.split(','),
            "dropout_rate": FLAGS.dropout_rate,
            "batch_norm": FLAGS.batch_norm,
            "learning_rate": FLAGS.learning_rate,
        }
    print(params)
    print(FLAGS.embedding_dim, FLAGS.num_epochs)

    estimator = tf.estimator.Estimator(
        model_fn=deepfm_model_fn,
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
