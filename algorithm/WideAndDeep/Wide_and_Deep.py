<<<<<<< HEAD
"""
    [1] Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. ACM, 2016.
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
flags.DEFINE_float("wide_part_learning_rate", 0.005, "Wide part learning rate")
flags.DEFINE_float("deep_part_learning_rate", 0.001, "Deep part learning rate")
flags.DEFINE_string("deep_part_optimizer", "Adam",
                    "Wide part optimizer, supported strings are in {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0, "Dropout rate")

# 集群参数
# flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222",
#                     "Comma-separated list of hostname:port pairs")
# flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
#                     "Comma-separated list of hostname:port pairs")
# flags.DEFINE_string("job_name", None, "job name: worker or ps")
# flags.DEFINE_integer("task_index", None,
#                      "Worker task index, should be >= 0. task_index=0 is "
#                      "the master worker task the performs the variable "
#                      "initialization ")
# flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")

FLAGS = flags.FLAGS

global total_feature_columns


def create_feature_columns() -> Tuple[List[Any], List[Any]]:
    """

    Returns:
        wide_part_feature_columns (list): wide部分的feature_columns
        deep_part_feature_columns (list): deep部分的feature_columns
    """

    wide_part_feature_columns, deep_part_feature_columns = [], []

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

    deep_part_feature_columns += [videoplayseconds, u_read_comment_7d_sum, u_like_7d_sum, u_click_avatar_7d_sum,
                                  u_forward_7d_sum, u_comment_7d_sum, u_follow_7d_sum, u_favorite_7d_sum,
                                  i_read_comment_7d_sum, i_like_7d_sum, i_click_avatar_7d_sum, i_forward_7d_sum,
                                  i_comment_7d_sum, i_follow_7d_sum, i_favorite_7d_sum,
                                  c_user_author_read_comment_7d_sum]

    # 类别特征
    userid = fc.categorical_column_with_vocabulary_file('userid', os.path.join(FLAGS.vocabulary_dir, 'userid.txt'))
    feedid = fc.categorical_column_with_vocabulary_file('feedid', os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))
    device = fc.categorical_column_with_vocabulary_file('device', os.path.join(FLAGS.vocabulary_dir, 'device.txt'))
    authorid = fc.categorical_column_with_vocabulary_file('authorid', os.path.join(FLAGS.vocabulary_dir, 'authorid.txt'))
    bgm_song_id = fc.categorical_column_with_vocabulary_file('bgm_song_id', os.path.join(FLAGS.vocabulary_dir, 'bgm_song_id.txt'))
    bgm_singer_id = fc.categorical_column_with_vocabulary_file('bgm_singer_id',
                                                               os.path.join(FLAGS.vocabulary_dir, 'bgm_singer_id.txt'))

    manual_tag_list = fc.categorical_column_with_vocabulary_file('manual_tag_list',
                                                                 os.path.join(FLAGS.vocabulary_dir, 'manual_tag_id.txt'))
    his_read_comment_7d_seq = fc.categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
                                                                         os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))

    userid_emb = fc.embedding_column(userid, 16)
    feedid_emb = fc.shared_embedding_columns([feedid, his_read_comment_7d_seq], 16, combiner='mean')
    device_emb = fc.embedding_column(device, 2)
    authorid_emb = fc.embedding_column(authorid, 4)
    bgm_song_id_emb = fc.embedding_column(bgm_song_id, 4)
    bgm_singer_id_emb = fc.embedding_column(bgm_singer_id, 4)
    manual_tag_id_emb = fc.embedding_column(manual_tag_list, 4, combiner='mean')

    deep_part_feature_columns += [userid_emb, device_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb,
                                  manual_tag_id_emb]
    deep_part_feature_columns += feedid_emb # feedid_emb是list

    # 交叉特征
    cross_userid_manualtag = fc.crossed_column([userid, manual_tag_list], hash_bucket_size=100000)
    cross_userid_manualtag_indicator = fc.indicator_column(cross_userid_manualtag)

    wide_part_feature_columns += [cross_userid_manualtag_indicator]

    return wide_part_feature_columns, deep_part_feature_columns


def example_parser(serialized_example):
    """
        批量解析Example
    Args:
        serialized_example:

    Returns:
        features, labels
    """
    read_comment = fc.numeric_column("read_comment", default_value=0.0)
    fea_columns = total_feature_columns + [read_comment]

    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
    features = tf.parse_example(serialized_example, features=feature_spec)
    read_comment = features.pop("read_comment")

    return features, {"read_comment": read_comment}


def train_input_fn(filepath, example_parser, batch_size, num_epochs, shuffle_buffer_size):
    """
        wide&deep模型的input_fn
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
        wide&deep模型的eval阶段input_fn
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


def wide_and_deep_model_fn(features, labels, mode, params):
    """
        wide&deep模型的model_fn
    Args:
        features (dict): input_fn的第一个返回值, 模型输入样本特征
        labels (dict): input_fn的第二个返回值, 样本标签
        mode: tf.estimator.ModeKeys
        params (dict): 模型超参数

    Returns:
        tf.estimator.EstimatorSpec
    """

    # wide部分
    with tf.variable_scope("wide_part", reuse=tf.AUTO_REUSE):
        wide_input = fc.input_layer(features, params["wide_part_feature_columns"])
        wide_logit = tf.layers.dense(wide_input, 1, name="wide_part_variables")

    # deep部分
    with tf.variable_scope("deep_part"):
        deep_input = fc.input_layer(features, params["deep_part_feature_columns"])
        net = deep_input
        for unit in params["hidden_units"]:
            net = tf.layers.dense(net, unit, activation=tf.nn.relu)
            if "dropout_rate" in params and 0.0 < params["dropout_rate"] < 1.0:
                net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))
            if params["batch_norm"]:
                net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        deep_logit = tf.layers.dense(net, 1)

    # 总体logit
    total_logit = tf.add(wide_logit, deep_logit, name="total_logit")

    # -----定义PREDICT阶段行为-----
    prediction = tf.sigmoid(total_logit, name="prediction")
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
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=total_logit), name="loss")

    accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
    auc = tf.metrics.auc(labels=y, predictions=prediction)

    # -----定义EVAL阶段行为-----
    metrics = {"eval_accuracy": accuracy, "eval_auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # -----定义完毕-----

    wide_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_part')
    deep_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deep_part')

    # wide部分优化器op
    wide_part_optimizer = tf.train.FtrlOptimizer(learning_rate=params["wide_part_learning_rate"])
    wide_part_op = wide_part_optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(),
                                                var_list=wide_part_vars)

    # deep部分优化器op
    if params["deep_part_optimizer"] == 'Adam':
        deep_part_optimizer = tf.train.AdamOptimizer(learning_rate=params["deep_part_learning_rate"], beta1=0.9,
                                                     beta2=0.999, epsilon=1e-8)
    elif params["deep_part_optimizer"] == 'Adagrad':
        deep_part_optimizer = tf.train.AdagradOptimizer(learning_rate=params["deep_part_learning_rate"],
                                                        initial_accumulator_value=1e-8)
    elif params["deep_part_optimizer"] == 'RMSProp':
        params["deep_part_optimizer"] = tf.train.RMSPropOptimizer(learning_rate=params["deep_part_learning_rate"])
    elif params["deep_part_optimizer"] == 'ftrl':
        params["deep_part_optimizer"] = tf.train.FtrlOptimizer(learning_rate=params["deep_part_learning_rate"])
    elif params["deep_part_optimizer"] == 'SGD':
        deep_part_optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["deep_part_learning_rate"])
    deep_part_op = deep_part_optimizer.minimize(loss=loss, global_step=None, var_list=deep_part_vars)

    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.group(wide_part_op, deep_part_op)

    # -----定义TRAIN阶段行为-----
    assert mode == tf.estimator.ModeKeys.TRAIN
    # 待观测的变量
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if var.name == "wide_part/wide_part_variables/kernel:0":
            wide_part_dense_kernel = var
        if var.name == "wide_part/wide_part_variables/bias:0":
            wide_part_dense_bias = var

    # tensorboard收集
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_auc", auc[1])
    tf.summary.histogram("wide_part_dense_kernel", wide_part_dense_kernel)
    tf.summary.scalar("wide_part_dense_kernel_l2_norm", tf.norm(wide_part_dense_kernel))

    # 训练log打印
    log_hook = tf.train.LoggingTensorHook(
        {
            "train_loss": loss,
            "train_auc_0": auc[0],
            "train_auc_1": auc[1],
            # "wide_logit": wide_logit,
            # "deep_logit": deep_logit,
            # "wide_part_dense_kernel": wide_part_dense_kernel,
            # "wide_part_dense_bias": wide_part_dense_bias,
            # "wide_part_dense_kernel_l2_norm": tf.norm(wide_part_dense_kernel)
        },
        every_n_iter=100
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
    # -----定义完毕-----



def main(unused_argv):
    """训练入口"""

    wide_columns, deep_columns = create_feature_columns()
    global total_feature_columns
    total_feature_columns = wide_columns + deep_columns

    params = {
        "wide_part_feature_columns": wide_columns,
        "deep_part_feature_columns": deep_columns,
        'hidden_units': FLAGS.hidden_units.split(','),
        "dropout_rate": FLAGS.dropout_rate,
        "batch_norm": FLAGS.batch_norm,
        "deep_part_optimizer": FLAGS.deep_part_optimizer,
        "wide_part_learning_rate": FLAGS.wide_part_learning_rate,
        "deep_part_learning_rate": FLAGS.deep_part_learning_rate,
    }
    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=wide_and_deep_model_fn,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(filepath=FLAGS.train_data, example_parser=example_parser, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, shuffle_buffer_size=FLAGS.shuffle_buffer_size),
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
        input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser, batch_size=FLAGS.batch_size),
        throttle_secs=600,
        steps=None,
        exporters=exporters
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Evaluate Metrics.
    metrics = estimator.evaluate(input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser, batch_size=FLAGS.batch_size))
    for key in sorted(metrics):
        print('%s: %s' % (key, metrics[key]))

    # print("exporting model ...")
    # feature_spec = tf.feature_column.make_parse_example_spec(total_feature_columns)
    # print(feature_spec)
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    # estimator.export_savedmodel(FLAGS.output_dir, serving_input_receiver_fn)

    results = estimator.predict(input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser, batch_size=FLAGS.batch_size))
    predicts_df = pd.DataFrame.from_dict(results)
    predicts_df['probabilities'] = predicts_df['probabilities'].apply(lambda x: x[0])
    test_df = pd.read_csv("../../dataset/wechat_algo_data1/dataframe/test.csv")
    predicts_df['read_comment'] = test_df['read_comment']
    predicts_df.to_csv("predictions.csv")
    print("after evaluate")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
=======
"""
    [1] Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. ACM, 2016.
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
flags.DEFINE_float("wide_part_learning_rate", 0.005, "Wide part learning rate")
flags.DEFINE_float("deep_part_learning_rate", 0.001, "Deep part learning rate")
flags.DEFINE_string("deep_part_optimizer", "Adam",
                    "Wide part optimizer, supported strings are in {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")

# 集群参数
# flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222",
#                     "Comma-separated list of hostname:port pairs")
# flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
#                     "Comma-separated list of hostname:port pairs")
# flags.DEFINE_string("job_name", None, "job name: worker or ps")
# flags.DEFINE_integer("task_index", None,
#                      "Worker task index, should be >= 0. task_index=0 is "
#                      "the master worker task the performs the variable "
#                      "initialization ")
# flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")

FLAGS = flags.FLAGS

global total_feature_columns


def create_feature_columns() -> Tuple[List[Any], List[Any]]:
    """

    Returns:
        wide_part_feature_columns (list): wide部分的feature_columns
        deep_part_feature_columns (list): deep部分的feature_columns
    """

    wide_part_feature_columns, deep_part_feature_columns = [], []

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

    deep_part_feature_columns += [videoplayseconds, u_read_comment_7d_sum, u_like_7d_sum, u_click_avatar_7d_sum,
                                  u_forward_7d_sum, u_comment_7d_sum, u_follow_7d_sum, u_favorite_7d_sum,
                                  i_read_comment_7d_sum, i_like_7d_sum, i_click_avatar_7d_sum, i_forward_7d_sum,
                                  i_comment_7d_sum, i_follow_7d_sum, i_favorite_7d_sum,
                                  c_user_author_read_comment_7d_sum]

    # 类别特征
    userid = fc.categorical_column_with_vocabulary_file('userid', os.path.join(FLAGS.vocabulary_dir, 'userid.txt'))
    feedid = fc.categorical_column_with_vocabulary_file('feedid', os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))
    device = fc.categorical_column_with_vocabulary_file('device', os.path.join(FLAGS.vocabulary_dir, 'device.txt'))
    authorid = fc.categorical_column_with_vocabulary_file('authorid', os.path.join(FLAGS.vocabulary_dir, 'authorid.txt'))
    bgm_song_id = fc.categorical_column_with_vocabulary_file('bgm_song_id', os.path.join(FLAGS.vocabulary_dir, 'bgm_song_id.txt'))
    bgm_singer_id = fc.categorical_column_with_vocabulary_file('bgm_singer_id',
                                                               os.path.join(FLAGS.vocabulary_dir, 'bgm_singer_id.txt'))

    manual_tag_list = fc.categorical_column_with_vocabulary_file('manual_tag_list',
                                                                 os.path.join(FLAGS.vocabulary_dir, 'manual_tag_id.txt'))
    his_read_comment_7d_seq = fc.categorical_column_with_vocabulary_file('his_read_comment_7d_seq',
                                                                         os.path.join(FLAGS.vocabulary_dir, 'feedid.txt'))

    userid_emb = fc.embedding_column(userid, 16)
    feedid_emb = fc.shared_embedding_columns([feedid, his_read_comment_7d_seq], 16, combiner='mean')
    device_emb = fc.embedding_column(device, 2)
    authorid_emb = fc.embedding_column(authorid, 4)
    bgm_song_id_emb = fc.embedding_column(bgm_song_id, 4)
    bgm_singer_id_emb = fc.embedding_column(bgm_singer_id, 4)
    manual_tag_id_emb = fc.embedding_column(manual_tag_list, 4, combiner='mean')

    deep_part_feature_columns += [userid_emb, device_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb,
                                  manual_tag_id_emb]
    deep_part_feature_columns += feedid_emb # feedid_emb是list

    # 交叉特征
    cross_userid_manualtag = fc.crossed_column([userid, manual_tag_list], hash_bucket_size=100000)
    cross_userid_manualtag_indicator = fc.indicator_column(cross_userid_manualtag)

    wide_part_feature_columns += [cross_userid_manualtag_indicator]

    return wide_part_feature_columns, deep_part_feature_columns


def example_parser(serialized_example):
    """
        批量解析Example
    Args:
        serialized_example:

    Returns:
        features, labels
    """
    read_comment = fc.numeric_column("read_comment", default_value=0.0)
    fea_columns = total_feature_columns + [read_comment]

    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
    features = tf.parse_example(serialized_example, features=feature_spec)
    read_comment = features.pop("read_comment")

    return features, {"read_comment": read_comment}


def train_input_fn(filepath, example_parser, batch_size, num_epochs, shuffle_buffer_size):
    """
        wide&deep模型的input_fn
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
        wide&deep模型的eval阶段input_fn
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


def wide_and_deep_model_fn(features, labels, mode, params):
    """
        wide&deep模型的model_fn
    Args:
        features (dict): input_fn的第一个返回值, 模型输入样本特征
        labels (dict): input_fn的第二个返回值, 样本标签
        mode: tf.estimator.ModeKeys
        params (dict): 模型超参数

    Returns:
        tf.estimator.EstimatorSpec
    """

    # wide部分
    with tf.variable_scope("wide_part", reuse=tf.AUTO_REUSE):
        wide_input = fc.input_layer(features, params["wide_part_feature_columns"])
        wide_logit = tf.layers.dense(wide_input, 1, name="wide_part_variables")

    # deep部分
    with tf.variable_scope("deep_part"):
        deep_input = fc.input_layer(features, params["deep_part_feature_columns"])
        net = deep_input
        for unit in params["hidden_units"]:
            net = tf.layers.dense(net, unit, activation=tf.nn.relu)
            if "dropout_rate" in params and 0.0 < params["dropout_rate"] < 1.0:
                net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))
            if params["batch_norm"]:
                net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        deep_logit = tf.layers.dense(net, 1)

    # 总体logit
    total_logit = tf.add(wide_logit, deep_logit, name="total_logit")

    # -----定义PREDICT阶段行为-----
    prediction = tf.sigmoid(total_logit, name="prediction")
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
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=total_logit), name="loss")

    accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
    auc = tf.metrics.auc(labels=y, predictions=prediction)

    # -----定义EVAL阶段行为-----
    metrics = {"eval_accuracy": accuracy, "eval_auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # -----定义完毕-----

    wide_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_part')
    deep_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deep_part')

    # wide部分优化器op
    wide_part_optimizer = tf.train.FtrlOptimizer(learning_rate=params["wide_part_learning_rate"])
    wide_part_op = wide_part_optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(),
                                                var_list=wide_part_vars)

    # deep部分优化器op
    if params["deep_part_optimizer"] == 'Adam':
        deep_part_optimizer = tf.train.AdamOptimizer(learning_rate=params["deep_part_learning_rate"], beta1=0.9,
                                                     beta2=0.999, epsilon=1e-8)
    elif params["deep_part_optimizer"] == 'Adagrad':
        deep_part_optimizer = tf.train.AdagradOptimizer(learning_rate=params["deep_part_learning_rate"],
                                                        initial_accumulator_value=1e-8)
    elif params["deep_part_optimizer"] == 'RMSProp':
        deep_part_optimizer = tf.train.RMSPropOptimizer(learning_rate=params["deep_part_learning_rate"])
    elif params["deep_part_optimizer"] == 'ftrl':
        deep_part_optimizer = tf.train.FtrlOptimizer(learning_rate=params["deep_part_learning_rate"])
    elif params["deep_part_optimizer"] == 'SGD':
        deep_part_optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["deep_part_learning_rate"])
    deep_part_op = deep_part_optimizer.minimize(loss=loss, global_step=None, var_list=deep_part_vars)

    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.group(wide_part_op, deep_part_op)

    # -----定义TRAIN阶段行为-----
    assert mode == tf.estimator.ModeKeys.TRAIN
    # 待观测的变量
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if var.name == "wide_part/wide_part_variables/kernel:0":
            wide_part_dense_kernel = var
        if var.name == "wide_part/wide_part_variables/bias:0":
            wide_part_dense_bias = var

    # tensorboard收集
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_auc", auc[1])
    tf.summary.histogram("wide_part_dense_kernel", wide_part_dense_kernel)
    tf.summary.scalar("wide_part_dense_kernel_l2_norm", tf.norm(wide_part_dense_kernel))

    # 训练log打印
    log_hook = tf.train.LoggingTensorHook(
        {
            "train_loss": loss,
            "train_auc_0": auc[0],
            "train_auc_1": auc[1],
            # "wide_logit": wide_logit,
            # "deep_logit": deep_logit,
            # "wide_part_dense_kernel": wide_part_dense_kernel,
            # "wide_part_dense_bias": wide_part_dense_bias,
            # "wide_part_dense_kernel_l2_norm": tf.norm(wide_part_dense_kernel)
        },
        every_n_iter=100
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
    # -----定义完毕-----



def main(unused_argv):
    """训练入口"""

    wide_columns, deep_columns = create_feature_columns()
    global total_feature_columns
    total_feature_columns = wide_columns + deep_columns

    params = {
        "wide_part_feature_columns": wide_columns,
        "deep_part_feature_columns": deep_columns,
        'hidden_units': FLAGS.hidden_units.split(','),
        "dropout_rate": FLAGS.dropout_rate,
        "batch_norm": FLAGS.batch_norm,
        "deep_part_optimizer": FLAGS.deep_part_optimizer,
        "wide_part_learning_rate": FLAGS.wide_part_learning_rate,
        "deep_part_learning_rate": FLAGS.deep_part_learning_rate,
    }
    print(params)

    estimator = tf.estimator.Estimator(
        model_fn=wide_and_deep_model_fn,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(filepath=FLAGS.train_data, example_parser=example_parser, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs, shuffle_buffer_size=FLAGS.shuffle_buffer_size),
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
        input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser, batch_size=FLAGS.batch_size),
        throttle_secs=600,
        steps=None,
        exporters=exporters
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Evaluate Metrics.
    metrics = estimator.evaluate(input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser, batch_size=FLAGS.batch_size))
    for key in sorted(metrics):
        print('%s: %s' % (key, metrics[key]))

    # print("exporting model ...")
    # feature_spec = tf.feature_column.make_parse_example_spec(total_feature_columns)
    # print(feature_spec)
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    # estimator.export_savedmodel(FLAGS.output_dir, serving_input_receiver_fn)

    results = estimator.predict(input_fn=lambda: eval_input_fn(filepath=FLAGS.eval_data, example_parser=example_parser, batch_size=FLAGS.batch_size))
    predicts_df = pd.DataFrame.from_dict(results)
    predicts_df['probabilities'] = predicts_df['probabilities'].apply(lambda x: x[0])
    test_df = pd.read_csv("../../dataset/wechat_algo_data1/dataframe/test.csv")
    predicts_df['read_comment'] = test_df['read_comment']
    predicts_df.to_csv("predictions.csv")
    print("after evaluate")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
>>>>>>> 734986b93a9246f05fb1b15f98977242f436de04
