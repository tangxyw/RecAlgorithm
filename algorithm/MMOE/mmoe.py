"""
    [1]  Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018.
         Modeling task relationships in multi-task learning with multi-gate mixture-ofexperts.
         In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1930–1939.
"""

import os
from typing import List, Tuple, Any
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
from tower_layer import tower_layer

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
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the final output part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")
flags.DEFINE_integer("num_experts", 3, "Number of experts")
flags.DEFINE_integer("expert_hidden_units", 512, "Expert module output dimension")
flags.DEFINE_integer("num_tasks", 3, "Number of tasks, that's number of gates")
flags.DEFINE_string("task_names", "read_comment,like,click_avatar", "Comma-separated list of task names, each must be in keys of tfrecord file")

FLAGS = flags.FLAGS


def create_feature_columns() -> Tuple[list, list, list]:
    """
        生成MMOE模型输入特征和label
    Returns:
        dense_feature_columns (list): 连续特征的feature_columns
        category_feature_columns (list): 类别特征的feature_columns(包括序列特征)
        label_feature_columns (list): 因变量的feature_columns
    """

    dense_feature_columns, category_feature_columns, label_feature_columns = [], [], []

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

    userid_emb = fc.embedding_column(userid, 16)
    feedid_emb = fc.shared_embedding_columns([feedid, his_read_comment_7d_seq], 16, combiner='mean')
    device_emb = fc.embedding_column(device, 2)
    authorid_emb = fc.embedding_column(authorid, 4)
    bgm_song_id_emb = fc.embedding_column(bgm_song_id, 4)
    bgm_singer_id_emb = fc.embedding_column(bgm_singer_id, 4)
    manual_tag_id_emb = fc.embedding_column(manual_tag_list, 4, combiner='mean')

    category_feature_columns += [userid_emb, device_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb,
                                 manual_tag_id_emb]
    category_feature_columns += feedid_emb  # feedid_emb是list

    # label
    label_feature_columns += [fc.numeric_column(task_name, default_value=0.0) for task_name in FLAGS.task_names.split(",")]

    return dense_feature_columns, category_feature_columns, label_feature_columns


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
    labels = {task_name: features.pop(task_name) for task_name in FLAGS.task_names.split(",")}
    return features, labels


def train_input_fn(filepath, example_parser, batch_size, num_epochs, shuffle_buffer_size):
    """
        mmoe模型的input_fn
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
        mmoe模型的eval阶段input_fn
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


def mmoe_model_fn(features, labels, mode, params):
    """
        mmoe模型的model_fn
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

    # concat all
    concat_all_input = tf.concat([dense_input, category_input], axis=-1)

    # experts
    with tf.variable_scope("experts"):
        # 专家网络输出列表
        experts = [tf.layers.dense(concat_all_input,
                                   params["expert_hidden_units"],
                                   activation=tf.nn.relu,
                                   name=f"expert_{i}") for i in range(params["num_experts"])]
        # [B, expert_hidden_units] * num_experts
        experts = [e[:, tf.newaxis, :] for e in experts]    # (B, 1, expert_hidden_units) * num_experts
        experts = tf.concat(experts, axis=1)    # (B, num_experts, expert_hidden_units)

    with tf.variable_scope("gates"):
        # 门输出列表
        gates = [tf.layers.dense(concat_all_input,
                                 params["num_experts"],
                                 activation=tf.nn.softmax,
                                 use_bias=False,    # 论文中省略了bias
                                 name=f"gate_{i}") for i in range(params["num_tasks"])]
        # [B, num_experts] * num_tasks

    with tf.variable_scope("tower"):
        # 任务塔列表
        towers = []
        for i in range(params["num_tasks"]):
            # 每一个门输出都要增加一个维度, 以便做矩阵乘法
            gate = tf.expand_dims(gates[i], axis=-1)    # (B, num_experts, 1)
            # 专家网络输出分别和每一个门输出做矩阵乘法
            tower = tf.matmul(experts, gate, transpose_a=True)  # (B, expert_hidden_units, num_experts) * (B, num_experts, 1) = (B, expert_hidden_units, 1)
            tower = tf.squeeze(tower, axis=-1)  # (B, expert_hidden_units)
            towers.append(tower)    # (B, expert_hidden_units) * num_tasks

        # 任务名列表
        # task_names = list(labels.keys()) 这么写在导出saved_model时会报错
        task_names = params["task_names"]
        logit_list = [tower_layer(x,
                                  params["hidden_units"],
                                  mode,
                                  params["batch_norm"],
                                  params["dropout_rate"],
                                  task_name) for x, task_name in zip(towers, task_names)]
        # (B, 1) * num_tasks

    # -----定义PREDICT阶段行为-----
    prediction_list = [tf.sigmoid(logit, name=f"prediction_{task_name}") for logit, task_name in zip(logit_list, task_names)]
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            f"{task_name}_probabilities": prediction for task_name, prediction in zip(task_names, prediction_list)
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    # -----定义完毕-----

    losses = [tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[task_name], logits=logit), name=f"loss_{task_name}")
              for logit, task_name in zip(logit_list, task_names)]
    total_loss = tf.add_n(losses)

    accuracy_list = [tf.metrics.accuracy(labels=labels[task_name], predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
                     for task_name, prediction in zip(task_names, prediction_list)]
    auc_list = [tf.metrics.auc(labels=labels[task_name], predictions=prediction)
                for task_name, prediction in zip(task_names, prediction_list)]

    # -----定义EVAL阶段行为-----
    auc_metrics = {f"eval_{task_name}_auc": auc for task_name, auc in zip(task_names, auc_list)}
    accuracy_metrics = {f"eval_{task_name}_accuracy": accuracy for task_name, accuracy in zip(task_names, accuracy_list)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops={**accuracy_metrics, **auc_metrics})
    # -----定义完毕-----

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=0.9,
                                       beta2=0.999, epsilon=1e-8)
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=total_loss, global_step=tf.train.get_global_step())

    # -----定义TRAIN阶段行为-----
    assert mode == tf.estimator.ModeKeys.TRAIN

    # tensorboard收集
    for task_name, auc in zip(task_names, auc_list):
        tf.summary.scalar(f"train_{task_name}_auc", auc[1])
    for task_name, accuracy in zip(task_names, accuracy_list):
        tf.summary.scalar(f"train_{task_name}_accuracy", accuracy[1])

    # 训练log打印
    # 观测loss
    loss_log = {f"train_{task_name}_loss": loss for task_name, loss in zip(task_names, losses)}
    # 观测训练auc
    auc_log = {f"train_{task_name}_auc": auc[1] for task_name, auc in zip(task_names, auc_list)}
    # 观测gate输出
    gate_log = {f"{task_name}_gate_expert_weight": gate for task_name, gate, in zip(task_names, gates)}

    loss_log_hook = tf.train.LoggingTensorHook(
        loss_log,
        every_n_iter=100
    )
    auc_log_hook = tf.train.LoggingTensorHook(
        auc_log,
        every_n_iter=100
    )
    gate_log_hook = tf.train.LoggingTensorHook(
        gate_log,
        every_n_iter=100
    )
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op,
                                      training_hooks=[loss_log_hook, auc_log_hook, gate_log_hook])
    # -----定义完毕-----


def main(unused_argv):
    """训练入口"""

    global total_feature_columns, label_feature_columns
    dense_feature_columns, category_feature_columns, label_feature_columns = create_feature_columns()
    total_feature_columns = dense_feature_columns + category_feature_columns

    params = {
                 "dense_feature_columns": dense_feature_columns,
                 "category_feature_columns": category_feature_columns,
                 "hidden_units": FLAGS.hidden_units.split(','),
                 "dropout_rate": FLAGS.dropout_rate,
                 "batch_norm": FLAGS.batch_norm,
                 "learning_rate": FLAGS.learning_rate,
                 "num_experts": FLAGS.num_experts,
                 "num_tasks": FLAGS.num_tasks,
                 "expert_hidden_units": FLAGS.expert_hidden_units,
                 "task_names": FLAGS.task_names.split(','),
             }
    print(params)
    # 任务数要和任务名列表长度一致
    assert params["num_tasks"] == len(params["task_names"]), "num_tasks must equals length of task_names"

    estimator = tf.estimator.Estimator(
        model_fn=mmoe_model_fn,
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
    for task_name in params["task_names"]:
        predicts_df[f"{task_name}_probabilities"] = predicts_df[f"{task_name}_probabilities"].apply(lambda x: x[0])
    test_df = pd.read_csv("../../dataset/wechat_algo_data1/dataframe/test.csv")
    for task_name in params["task_names"]:
        predicts_df[task_name] = test_df[task_name]
    predicts_df.to_csv("predictions.csv")
    print("after evaluate")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
