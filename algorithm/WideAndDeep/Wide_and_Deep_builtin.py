import os
from typing import List, Tuple, Any
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
# from algorithm.WideAndDeep.Wide_and_Deep import example_parser, train_input_fn, eval_input_fn, create_feature_columns, FLAGS



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
flags.DEFINE_string("deep_part_optimizer", "Adam",
                    "Wide part optimizer, supported strings are in {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", False, "Perform batch normalization (True or False)")

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
    # label不需要dict封装
    return features, read_comment


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



def main(unused_argv):
    """训练入口"""

    wide_columns, deep_columns = create_feature_columns()
    global total_feature_columns
    total_feature_columns = wide_columns + deep_columns

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=FLAGS.hidden_units.split(','),
        dnn_optimizer='Adam',
        loss_reduction='weighted_mean',
        batch_norm=True,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps))

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
    predicts_df.to_csv("predictions.csv")
    print("after evaluate")



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)