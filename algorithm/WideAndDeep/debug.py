import tensorflow as tf
from tensorflow import feature_column as fc
from typing import List, Tuple, Any
import os

flags = tf.app.flags
flags.DEFINE_string("vocabulary_dir", "../../dataset/wechat_algo_data1/vocabulary/", "Folder where the vocabulary file is stored")
FLAGS = flags.FLAGS


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


def data_input_fn(filepath, example_parser, batch_size, num_epochs, shuffle_buffer_size):
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


def main(_):
    wide_columns, deep_columns = create_feature_columns()
    global total_feature_columns
    total_feature_columns = wide_columns + deep_columns

    dataset = data_input_fn(filepath="../../dataset/wechat_algo_data1/tfrecord/train.tfrecord",
                            example_parser=example_parser,
                            batch_size=512,
                            num_epochs=1,
                            shuffle_buffer_size=1000)
    iter = dataset.make_one_shot_iterator()
    element = iter.get_next()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        print(sess.run(element))

def main1(_):
    wide_columns, deep_columns = create_feature_columns()
    global total_feature_columns
    total_feature_columns = wide_columns + deep_columns
    dataset = tf.data.TFRecordDataset("../../dataset/wechat_algo_data1/tfrecord/train.tfrecord")

    dataset = dataset.repeat(1)
    dataset = dataset.batch(512)
    dataset = dataset.map(example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(1)

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
