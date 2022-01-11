"""
    生成微信视频号比赛训练集/测试集的tfrecord文件

    训练集: date_ = 8-13(生成特征需要开7天的窗口)
    测试集: date_ = 14

    特征：
        user侧：
            userid: 用户id
            u_read_comment_7d_sum: 近7天查看评论次数
            u_like_7d_sum: 近7天点赞次数
            u_click_avatar_7d_sum: 近7天点击头像次数
            u_favorite_7d_sum: 近7天收藏次数
            u_forward_7d_sum: 近7天转发次数
            u_comment_7d_sum: 近7天评论次数
            u_follow_7d_sum: 近7天关注次数
            his_read_comment_7d_seq: 近7天查看评论序列, 最长50个
            device: 设备类型


        item侧:
            feedid: feedid
            i_read_comment_7d_sum: 近7天被查看评论次数
            i_like_7d_sum: 近7天被点赞次数
            i_click_avatar_7d_sum: 近7天被点击头像次数
            i_favorite_7d_sum: 近7天被收藏次数
            i_forward_7d_sum: 近7天被转发次数
            i_comment_7d_sum: 近7天被评论次数
            i_follow_7d_sum: 近7天经由此feedid, 作者被关注次数
            videoplayseconds: feed时长
            authorid: 作者id
            bgm_song_id: 背景音乐id
            bgm_singer_id: 背景音乐歌手id
            manual_tag_list: 人工标注的分类标签


        交叉侧:(过于稀疏且耗费资源, 暂时只考虑第一个)
            c_user_author_read_comment_7d_sum:  user对当前item作者的查看评论次数
            c_user_author_like_7d_sum:  user对当前item作者的点赞次数
            c_user_author_click_avatar_7d_sum:  user对当前item作者的点击头像次数
            c_user_author_favorite_7d_sum:  user对当前item作者的收藏次数
            c_user_author_forward_7d_sum:  user对当前item作者的转发次数
            c_user_author_comment_7d_sum:  user对当前item作者的评论次数
            c_user_author_follow_7d_sum:  user对当前item作者的关注次数
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
from collections import Counter

ACTION_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
END_DAY = 14


class DataGenerator:
    """生成微信视频号训练集/测试集的tfrecord文件"""

    def __init__(self, dataset_dir: str = './', out_path: str = './'):
        """
        Args:
            dataset_dir: 数据文件所在文件夹路径
            out_path: tfrecord文件以及类别特征vocabulary文件输出文件夹路径
        """

        self.dataset_dir = dataset_dir
        self.out_path = out_path
        
        self.dense_features = [
            "videoplayseconds",
            "u_read_comment_7d_sum",
            "u_like_7d_sum",
            "u_click_avatar_7d_sum",
            "u_forward_7d_sum",
            "u_comment_7d_sum",
            "u_follow_7d_sum",
            "u_favorite_7d_sum",
            "i_read_comment_7d_sum",
            "i_like_7d_sum",
            "i_click_avatar_7d_sum",
            "i_forward_7d_sum",
            "i_comment_7d_sum",
            "i_follow_7d_sum",
            "i_favorite_7d_sum",
            "c_user_author_read_comment_7d_sum",
        ]
        self.category_featurs = [
            "userid",
            "feedid",
            "device",
            "authorid",
            "bgm_song_id",
            "bgm_singer_id",
        ]
        self.seq_features = ["his_read_comment_7d_seq", "manual_tag_list"]
        self.labels = [
            "read_comment",
            "comment",
            "like",
            "click_avatar",
            "forward",
            "follow",
            "favorite",
        ]

        # 创建存放vocabulary文件的文件夹
        self.vocab_dir = os.path.join(self.out_path, 'vocabulary')
        # 创建存放features分片的文件夹
        self.features_dir = os.path.join(self.out_path, 'features')
        # 创建存放dataframe的文件夹
        self.dataframe_dir = os.path.join(self.out_path, 'dataframe')
        # 创建存放dataframe的文件夹
        self.tfrecord_dir = os.path.join(self.out_path, 'tfrecord')

        if not os.path.exists(os.path.join(self.dataframe_dir, 'DATAFRAME_ALREADY')):
            self._load_data()
            self._preprocess()

        self._generate_vocabulary_file()
        self._generate_features()
        self._generate_dataframe()
        self._generate_tfrecord()

    def _load_data(self):
        """读入数据"""

        self.user_action = pd.read_csv(self.dataset_dir + 'user_action.csv')
        self.feed_info = pd.read_csv(self.dataset_dir + 'feed_info.csv',
                                     usecols=["feedid", "authorid", "videoplayseconds", "bgm_song_id", "bgm_singer_id",
                                              "manual_tag_list"])

    def _preprocess(self):
        """数据预处理，把所有类别变量取值前面都加上前缀"""

        self.feed_info['feedid'] = self.feed_info['feedid'].astype(str)
        self.feed_info['authorid'] = self.feed_info['authorid'].astype(str)
        # int型column中有空值存在的情况下, pd.read_csv后会被cast成float, 需要用扩展类型代替int
        self.feed_info['bgm_song_id'] = self.feed_info['bgm_song_id'].astype(pd.Int16Dtype()).astype(str)
        self.feed_info['bgm_singer_id'] = self.feed_info['bgm_singer_id'].astype(pd.Int16Dtype()).astype(str)

        for index, row in tqdm(self.feed_info.iterrows(), total=self.feed_info.shape[0]):
            self.feed_info.at[index, 'feedid'] = 'feedid_' + row['feedid']
            self.feed_info.at[index, 'authorid'] = 'authorid_' + row['authorid']
            self.feed_info.at[index, 'bgm_song_id'] = 'bgm_song_id_' + row['bgm_song_id'] if row[
                                                                                                 'bgm_song_id'] != '<NA>' else np.nan
            self.feed_info.at[index, 'bgm_singer_id'] = 'bgm_singer_id_' + row['bgm_singer_id'] if row[
                                                                                                       'bgm_singer_id'] != '<NA>' else np.nan
            self.feed_info.at[index, 'manual_tag_list'] = ['manual_tag_id_' + tag for tag in
                                                           row['manual_tag_list'].split(';')] if row[
                                                                                                     'manual_tag_list'] is not np.nan else np.nan

        self.user_action['userid'] = 'userid_' + self.user_action['userid'].astype(str)
        self.user_action['feedid'] = 'feedid_' + self.user_action['feedid'].astype(str)
        self.user_action['device'] = 'device_' + self.user_action['device'].astype(str)

    def _generate_vocabulary_file(self):
        """
            生成所有类别特征的vocabulary文件(txt格式)
            userid, feedid, device, authorid, bgm_song_id, bgm_singer_id, manual_tag_id
        """

        # 创建存放vocabulary文件的文件夹
        # self.vocab_dir = os.path.join(self.out_path, 'vocabulary')
        if not os.path.exists(self.vocab_dir):
            os.mkdir(self.vocab_dir)
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.vocab_dir, 'VOCAB_FILE_ALREADY')):
            print("Vocabulary files ready!")
            return

        # 每个分类变量对应一个Counter对象
        vocabulary_dict = {}

        # user_id, device
        action_scope = self.user_action[self.user_action['date_'].between(8, 14)]
        user_vocab = Counter(action_scope['userid'])
        device_vocab = Counter(action_scope['device'])
        vocabulary_dict['userid'] = user_vocab
        vocabulary_dict['device'] = device_vocab

        # feedid, authorid, bgm_song_id, bgm_singer_id
        feedid_vocab = Counter(self.feed_info['feedid'])
        authorid_vocab = Counter(self.feed_info['authorid'])
        # bgm_song_id_vocab和bgm_singer_id_vocab有空值, 需要处理Counter
        bgm_song_id_vocab = Counter(self.feed_info['bgm_song_id'])
        bgm_song_id_vocab.pop(np.nan)
        bgm_singer_id_vocab = Counter(self.feed_info['bgm_singer_id'])
        bgm_singer_id_vocab.pop(np.nan)
        vocabulary_dict['feedid'] = feedid_vocab
        vocabulary_dict['authorid'] = authorid_vocab
        vocabulary_dict['bgm_song_id'] = bgm_song_id_vocab
        vocabulary_dict['bgm_singer_id'] = bgm_singer_id_vocab

        # manual_tag_list
        manual_tag_id_vacab = Counter()
        for index, row in tqdm(self.feed_info.iterrows(), total=self.feed_info.shape[0]):
            if self.feed_info.at[index, 'manual_tag_list'] is not np.nan:
                manual_tag_id_vacab += Counter(self.feed_info.at[index, 'manual_tag_list'])
        vocabulary_dict['manual_tag_id'] = manual_tag_id_vacab

        for variable_name, vocab in vocabulary_dict.items():
            vocabulary_file_path = os.path.join(self.vocab_dir, variable_name + '.txt')
            with open(vocabulary_file_path, 'w') as f:
                for key, _ in vocab.items():
                    f.write(key + '\n')

        # 生成完成标识
        with open(os.path.join(self.vocab_dir, 'VOCAB_FILE_ALREADY'), 'w'):
            pass

    def _generate_features(self, start_day: int = 1, window_size: int = 7):
        """
        生成user侧, item侧, 交叉侧特征
        Args:
            start_day:  从第几天开始构建特征, 默认从第1天开始
            window_size:  特征窗口大小, 默认为7
        """

        # 创建存放features分片的文件夹
        # self.features_dir = os.path.join(self.out_path, 'features')
        if not os.path.exists(self.features_dir):
            os.mkdir(self.features_dir)
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.features_dir, 'FEATURES_PKL_ALREADY')):
            print("Features pkl ready!")
            return

        # user侧
        user_features_data = self.user_action[["userid", "date_"] + ACTION_COLUMN_LIST]
        # 1. 统计特征
        user_arr = []
        # start_day = 1, END_DAY - window_size + 1 = 14 - 7 + 1 = 8
        # 即start的范围为1-7, 对应的date_的范围为8-14: 8:1→7 9:2→8 ... 14:7→13
        for start in range(start_day, END_DAY - window_size + 1):
            # 需要聚合的数据范围
            temp = user_features_data[
                (user_features_data['date_'] >= start) & (user_features_data['date_'] < (start + window_size))]
            # date_列要重新生成
            temp.drop(columns=['date_'], inplace=True)
            # 聚合
            temp = temp.groupby(['userid']).agg(['sum']).reset_index()
            # 结果数据的列名
            new_column_names = []
            for col, agg_name in temp.columns.values:
                if col == 'userid':
                    new_column_names.append('userid')
                else:
                    new_column_names.append('u_' + col + '_7d_' + agg_name)
            temp.columns = new_column_names
            # 重新生成date_列
            temp['date_'] = start + window_size
            user_arr.append(temp)
        user_agg_features = pd.concat(user_arr, ignore_index=True)
        # user_agg_features.to_csv(os.path.join(self.out_path, 'user_agg_features.csv'), index=False)
        user_agg_features.to_pickle(os.path.join(self.features_dir, 'user_agg_features.pkl'))

        # 2. 历史序列特征
        user_arr = []
        user_features_data = self.user_action[["userid", "feedid", "date_"] + ACTION_COLUMN_LIST]
        # 基本逻辑同统计特征, 独立拆分出来是为了逻辑清晰
        for start in range(start_day, END_DAY - window_size + 1):
            temp = user_features_data[(user_features_data['date_'] >= start) &
                                      (user_features_data['date_'] < (start + window_size)) &
                                      (user_features_data['read_comment'] == 1)]
            temp.drop(columns=['date_'], inplace=True)
            temp = temp.groupby(['userid']).agg(
                his_read_comment_7d_seq=pd.NamedAgg(column="feedid", aggfunc=list)).reset_index()
            # 只取后50个元素
            temp['his_read_comment_7d_seq'] = temp.apply(
                lambda row: row.his_read_comment_7d_seq[len(row.his_read_comment_7d_seq) - 50:] if len(
                    row.his_read_comment_7d_seq) > 50 else row.his_read_comment_7d_seq, axis=1)
            temp['date_'] = start + window_size
            user_arr.append(temp)
        user_seq_features = pd.concat(user_arr, ignore_index=True)
        # user_seq_features.to_csv(os.path.join(self.out_path, 'user_seq_features.csv'), index=False)
        user_seq_features.to_pickle(os.path.join(self.features_dir, 'user_seq_features.pkl'))

        # item侧
        feed_features_data = self.user_action[["feedid", "date_"] + ACTION_COLUMN_LIST]
        # 1. 统计特征
        feed_arr = []
        for start in range(start_day, END_DAY - window_size + 1):
            # 需要聚合的数据范围
            temp = feed_features_data[
                (feed_features_data['date_'] >= start) & (feed_features_data['date_'] < (start + window_size))]
            # date_列要重新生成
            temp.drop(columns=['date_'], inplace=True)
            # 聚合
            temp = temp.groupby(['feedid']).agg(['sum']).reset_index()
            # 结果数据的列名
            new_column_names = []
            for col, agg_name in temp.columns.values:
                if col == 'feedid':
                    new_column_names.append('feedid')
                else:
                    new_column_names.append('i_' + col + '_7d_' + agg_name)
            temp.columns = new_column_names
            # 重新生成date_列
            temp['date_'] = start + window_size
            feed_arr.append(temp)
        feed_agg_features = pd.concat(feed_arr, ignore_index=True)
        feed_agg_features.to_pickle(os.path.join(self.features_dir, 'feed_agg_features.pkl'))

        # 交叉侧
        # cross_features_data = self.user_action.join(self.feed_info, on='feedid', how='left', rsuffix='_')[["userid", "authorid", "date_"] + ACTION_COLUMN_LIST]
        cross_features_data = \
            pd.merge(self.user_action[["userid", "feedid", "date_"] + ACTION_COLUMN_LIST], self.feed_info, on="feedid",
                     how="left")[["userid", "authorid", "date_"] + ['read_comment']]
        # 1. 统计特征
        cross_arr = []
        for start in range(start_day, END_DAY - window_size + 1):
            temp = cross_features_data[
                (cross_features_data['date_'] >= start) & (cross_features_data['date_'] < (start + window_size))]
            temp.drop(columns=['date_'], inplace=True)
            # 聚合
            temp = temp.groupby(["userid", "authorid"]).agg(['sum']).reset_index()
            # 结果数据的列名
            new_column_names = []
            for col, agg_name in temp.columns.values:
                if col == 'userid' or col == "authorid":
                    new_column_names.append(col)
                else:
                    new_column_names.append('c_user_author_' + col + '_7d_' + agg_name)
            temp.columns = new_column_names
            # 只保留大于0的行, 节省空间资源
            temp = temp[temp['c_user_author_read_comment_7d_sum'] > 0]
            # 重新生成date_列
            temp['date_'] = start + window_size
            cross_arr.append(temp)

        cross_agg_features = pd.concat(cross_arr, ignore_index=True)
        # cross_agg_features.to_pickle(os.path.join(self.out_path, 'cross_agg_features.csv'), index=False)
        cross_agg_features.to_pickle(os.path.join(self.features_dir, 'cross_agg_features.pkl'))

        # 生成完成标识
        with open(os.path.join(self.features_dir, 'FEATURES_PKL_ALREADY'), 'w'):
            pass

    def _generate_dataframe(self):
        """生成样本表"""

        # 创建存放dataframe的文件夹
        # self.dataframe_dir = os.path.join(self.out_path, 'dataframe')
        if not os.path.exists(self.dataframe_dir):
            os.mkdir(self.dataframe_dir)
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.dataframe_dir, 'DATAFRAME_ALREADY')):
            print("DataFrame ready!")
            return

        self.user_action = self.user_action[self.user_action['date_'].between(8, 14)]
        user_agg_features = pd.read_pickle(os.path.join(self.features_dir, 'user_agg_features.pkl'))
        user_seq_features = pd.read_pickle(os.path.join(self.features_dir, 'user_seq_features.pkl'))
        feed_agg_features = pd.read_pickle(os.path.join(self.features_dir, 'feed_agg_features.pkl'))
        cross_agg_features = pd.read_pickle(os.path.join(self.features_dir, 'cross_agg_features.pkl'))

        self.user_action = pd.merge(self.user_action, self.feed_info, on=['feedid'], how='left')
        self.user_action = pd.merge(self.user_action, user_agg_features, on=['userid', 'date_'], how='left')
        self.user_action = pd.merge(self.user_action, user_seq_features, on=['userid', 'date_'], how='left')
        self.user_action = pd.merge(self.user_action, feed_agg_features, on=['feedid', 'date_'], how='left')
        self.user_action = pd.merge(self.user_action, cross_agg_features, on=['userid', 'authorid', 'date_'],
                                    how='left')


        # # debug
        # self.user_action = self.user_action.iloc[:100]

        # 填补空值
        for index, row in tqdm(self.user_action.iterrows(), total=self.user_action.shape[0]):
            for col in self.dense_features:
                self.user_action.at[index, col] = np.log(row[col] + 1) if not pd.isna(
                    row[col]) else 0  # 空值有很多种情况, 用pd.isna统一判断
            for col in self.seq_features:
                self.user_action.at[index, col] = ','.join(row[col]) if row[col] is not np.nan else row[
                    col]  # pd.isna是array-like的, 不能用在这里

        # self.user_action.to_pickle(os.path.join(self.dataframe_dir, 'data.pkl'), compression='bz2')
        self.user_action[self.user_action['date_'].between(8, 13)].to_csv(os.path.join(self.dataframe_dir, 'train.csv'))
        self.user_action[self.user_action['date_'] == 14].to_csv(os.path.join(self.dataframe_dir, 'test.csv'))

        # 生成完成标识
        with open(os.path.join(self.dataframe_dir, 'DATAFRAME_ALREADY'), 'w'):
            pass

    def _generate_tfrecord(self):
        """生成训练集和测试集的tfrecord"""

        if not os.path.exists(self.tfrecord_dir):
            os.mkdir(self.tfrecord_dir)
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.tfrecord_dir, 'TFRECORD_ALREADY')):
            print("Tfrecord ready!")
            return

        data_pairs = [("train.csv", "train.tfrecord"), ("test.csv", "test.tfrecord")]
        for data_file, tfr_file in data_pairs:
            data = pd.read_csv(os.path.join(self.dataframe_dir, data_file))
            with tf.io.TFRecordWriter(os.path.join(self.tfrecord_dir, tfr_file)) as f:
                # chunk_data = test_data.get_chunk(50000)
                for index, row in tqdm(data.iterrows(), total=data.shape[0]):
                    features = {}
                    # 写入定长特征
                    # 写入dense特征
                    for dense_feature in self.dense_features:
                        features[dense_feature] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=[row[dense_feature]])
                        )

                    # 写入category特征
                    for category_feature in self.category_featurs:
                        category_value = row[category_feature] if not pd.isna(row[category_feature]) else ''
                        features[category_feature] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[category_value.encode()])
                        )

                    # 写入label
                    for label in self.labels:
                        features[label] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=[row[label]])
                        )



                    # 写入序列特征
                    feature_list = {}
                    for seq_feature in self.seq_features:
                        seq_list = row[seq_feature].split(',') if not pd.isna(row[seq_feature]) else []
                        feature_list[seq_feature] = tf.train.FeatureList(
                            feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode()])) for v in
                                     seq_list]
                        )

                    # 封装example
                    example = tf.train.SequenceExample(
                        context=tf.train.Features(feature=features),
                        feature_lists=tf.train.FeatureLists(feature_list=feature_list)
                    )
                    f.write(example.SerializeToString())

        # 生成完成标识
        with open(os.path.join(self.tfrecord_dir, 'TFRECORD_ALREADY'), 'w'):
            pass


if __name__ == '__main__':
    data = DataGenerator()
    # 验证tfrecord
    def batch_sequence_example_parser(serialized_example):
        """批量解析SequenceExample"""
        context_features = {
            "feedid": tf.FixedLenFeature([], dtype=tf.string),
        }
        sequence_features = {
            "his_read_comment_7d_seq": tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
        }

        context_parsed, sequence_parsed, sequence_length = tf.io.parse_sequence_example(
            serialized=[serialized_example],
            context_features=context_features,
            sequence_features=sequence_features
        )

        feedid = context_parsed['feedid']
        his_read_comment_7d_seq = sequence_parsed['his_read_comment_7d_seq']
        his_read_comment_7d_seq_length = sequence_length['his_read_comment_7d_seq']
        return feedid, his_read_comment_7d_seq, his_read_comment_7d_seq_length

    def examle_parse_within_fc(serialized_example):
        """尝试用feature_columns解析成dense input直接作为input_layer的输入"""
        features = {
            "feedid": tf.FixedLenFeature([], dtype=tf.string),
            "his_read_comment_7d_seq": tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
        }
        features_parsed = tf.io.parse_example(
            serialized=[serialized_example],
            features=features
        )
        feedid = features_parsed['feedid']
        his_read_comment_7d_seq = features_parsed['his_read_comment_7d_seq']
        return feedid, his_read_comment_7d_seq


    def batched_data(tfrecord_filename, example_parser, batch_size, padded_shapes=None, padding_values=None,
                     num_epochs=4,
                     buffer_size=1000):
        dataset = tf.data.TFRecordDataset(tfrecord_filename) \
            .map(example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .shuffle(buffer_size) \
            .repeat(num_epochs) \
            .padded_batch(batch_size,
                          padded_shapes=padded_shapes,
                          # 使用默认, 数字补0(如果是用来lookup embedding, 则设为-1), 字符串补''
                          padding_values=padding_values) \
            .prefetch(0)

        return dataset.make_one_shot_iterator()

    iter = batched_data('./tfrecord/train.tfrecord',
                        example_parser=examle_parse_within_fc,
                        batch_size=2048,
                        padded_shapes=([None], [None, None]),  # 这里必须显式写明所有的padded_shape
                        padding_values=(tf.constant(u"", tf.string), tf.constant(u"padding_value", tf.string)),
                        num_epochs=1,
                        buffer_size=10000)


    i = 1
    next_element = iter.get_next()
    with tf.Session() as sess:
        while True:
            try:
                feedid, his_behaviors = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                print("done training")
                break
            finally:
                print('==============batch %s ==============' % i)
                print('userid: value: \n %s \n shape: %s | type: %s' % (feedid, feedid.shape, feedid.dtype))
                print('his_behaviors: value: \n %s \n shape: %s | type: %s' % (
                    his_behaviors, his_behaviors.shape, his_behaviors.dtype))
                # print('his_behaviors_length: value: \n %s \n shape: %s | type: %s' % (
                #     his_behaviors_length, his_behaviors_length.shape, his_behaviors_length.dtype))
                # print('squeezed:')
                # # 调整shape, 将多余的dimension为1去掉(在生产环境里, sess内部不要调用任何tf操作, 否则会在graph里不停加node, 直到内存OOM)
                # his_behaviors_s = sess.run(tf.squeeze(his_behaviors, axis=1))
                # print('his_behaviors: value: \n %s \n shape: %s | type: %s' % (
                #     his_behaviors_s, his_behaviors_s.shape, his_behaviors_s.dtype))
                print('\n')
            i += 1

