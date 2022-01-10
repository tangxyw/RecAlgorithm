"""
    生成豆瓣电影训练集的tfrecord文件

    训练集: 2019年1月-8月评分(生成特征需要开360天的窗口)
    测试集: 2019年9月评分
    正负样本生成逻辑：大于等于4分为正样本，小于4分为负样本。

    特征：
        user侧：
            user_id: 用户id

        item侧:
            movie_id: 电影id

        交叉:
"""
import os
import numpy as np
import pandas as pd
# import swifter
# swifter.register_modin()
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=3)
from collections import Counter


class DataGenerator:
    """生成豆瓣电影训练集的tfrecord文件"""

    def __init__(self, dataset_dir='./', out_path='./'):
        """
        Args:
            dataset_dir (str): 数据文件所在文件夹路径
            out_path (str): tfrecord文件以及类别特征vocabulary文件输出文件夹路径
        """
        self.dataset_dir = dataset_dir
        self.out_path = out_path

        self._load_data()

        # self._generate_vocabulary_file()

        self._generate_tfrecord()

    def _load_data(self):
        """读入数据"""
        self.ratings = pd.read_csv(self.dataset_dir + 'ratings.csv')
        self.movies = pd.read_csv(self.dataset_dir + 'movies.csv')
        self.ratings['RATING_TIME'] = pd.to_datetime(self.ratings['RATING_TIME'])
        self.ratings['RATING_MONTH'] = self.ratings['RATING_TIME'].dt.to_period('M')
        self.ratings['RATING_DAY'] = self.ratings['RATING_TIME'].dt.to_period('D')
        # YEAR为0的是异常值, 不能进入vocabulary
        self.movies['YEAR'] = self.movies['YEAR'].apply(lambda x: np.nan if x == 0 else str(np.int16(x)))

    def _generate_vocabulary_file(self):
        """生成所有类别特征的vocabulary文件(txt格式)"""

        # 拼接评分表和电影表
        data = self.ratings[self.ratings['RATING_MONTH'] >= '2019-01']
        data = pd.merge(data, self.movies, how='left', on='MOVIE_ID')

        # 生成各个类别变量的vocabulary文件
        category_columns = ['USER_MD5', 'MOVIE_ID', 'GENRES', 'ACTOR_IDS', 'DIRECTOR_IDS', 'LANGUAGES', 'REGIONS',
                            'YEAR', ]
        # 与category_columns相对应的分隔符
        seps = ['/', '/', '/', '|', '|', ' / ', ' / ', '/', ]
        # 与category_columns相对应的min_count阈值,小于阈值的不在词典中保留
        min_counts = [5, 5, 20, 5, 5, 20, 20, 20]

        self.vocabulary_dict = {}  # 存放每个类别变量的vocabulary
        for col_name, sep, min_count in zip(category_columns, seps, min_counts):
            if col_name != 'YEAR':
                continue
            vocabulary = self._category_value_count(data[data['RATING_MONTH'] <= '2019-08'], col_name, sep)
            self.vocabulary_dict[col_name] = vocabulary

            vocabulary_file_path = os.path.join(self.out_path, col_name + '.txt')
            with open(vocabulary_file_path, 'w') as f:
                for key, count in vocabulary.items():
                    if count >= min_count:
                        f.write(key + '\n')

    def _generate_tfrecord(self):
        """生成trecord格式的训练集和测试集"""
        columns = ['USER_MD5', 'MOVIE_ID', 'RATING_TIME', 'RATING_DAY', 'GENRES', 'ACTOR_IDS', 'DIRECTOR_IDS',
                   'LANGUAGES', 'REGIONS',
                   'YEAR', 'RATING']
        data = self.ratings[self.ratings['RATING_TIME'] >= '2018-01-01']
        data = pd.merge(data, self.movies, how='left', on='MOVIE_ID')[columns]
        data.sort_values(by=['USER_MD5', 'RATING_TIME'], ascending=True, inplace=True)

        del self.ratings
        del self.movies

        print(data.columns)

        def generate_features_step1():
            # 使用apply有内存溢出的问题, 改用for循环代替
            for index, row in tqdm(data.iterrows(), total=data.shape[0]):
                data.at[index, 'DIRECTOR_ID'] = row['DIRECTOR_IDS'].split('|')[0] if not pd.isnull(
                    row['DIRECTOR_IDS']) else pd.NA  # 导演只取第1个

                if row['RATING_TIME'] >= pd.to_datetime('2019-01-01'):
                    data.at[index, 'ACTOR_IDS'] = row['ACTOR_IDS'].split('|') if not pd.isnull(
                        row['ACTOR_IDS']) else pd.NA
                    data.at[index, 'GENRES'] = row['GENRES'].split('/') if not pd.isnull(row['GENRES']) else pd.NA
                    data.at[index, 'LANGUAGES'] = row['LANGUAGES'].split(' / ') if not pd.isnull(
                        row['LANGUAGES']) else pd.NA
                    data.at[index, 'REGIONS'] = row['REGIONS'].split(' / ') if not pd.isnull(row['REGIONS']) else pd.NA

        def generate_features_step2():
            # 使用apply有内存溢出的问题, 改用for循环代替
            for index, row in tqdm(data.iterrows(), total=data.shape[0]):
                if row['RATING_TIME'] >= pd.to_datetime('2019-01-01'):
                    window_size = 360  # 窗口大小360天
                    cur_user = row['USER_MD5']  # 当前user
                    cur_date = row['RATING_DAY']  # 当前行样本日期
                    cur_director = row['DIRECTOR_ID']  # 当前导演
                    user_data = data.loc[cur_user]
                    # 生成窗口数据
                    w_data = user_data[(user_data['RATING_DAY'] < cur_date) &
                                       (cur_date - user_data['RATING_DAY'] <= pd.Timedelta(window_size, unit='d'))
                                       ]
                    history_movies = list(w_data['MOVIE_ID'])
                    # 因为使用了多重索引, 所以赋值的时候不能是list, 只能先拼成字符串, 之后再转换回来
                    # 只取最近20个评分的电影
                    data.at[index, 'HISTORY_MOVIES'] = '/'.join(history_movies[(len(history_movies) - 20):])

                    # 最近一次评分距今天数差
                    # 最近一次评分分数
                    # 360天内所有评分的最大值/最小值/平均值
                    # 360天内所有同一个导演的评分的最大值/最小值/平均值


        # generate_features_step1()
        # print(data.columns)
        # pd.to_pickle(data, 'data_step1.pickle')
        # del data

        data = pd.read_pickle('data_step1.pickle')
        data.reset_index(inplace=True)
        data.set_index(['USER_MD5', 'index'], append=False, drop=False, inplace=True)
        data['MOVIE_ID'] = data['MOVIE_ID'].astype(str)
        generate_features_step2()
        # data = data.progress_apply(generate_features_step1, axis=1)
        # data = data.progress_apply(generate_features_step2, axis=1)
        # data = data.swifter.allow_dask_on_strings().apply(generate_features_step2, axis=1)  # 这里用swifer会异常,不报错但跑不出结果
        # print(data['HISTORY_MOVIES'])

    def _category_value_count(self, df, column_name, sep='/'):
        """
            统计df[column_name]里所有取值出现的次数
        """
        vocabulary = Counter()
        for _, row in df.iterrows():
            s = row[column_name]
            # if s is np.nan:
            if pd.isnull(s):
                continue
            else:
                value_list = str(s).split(sep)
                print(value_list)
            vocabulary += Counter(value_list)
        return vocabulary


if __name__ == '__main__':
    # print(os.getcwd())
    douban = DataGenerator()
