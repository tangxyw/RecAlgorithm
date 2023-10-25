# 数据集说明

## wechat_algo_data1

* 微信视频号推荐算法比赛：[Link](https://algo.weixin.qq.com/problem-description)；
* 原始文件为：`user_action.csv` `feed_info.csv` ，~~可由大赛页面下载，~~ 目前大赛页面已经下线，可联系作者索取；
* `EDA.ipynb`为数据探索结果；
* `DataGenerator.py`为ETL脚本，生成数据集：训练集3322313条，测试集609037条，特征工程逻辑请见`DataGenerator.py`头部注释。



### 启动ETL

```shell
cd ./dataset/wechat_algo_data1
python DataGenerator.py && cd ..
```

生成的目录结构如下：

```
├── dataframe
│   ├── DATAFRAME_ALREADY
│   ├── test.csv
│   └── train.csv
├── DataGenerator.py
├── EDA.ipynb
├── features
│   ├── cross_agg_features.pkl
│   ├── FEATURES_PKL_ALREADY
│   ├── feed_agg_features.pkl
│   ├── user_agg_features.pkl
│   └── user_seq_features.pkl
├── feed_info.csv
├── tfrecord
│   ├── test.tfrecord
│   ├── TFRECORD_ALREADY
│   └── train.tfrecord
├── user_action.csv
└── vocabulary
    ├── authorid.txt
    ├── bgm_singer_id.txt
    ├── bgm_song_id.txt
    ├── device.txt
    ├── feedid.txt
    ├── manual_tag_id.txt
    ├── userid.txt
    └── VOCAB_FILE_ALREADY
```

- `dataframe` `features` 文件夹内为ETL中间结果

- `vovabulary`文件夹内为各个类别特征的词典

- `tfrecord`文件夹内为最终生成的训练集和测试集
