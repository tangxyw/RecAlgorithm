# 主流推荐系统Rank算法的实现

![Python](https://img.shields.io/badge/Python-3.6-green?logo=python)
![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-1.14-blue.svg)

### 项目简介

- 实现推荐系统中主要使用的Rank算法，并使用公开数据集评测，所有算法均已跑通并完成完整的训练，最终生成`saved_model`和`checkpoint`供`tf-serving`部署；
* 使用[微信视频号推荐算法比赛](https://algo.weixin.qq.com/problem-description)数据集，数据详情请见 [./dataset/README.md](./dataset/README.md)；
* 为了贴合工业界使用情况，使用`TensorFlow Estimator`框架，数据format为`Tfrecord`；
* 算法实现在`./algrithm`下，每个算法单独一个文件夹，名字为普遍接受的大写算法名称，训练入口为文件夹下对应的小写算法名称py文件，如DIN文件夹下的`din.py`文件为训练DIN模型的入口，具体请见末尾的示例部分；
* 每个算法都实现了自己的`model_fn`，没有使用`Keras`高阶API，只使用`TensorFlow`的中低阶API构造静态图；
* 算法超参数可由`--parameter_name=parameter_value`方式传入训练入口脚本，超参数定义请见训练入口脚本`tf.app.flags`部分；
* 单任务模型使用数据集因变量中的`read_comemnt`评测，多任务模型使用`read_commet` `like` `click_avatar`三个任务评测；

### 单任务Models列表

| Model        | Paper                                                                                                                                                      | *Best_read_comment_Auc |
|:------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------:|
| FFM          | [2016] [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)                                          | 0.8911285              |
| DeepCrossing | [2016] [Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)    | 0.9185908              |
| PNN          | [2016] [Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                  | 0.9065931              |
| Wide & Deep  | [2016] [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                | 0.9133482              |
| DeepFM       | [2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           | 0.8529998              |
| DCN          | [2017] [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   | 0.9183242              |
| AFM          | [2017] [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) | 0.9117872              |
| xDeepFM      | [2018] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                       | 0.9152467              |
| FwFM         | [2018] [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)              | 0.9118794              |
| DIN          | [2018] [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                     | 0.9116896              |
| DIEN         | [2018] [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                                     | -                      |
| FiBiNet      | [2019] [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)    | 0.9149044              |
| BST          | [2019] [Behavior sequence transformer for e-commerce recommendation in Alibaba](https://arxiv.org/pdf/1905.06874.pdf)                                      | 0.9165866              |

*Best_read_comment_Auc为每个model各自调参后的测试集最大Auc，每个model各自的评测见每个model路径下的`result.md`。 </br>
*DIEN不适用于微信视频号数据集，故只实现了静态图，并没有评测。

### 多任务Models列表

| Model | Paper                                                                                                                                                                   | *Best_read_commet_AUC | *Best_like_AUC | *Best_click_avatar_AUC |
|:-----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|:--------------:|:----------------------:|
| ESMM  | [2018] [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)                               | -                     | -              | -                      |
| MMOE  | [2018] [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)                      | 0.91860557            | 0.8126400      | 0.8139362              |
| PLE   | [2020] [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236) | 0.91965175            | 0.8136461      | 0.8154559              |

*Best_xx_AUC为所有超参数组合中的最高值，横向的三个AUC可能不在同一组超参数中。</br>
*由于ESMM的结构特殊性，不适用于微信视频号数据集，故只实现了静态图，并没有评测。

### 示例

```shell
# 先执行以下命令确保生成了tfrecord
# cd ./dataset/wechat_algo_data1
# python DataGenerator.py && cd ..
cd ./DIN
# 训练时可自定义参数
python din.py --use_softmax=True 
```

### To Do List
* 增加多任务学习Trick: Uncertainty, GradNorm, PCGrad, etc.
* 增加AutoInt, FLEN, etc.
* 重构特征工程部分, 包括配置化输入等, 参考https://github.com/Shicoder/Deep_Rec

## 欢迎提issue，或直接勾搭

<img src="./docs/Wechat.jpeg" alt="pic" width="220" height="220">
