import logging
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)


class DataUtil:
    """
    path: 文件路径(可以用相对路径)
    train_num: 是否设置训练集，训练集的数量
    tar_idx: 标签的列号，默认是最后一列
    shuffle: 是否对数据集进行洗牌，对于贝叶斯算法，在没有区分训练集的情况下，
    洗牌只是让最初的数据变得混乱，对最后的结果没有影响
    这里为了能得到和我们前面的手算结果一致我们就不洗牌了
    """
    # 将输入特征(x)和标签(y)分离
    def get_dataset(path, train_num=None, tar_idx=None, shuffle=False):
        x = []
        with open(path, mode="r", encoding="utf-8") as f:
            for sample in f:
                x.append(sample.strip().split("，"))
        if shuffle:
            np.random.shuffle(x)
        # python中的三元运算
        tar_idx = -1 if tar_idx is None else tar_idx
        # 得到分离后的特征（x）和标签（y）
        y = np.array([xx.pop(tar_idx) for xx in x])
        x = np.array(x)
        logging.debug("get_dataset-x")
        logging.debug(x)
        logging.debug("get_dataset-y")
        logging.debug(y)

        if train_num is None:
            return x, y
        return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])


class MultinomialNB(object):
    def __init__(self):
        # 数值化后的输入向量
        self._x = None
        # 数值化后的标签
        self._y = None
        # 训练结果，条件概率分布
        self._data = None
        # 决策函数
        self._func = None
        # 每个向量的取值情况
        self._n_possibilities = None
        # 按标签分类后的x
        self._labelled_x = None
        # 将x，y进行捆绑
        self._label_zip = None
        # 标签的统计
        self._cat_counter = None
        # 特征值的统计
        self._con_counter = None
        # 标签的字典
        self._label_dic = None
        # 特征值的字典
        self._feat_dic = None
        #最终数据
        self.y_pred=None

    # 进行转换分析
    def feed_data(self, x, y, sample_weight=None):
        # 进行转置
        if isinstance(x, list):
            features = map(list, zip(*x))
        else:
            features = x.T
        # 利用集合获取各个维度的特征值
        # TODO:[0.2] 对每个维度生成字典
        features = [set(feat) for feat in features]
        feat_dict = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dic = {_l: i for i, _l in enumerate(set(y))}
        # [0.4] 按照特征值字典进行映射
        x = np.array([[feat_dict[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dic[yy] for yy in y])
        # [1.0] 统计y中每个特征出现的频率
        cat_counter = np.bincount(y)
        # [0.3] 为了平滑，我们需要计算K(每个向量的特征值个数)
        n_possibilities = [len(feads) for feads in features]
        # 获取各类别数据的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下标获取记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]

        self._x = x
        logging.debug("self._x")
        logging.debug(x)

        self._y = y
        logging.debug("self._y")
        logging.debug(y)

        self._labelled_x = labelled_x
        logging.debug("labelled_x")
        logging.debug(labelled_x)

        self._label_zip = list(zip(labels, labelled_x))
        logging.debug("_label_zip")
        logging.debug(self._label_zip)

        self._cat_counter = cat_counter
        logging.debug("cat_counter")
        logging.debug(cat_counter)

        self._feat_dic = feat_dict
        logging.debug("feat_dict")
        logging.debug(feat_dict)

        self._n_possibilities = n_possibilities
        logging.debug("n_possibilities")
        logging.debug(n_possibilities)
        # 注意这里是反字典
        self._label_dic = {i: _l for _l, i in label_dic.items()}
        logging.debug("label_dic")
        logging.debug(self._label_dic)

        # 处理样本权重的函数，以更新记录概率的数组
        self.feed_sample_weight(sample_weight)
        logging.debug("feed_sample_weight _con_counter")
        logging.debug(self._con_counter)

    # 计算先验概率
    def get_prior_probability(self, lb=0):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter)) \
                for _c_num in self._cat_counter]

    # 统计每个维度的特征值出现的次数
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                self._con_counter.append(
                    [np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p) for
                     label, xx in self._label_zip])

    def _fit(self, lb):
        # 维度
        n_dim = len(self._n_possibilities)
        # n种结果
        n_category = len(self._cat_counter)
        # TODO：计算先验概率的函数，lb就是平滑项
        p_category = self.get_prior_probability(lb)
        logging.debug("p_category")
        logging.debug(p_category)
        # 初始化 data
        # In [1]: [None]*3
        # Out[1]: [None, None, None]
        data = [None] * n_dim
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [[(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities) \
                          for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.array(dim_info) for dim_info in data]
        logging.debug("_data")
        logging.debug(self._data)

        # def func 是用来返回的
        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            return rs * p_category[tar_category]

        return func

    """计算各个特征概率 如 
        [
        array([[0.625, 0.375],
               [0.625, 0.375]]), 
        array([[0.75, 0.25],
               [0.25, 0.75]]), 
        array([[0.375, 0.625],
               [0.625, 0.375]]), 
        array([[0.25, 0.75],
               [0.75, 0.25]])]
        """
    def fit(self, x=None, y=None, sample_weight=None, lb=0):
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._func = self._fit(lb)

    def predict_one(self, x, get_raw_result=False):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = x[:]
        # 相关数值化方法进行数值话
        x = self._transfer_x(x)
        m_arg = 0
        m_probability = 0
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            logging.debug("p" + str(i))
            logging.debug(p)
            if p > m_probability:
                m_arg = i
                m_probability = p
        if not get_raw_result:
            return self._label_dic[m_arg]
        return m_probability

    def _transfer_x(self, x):
        return  np.array([self._feat_dic[i][_l] for i, _l in enumerate(x)])

    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    def evaluate(self, x, y):
        self.y_pred = self.predict(x)
        logging.debug("y_pred")
        logging.debug(self.y_pred)
        print("正确率:{:12.6}%".format(100 * np.sum(self.y_pred == y) / len(y)))

    def text_save(self, filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
        localtime = time.asctime(time.localtime(time.time()))
        file = open(filename, 'a')
        file.write(localtime+"测试结果为"+'\n')
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
            s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
            file.write(s)
        file.close()
        print("保存文件成功")

    def saveresult(self):
        np.save("./data", self._data)
        print("data保存成功")
        np.save("./label_dic", self._label_dic)
        print("label_dic保存成功")
        np.save("./feat_dic", self._feat_dic)
        print("_feat_dic保存成功")
        self.text_save("./dataresult.txt", self.y_pred)
        print("y_pred保存成功")

    def loadresult(self, datapath, label_dicpath, feat_dic):
        self._data = np.loadtxt(datapath)
        print("datapath导入成功")
        print(self._data)
        self._label_dic = np.load(label_dicpath)
        print("label_dicpath导入成功")
        print(self._label_dic)
        self._feat_dic = np.load(feat_dic)
        print("_feat_dic导入成功")
        print(self._feat_dic)


if __name__ == '__main__':
    import time

    # TODO:数据导入
    # 读入数据
    logging.debug("开始导入数据\n")
    _x, _y = DataUtil.get_dataset("testshuju.txt")
    # 实例化模型并进行训练
    # TODO：数据训练
    learning_time = time.time()
    nb = MultinomialNB()
    # nb.loadresult("./data.npy", "./label_dic.npy", "./feat_dic.npy")
    nb.fit(_x, _y)
    learning_time = time.time() - learning_time
    # TODO：进行评估
    estimation_time = time.time()
    nb.evaluate(_x, _y)
    estimation_time = time.time() - estimation_time
    nb.saveresult()
