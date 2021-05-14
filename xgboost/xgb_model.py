import numpy as np
from collections import Counter, defaultdict
import copy
from config import CONFIG


# %% 简单二分类
class Xgb:
    def __init__(self, target, reg_l2, reg_l1, gamma):
        self.target = target  # reg if target is regression else classify
        self.n_estimator = CONFIG['xgboost_cart']['params']['n_estimators']
        self.lr = CONFIG['xgboost_cart']['params']['lr']
        self.max_depth = CONFIG['xgboost_cart']['params']['max_depth']
        self.min_child_weight = CONFIG['xgboost_cart']['params']['min_child_weight']
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.tree_list = []
        self.gain_list = []
        if self.target.startswith('reg'):
            self.base_score = CONFIG['xgboost_cart']['params']['base_score']
        else:
            self.base_score = np.log(CONFIG['xgboost_cart']['params']['base_score'] /
                                     (1 - CONFIG['xgboost_cart']['params']['base_score']))
        self.gamma = gamma

    # %% 梯度和
    def cal_g(self, pred, y):
        return np.sum(pred - y)

    # %% 二阶导
    def cal_h(self, pred):
        if self.target.startswith('reg'):
            return len(pred)
        return np.sum(pred * (1 - pred))

    # %% 基于特征的所有取值 判断是否满足划分调教 分类输出 特征取值数据 和 首索引
    @staticmethod
    def split_data(data, feat, val, data_type='classifier'):
        if data_type == 'classifier':  # 分类
            arr1 = data[np.nonzero(data[:, feat] == val)]
            arr2 = data[np.nonzero(data[:, feat] != val)]
        else:  # 回归
            arr1 = data[np.nonzero(data[:, feat].astype(float) < val)]
            arr2 = data[np.nonzero(data[:, feat].astype(float) >= val)]
        return arr1, arr2, np.nonzero(data[:, feat].astype(float) < val)[0], \
               np.nonzero(data[:, feat].astype(float) >= val)[0]

    # %% 连续变量的切分点处理，输出 连续两个变量的均值
    @staticmethod
    def continuity_params_progress(arr, feat):
        c = arr[:, feat].astype(float)
        c_sort = sorted(set(c), reverse=True)  # 降序
        new_c = []
        for i in range(len(c_sort) - 1):
            val = (c_sort[i] + c_sort[i + 1]) / 2
            new_c.append(val)
        return new_c

    # %% 选择最优的切分点 max_gain最大 且>0
    def select_split(self, data, y):
        max_gain = -1
        best_feat = None
        best_val = None
        left, right, left_y, right_y, g_left, h_left, g_right, h_right = None, None, None, None, None, None, None, None
        data_type = 'continuity'
        for i in range(data.shape[1] - 1):
            c_set = self.continuity_params_progress(data, i)  # list [mean,mean...]
            for val in c_set:  # 取值遍历
                arr1, arr2, arr1_index, arr2_index = self.split_data(data, i, val, data_type)
                gain, G_left, H_left, G_right, H_right = self.cal_gain(arr1, y[arr1_index], arr2, y[arr2_index])
                if max_gain < gain and gain > 0 and self.min_child_weight <= min(H_left, H_right):
                    max_gain = gain
                    best_feat = i
                    best_val = val
                    left = arr1
                    right = arr2
                    left_y = y[arr1_index]
                    right_y = y[arr2_index]
                    g_left = G_left
                    h_left = H_left
                    g_right = G_right
                    h_right = H_right
        if best_feat is None:
            g = self.cal_g(data[:, -1], y)
            h = self.cal_h(data[:, -1])
            return best_feat, best_val, left, right, left_y, right_y, g, h, g, h
        self.gain_list.append({best_feat: max_gain+self.gamma})
        return best_feat, best_val, left, right, left_y, right_y, g_left, g_right, h_left, h_right

    # %% 计算基于某个点划分后的左右数据集的gain
    def cal_gain(self, left, left_y, right, right_y):
        g_left = self.cal_g(left[:, -1], left_y)
        h_left = self.cal_h(left[:, -1])
        g_right = self.cal_g(right[:, -1], right_y)
        h_right = self.cal_h(right[:, -1])

        # gain 相当于误差函数
        gain = (g_left ** 2 / (h_left + self.reg_l1) + g_right ** 2 / (h_right + self.reg_l1) -
                (g_left + g_right) ** 2 / (h_left + h_right + self.reg_l1)) - self.gamma
        return gain, g_left, h_left, g_right, h_right

    # %% 构建递归树
    def create_tree(self, data, y, n=0):
        tree = {}
        dd = data[:, :-1].tolist()   # 除了目标列的所有数据
        ddd = list(map(tuple, dd))
        cc = Counter(ddd)  # 统计目标列的频数
        if len(cc) == 1:  # 只有一个类别时 叶子节点直接return
            g = self.cal_g(data[:, -1], y)
            h = self.cal_h(data[:, -1])
            return - g / (h + self.reg_l1)   # 返回叶子节点的权重
        best_feat, best_val, left, right, left_y, right_y, g_left, g_right, h_left, h_right = self.select_split(data, y)

        if best_feat is None:
            return - g_left / (h_left + self.reg_l1)
        n += 1
        if n >= self.max_depth:
            tree[(best_feat, best_val, 'left')] = - g_left / (h_left + self.reg_l1)
            tree[(best_feat, best_val, 'right')] = - g_right / (h_right + self.reg_l1)
        else:
            tree[(best_feat, best_val, 'left')] = self.create_tree(left, left_y, n)
            tree[(best_feat, best_val, 'right')] = self.create_tree(right, right_y, n)
        return tree

    # %% 预测单课树
    def predict_one(self, tree, x):
        if type(tree) != dict:
            return tree
        for key in tree:
            if x[key[0]] < key[1]:
                r = tree[(key[0], key[1], 'left')]
            else:
                r = tree[(key[0], key[1], 'right')]
            return self.predict_one(r, x)

    # %% 基于所有树输出 分类概率
    def predict(self, x):
        result = self.tree_list[0]
        for tree in self.tree_list[1:]:
            result += self.lr * self.predict_one(tree, x)
        if self.target.startswith('reg'):
            return result
        return 1 / (1 + np.exp(-result))   # sigmoid 二分类概率输出

    # %%数据拟合 学习树结构
    def fit(self, datasets):
        data = copy.copy(datasets)
        self.tree_list.append(self.base_score)
        for i in range(self.n_estimator):
            for j in range(len(data)):
                data[j, -1] = self.predict(data[j, :-1])   # 对第j个样本预测
            # 负梯度学习 新一轮的学习data的预测结果和 上一轮的目标列（再上一轮的预测结果）
            self.tree_list.append(self.create_tree(data, datasets[:, -1]))

    def fea_importance(self):
        feat_imp = defaultdict(float)
        feat_counts = defaultdict(int)
        for item in self.gain_list:
            k, v = list(item.items())[0]
            feat_imp[k] += v
            feat_counts[k] += 1
        for k in feat_imp:
            feat_imp[k] /= feat_counts[k]
        v_sum = sum(feat_imp.values())
        for k in feat_imp:
            feat_imp[k] /= v_sum
        return feat_imp  # 输出特征权重占比


if __name__ == '__main__':
    from dt import dt_create_data
    datas, features = dt_create_data.CreateData().getdata()
    xgbclass = Xgb('classifier', target, reg_l2, reg_l1, gamma)
