from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle


# %% 针对离散特征-决策树
class DeciTree(object):
    def __init__(self):
        pass
        # self.data = data

    # %% 基于特征axis的划分点value划分数据集
    def split_data(self, data, axis, value):
        new_dataset = []
        for feat in data:
            if feat[axis] == value:
                reducefeat = list(feat[:axis])  # 去掉axis特征
                reducefeat.extend(feat[axis + 1:])
                new_dataset.append(reducefeat)
        return new_dataset

    def calculate(self, data):
        data_length = len(data)
        label_counts = {}  # 统计标签出现的次数
        for feat in data:
            current_label = feat[-1]  # 提取标签信息
            if current_label not in label_counts.keys():
                label_counts[current_label] = 1  # 第一次出现计数1
            label_counts[current_label] += 1
        shannon = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / data_length
            shannon -= prob * log(prob, 2)  # 熵计算公式 -累加(p*logp) p为概率
        return shannon

    def choose_best_feature(self, data):
        num_fea = len(data[0]) - 1  # 特征数量
        base_entropy = self.calculate(data)
        best_info_gain = 0.0  # 信息增益
        best_fea = -1  # 最优特征的索引值 return
        for i in range(num_fea):
            # 遍历，获取data的第i个特征的所有取值样本fea_list
            fea_list = [example[i] for example in data]
            unique_val = set(fea_list)  # 去重
            new_entropy = 0.0  # 计算经验条件信息熵
            for value in unique_val:
                sub_data = self.split_data(data, i, value)  # 划分后的子集
                pro = len(sub_data) / float(len(data))  # 计算子集的数量占比
                new_entropy += pro * self.calculate(sub_data)  # 累计每个特征取值带来的熵值
            info_gain = base_entropy - new_entropy  # 信息增益（用父节点的熵-划分后的熵）
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_fea = i
        return best_fea  # 输出特征标签索引

    # %% 输出标签中出现次数最多的标签
    def major_cnt(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys(): class_count[vote] = 1
            class_count[vote] += 1
        # 根据字典的值降序排序
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),
                                    reverse=True)
        return sorted_class_count[0][0]

    # %% 训练生成决策树 input : data数据集 labels特征标签集[]  fea_labels 选择的最优特征标签
    def create_Tree(self, data, labels, fea_labels):
        class_list = [example[-1] for example in data]
        if class_list.count(class_list[0]) == len(class_list):  # 类别一致 停止生成
            return class_list[0]
        if len(data[0]) == 1 or len(labels) == 0:
            return self.major_cnt(class_list)
        best_fea = self.choose_best_feature(data)  # 选择最优特征
        best_fea_label = labels[best_fea]
        fea_labels.append(best_fea_label)
        my_tree = {best_fea_label: {}}
        del (labels[best_fea])  # 删除已经使用的最优特征标签
        fea_value = [example[best_fea] for example in data]  # 得到训练集中最优特征的属性值
        unique_value = set(fea_value)  # 去重
        for value in unique_value:  # 对每一个取值分支生成tree
            sub_labels = labels[:]
            my_tree[best_fea_label][value] = self.create_Tree(self.split_data(data, best_fea, value), sub_labels,
                                                              fea_labels)
        return my_tree

    def get_num_leaf(self, myTree):
        num_leaf = 0
        first = next(iter(myTree))  # 获取节点
        second = myTree[first]  # 子节点（下一组字典）
        for key in second.keys():
            if type(second[key]).__name__ == 'dict':
                num_leaf += self.get_num_leaf(second[key])  # 为字典不是叶子节点 递归到下一个子节点
            else:
                num_leaf += 1  # 叶子节点+1
        return num_leaf

    def get_depth_(self, myTree):
        max_depth = 0
        first = next(iter(myTree))  # 获取节点
        second = myTree[first]  # 子节点（下一组字典）
        for key in second.keys():
            if type(second[key]).__name__ == 'dict':
                this_depth = 1 + self.get_depth_(second[key])
            else:
                this_depth = 1
            if this_depth > max_depth: max_depth = this_depth  # 更新最大层数
        return max_depth

    # %% 预测 test输入样例
    def classify(self, input_tree, feat_labels, test):
        first = next(iter(input_tree))  # 获取节点
        second = input_tree[first]
        feat_index = feat_labels.index(first)  # 初始最优特征索引
        for key in second.keys():
            if test[feat_index] == key:
                if type(second[key]).__name__ == 'dict':
                    class_label = self.classify(second[key], feat_labels, test)
                else:
                    class_label = second[key]
        return class_label
