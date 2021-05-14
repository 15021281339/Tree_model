import pandas as pd
import numpy as np
import random
import math
import collections
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import decisiontree_classify
from config import CONFIG


class RandomForestTrainer(object):
    def __init__(self, name='classify'):
        self.trees = None
        self.feature_importance_ = dict()
        self.type = name   # ['classify','regression']
        self.n_estimators = CONFIG['rfc']['params']['n_estimators']
        self.colsample_bytree = CONFIG['rfc']['params']['colsample_bytree']
        # self.subsample = CONFIG['rfc']['params']['subsample']

    # %% 数据导入
    def load_data(self):
        self.datas = load_breast_cancer()

        x, y = self.datas.data, self.datas.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                            test_size=0.2, random_state=10)
        if self.type == 'classify':
            self.rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                             random_state=11,
                                             max_features=self.colsample_bytree)
        else:
            self.rf = RandomForestRegressor()
        self.rf.fit(self.x_train, self.y_train)
        # return self.datas.feature_names

    # %% 测试集预测准确率
    def get_score(self):
        result = self.rf.score(self.x_test, self.y_test)
        return result

    # %% 获取特征重要性
    def get_importance(self):
        f_list = self.rf.feature_importances_
        feat_list = {self.datas.feature_names[i]: f_list[i] for i in range(len(f_list))}

        return sorted(feat_list.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    rftrainer = RandomForestTrainer()
    rftrainer.load_data()

    f_list = rftrainer.get_importance()
    print(f_list)
    # print(rftrainer.get_score())

