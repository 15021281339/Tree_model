import numpy as np
import scipy
import matplotlib
import sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# %% 输入数据处理
class CreateData(object):

    def __init__(self, data_path=None):
        self.data_path = data_path

    def getdata(self):
        datasets = [[0, 0, 0, 0, 'no'],  # 数据集
                    [0, 0, 0, 1, 'no'],
                    [0, 1, 0, 1, 'yes'],
                    [0, 1, 1, 0, 'yes'],
                    [0, 0, 0, 0, 'no'],
                    [1, 0, 0, 0, 'no'],
                    [1, 0, 0, 1, 'no'],
                    [1, 1, 1, 1, 'yes'],
                    [1, 0, 1, 2, 'yes'],
                    [1, 0, 1, 2, 'yes'],
                    [2, 0, 1, 2, 'yes'],
                    [2, 0, 1, 1, 'yes'],
                    [2, 1, 0, 1, 'yes'],
                    [2, 1, 0, 2, 'yes'],
                    [2, 0, 0, 0, 'no']]
        fea_labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
        return datasets, fea_labels

    # def get_fea_labels(self):
    #     iris = load_iris()
    #     return iris.feature_names
