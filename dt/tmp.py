# from sklearn.datasets import load_diabetes
# iris = load_diabetes()
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import numpy as np
# import pandas as pd
# # data = pd.DataFrame(np.concatenate([iris.data, iris.target.reshape(-1, 1)], axis=1))
# # print(data.describe())
# print(iris.data[:2])
# # print(iris.target[:2])
# # print(iris.target_names)
# # del (iris.feature_names['sepal length (cm)'])
# print(iris.feature_names)
# # train_ratio = 0.8
# # x_train_data = out[:int(train_ratio*(out.shape[0]))]
# # x_eval_data = iris.data[int(train_ratio*(iris.data.shape[0])):]
# # print(x_train_data.shape)
# # print(x_eval_data.shape)
# # ss = [1,2,3]
# # retDataSet = []
# # ss.extend([4,5,6])
# # retDataSet.append(ss)
# # print(retDataSet)

from config import CONFIG
n_estimator = CONFIG['xgboost_cart']['params']['n_estimators']
print(n_estimator)