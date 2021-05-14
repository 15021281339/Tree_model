import tempfile
import numpy as np
from sklearn import datasets, metrics, model_selection
# tmp_dir = tempfile.mkdtemp()
import lgb_model
# exec = r"C:\Users\Administrator\Desktop\model-code\lgm\lightgbm.exe"

X, y = datasets.load_diabetes(return_X_y=True)
clf = lgb_model.LgbRegression()

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
print("Mean Square Error: ", metrics.mean_squared_error(y_test, clf.predict(x_test)))

