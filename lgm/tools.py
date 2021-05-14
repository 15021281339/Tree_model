import os
import numpy as np
from sklearn import datasets


# %% 判断稀疏性 导入不同数据格式
def dump_data(X, y, f, issparse=False):
    """
    store data into CSV file and sparse data into SVM
    """
    if issparse:
        datasets.dump_svmlight_file(X, y, f)
    else:
        # Label is the data of first column, and there is no header in the file.
        np.savetxt(f, X=np.column_stack((y, X)), delimiter=',', newline=os.linesep)
