from __future__ import print_function
import tempfile
import os
import re
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import scipy.sparse as sps
import tools
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from config import CONFIG


# %% 基于Microsoft LightGBM  需要pip install lightgbm
class GeneralLgb(BaseEstimator):
    def __init__(self, config='', exc_path='./lightgbm.exe', verbose=True, model=None):
        self.model = model
        self.verbose = verbose
        self.config = config  # 若输入为''，则读取存储好的train.conf,否则为用户输入参数
        self.param = CONFIG['lightgbm']['params']  # 写到conf文件下 直接调用
        self.exec_path = exc_path
        try:
            self.exec_path = os.environ['LIGHTGBM_EXEC']
        except KeyError:
            print("pyLightGBM is looking for 'LIGHTGBM_EXEC' environment variable, cannot be found.")
            print("exec_path will be deprecated in favor of environment variable")
            self.exec_path = os.path.expanduser(exc_path)

    # %% 创建临时文件存储数据和模型
    def fit(self, x, y, test_data=None, init_scores=[]):
        tmp_dir = tempfile.mkdtemp()  # 创建临时文件
        issparse = sps.issparse(x)
        f_format = 'svm' if issparse else 'csv'

        train_filepath = os.path.abspath('{}/x.{}'.format(tmp_dir, f_format))  # 训练数据存储位置
        init_filepath = train_filepath + '.init'
        tools.dump_data(x, y, train_filepath, issparse)

        if len(init_scores) > 0:
            assert len(init_scores) == x.shape[0]  # 控制为样本数量
            np.savetxt(init_filepath, x=init_scores, delimiter=',', newline=os.linesep)

        if test_data:
            valid = []
            for i, (x_test, y_test) in enumerate(test_data):  # test_data = [x_test, y_test]
                test_filepath = os.path.abspath('{}/x{}_test.{}'.format(tmp_dir, i, f_format))  # 训练数据存储位置
                valid.append(test_filepath)
                tools.dump_data(x_test, y_test, test_filepath, issparse)
            self.param['valid'] = ','.join(valid)  # 验证路径

        self.param['task'] = 'train'
        self.param['data'] = train_filepath
        self.param['output_model'] = os.path.join(tmp_dir, 'lightgbm_model.txt')
        calls = ['{}={}\n'.format(k, self.param[k]) for k in self.param]  # 训练参数

        if self.config == '':  # 如果没有输入参数 则读取存储的默认训练参数
            conf_filepath = os.path.join(tmp_dir, 'train_conf')  # 存储 训练参数
            with open(conf_filepath, 'w') as f:
                f.writelines(calls)

            process = subprocess.Popen([self.exec_path, 'config = {}'.format(conf_filepath)],
                                       stdout=subprocess.PIPE, bufsize=1)  # 打开执行路径，加载训练参数，输出管道PIPE
        else:
            process = subprocess.Popen([self.exec_path, 'config = {}'.format(self.config)],
                                       stdout=subprocess.PIPE, bufsize=1)

        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.strip().decode('utf-8')) if self.verbose else None
        process.wait()

        with open(self.param['output_model'], mode='r') as f:  # 模型读取赋给self.model
            self.model = f.read()
        shutil.rmtree(tmp_dir)  # 递归删除

        if test_data and self.param['early_stopping_round'] > 0:
            self.best_round = max(map(int, re.findall('Tree=(\d+)', self.model))) + 1

    def predict(self, x):
        tmp_dir = tempfile.mkdtemp()
        issparse = sps.issparse(x)
        f_format = 'svm' if issparse else 'csv'

        predict_filepath = os.path.abspath(os.path.join(tmp_dir, 'x_to_pred.{}'.format(f_format)))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        output_results = os.path.abspath(os.path.join(tmp_dir, "LightGBM_predict_result.txt"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")

        # 写入模型 到文件中
        with open(output_model, mode='w') as file:
            file.write(self.model)

        # 将输入数据写入到文件中
        tools.dump_data(x, np.zeros(x.shape[0]), predict_filepath, issparse)

        calls = ['task = predict\n',
                 'data = {}\n'.format(predict_filepath),
                 'input_model = {}\n'.format(output_model),
                 'output_result = {}\n'.format(output_results)]

        # 写入配置信息calls到文件中
        with open(conf_filepath, 'w') as f:
            f.writelines(calls)

        # 报错 无法实现预测
        process = subprocess.Popen([self.exec_path, 'config={}'.format(conf_filepath)],
                                   stdout=subprocess.PIPE, bufsize=1)

        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.strip().decode('utf-8')) if self.verbose else None
        process.wait()
        # process.communicate()

        y_pred = np.loadtxt(output_results, dtype=float)
        shutil.rmtree(tmp_dir)

        return y_pred

    def get_params(self, deep=True):
        params = dict(self.param)
        params['exec_path'] = self.exec_path
        params['config'] = self.config
        params['model'] = self.model
        params['verbose'] = self.verbose
        if 'output_model' in params:
            del params['output_model']
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(params)
        self.__init__(**params)
        return self

    def feat_importance(self, feat_names=[], importance_type='weight'):
        assert importance_type in ['weight'], 'For now, only weighted feature importance is implemented'

        match = re.findall("Column_(\d+)=(\d+)", self.model)
        if importance_type == 'weight':
            if len(match) > 0:
                dic_fi = {int(k): int(value) for k, value in match}
                if len(feat_names) > 0:
                    dic_fi = {feat_names[key]: dic_fi[key] for key in dic_fi}
            else:
                dic_fi = {}
        return dic_fi


class LgbClassifier(GeneralLgb, ClassifierMixin):
    def __init__(self, config='', exc_path='./lightgbm.exe', verbose=True, model=None):
        super(LgbClassifier, self).__init__(config='', exc_path='./lightgbm.exe', verbose=True, model=None)

    def predict_pro(self, x):
        tmp_dir = tempfile.mkdtemp()
        issparse = sps.issparse(x)
        f_format = "svm" if issparse else "csv"

        predict_filepath = os.path.abspath(os.path.join(tmp_dir, "x_to_pred.{}".format(f_format)))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")
        output_results = os.path.abspath(os.path.join(tmp_dir, "LightGBM_predict_result.txt"))

        with open(output_model, mode="w") as file:
            file.write(self.model)

        tools.dump_data(x, np.zeros(x.shape[0]), predict_filepath, issparse)

        calls = [
            "task = predict\n",
            "data = {}\n".format(predict_filepath),
            "input_model = {}\n".format(output_model),
            "output_result={}\n".format(output_results)
        ]

        with open(conf_filepath, 'w') as f:
            f.writelines(calls)

        process = subprocess.Popen([self.exec_path, "config={}".format(conf_filepath)],
                                   stdout=subprocess.PIPE, bufsize=1)

        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.strip().decode('utf-8')) if self.verbose else None
        # wait for the subprocess to exit
        process.wait()

        raw_probabilities = np.loadtxt(output_results, dtype=float)

        if self.param['application'] == 'multiclass':
            y_prob = raw_probabilities

        elif self.param['application'] == 'binary':
            probability_of_one = raw_probabilities
            probability_of_zero = 1 - probability_of_one
            y_prob = np.transpose(np.vstack((probability_of_zero, probability_of_one)))
        else:
            raise

        shutil.rmtree(tmp_dir)
        return y_prob

    def predict(self, x):
        y_prob = self.predict_pro(x)
        return y_prob.argmax(-1)


class LgbRegression(GeneralLgb, RegressorMixin):
    def __init__(self, config='', exc_path='./lightgbm.exe', verbose=True, model=None):
        super(LgbRegression, self).__init__(config='', exc_path='./lightgbm.exe', verbose=True, model=None)


if __name__ == '__main__':

    from sklearn import datasets, metrics, model_selection
    import lgb_model
    # exec = r"C:\Users\Administrator\Desktop\model-code\lgm\lightgbm.exe"

    X, y = datasets.load_diabetes(return_X_y=True)
    clf = lgb_model.LgbRegression()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf.fit(x_train, y_train, test_data=[(x_test, y_test)])
    print("Mean Square Error: ", metrics.mean_squared_error(y_test, clf.predict(x_test)))