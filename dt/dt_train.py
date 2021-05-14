import pickle
# from sklearn.datasets import load_iris
from decisiontree_classify import DeciTree
from create_data import CreateData


# %% 决策树-训练器
class DTTrainer():
    def __init__(self, data_path=None):
        self.data_path = data_path

    def load_data(self):
        new_data = CreateData()
        self.datasets, self.fea_labels = new_data.getdata()

    def train(self):

        clf = DeciTree()
        self.best_fea_labels = []
        self.my_tree = clf.create_Tree(self.datasets, self.fea_labels, self.best_fea_labels)

    def save_model(self, filepath):
        with open(filepath+'dt_model.pkl', 'wb') as dt:
            pickle.dump(self.my_tree, dt)

        with open(filepath+'dt_config.pkl', 'wb') as f:
            pickle.dump(self.best_fea_labels, dt)   # 存储 按顺序输出的最优特征标签

    def run(self):
        self.load_data()
        self.train()
        self.save_model()


