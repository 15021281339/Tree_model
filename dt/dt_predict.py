import pickle
from dt_train import DTTrainer
from create_data import CreateData


# %% 决策树-样本案例预测  test_data=[[]]
def predict(test_data, pathname):
    with open(pathname + 'dt_model.pkl', 'rb') as dt:
        my_tree = pickle.load(dt)

    with open(pathname + 'dt_config.pkl', 'rb') as f:
        best_fea_labels = pickle.load(f)

    new_trainer = DTTrainer()
    res = []
    for i in test_data:
        result = new_trainer.classify(my_tree, best_fea_labels, i)
        res.append(result)

    return res
