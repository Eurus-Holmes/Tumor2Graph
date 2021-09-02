from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import random
import torch
import math
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys
from dataset.tumor_dataset_v2 import TumorDataset
from metrics.evaluate_cls import evaluate_multi_cls
current_path = osp.dirname(osp.realpath(__file__))

class LRTrainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = TumorDataset(
            train_split_param=self.param_dict['train_split_param'],
            ft_stand=self.param_dict['ft_stand']
        )
        self.model = RandomForestClassifier(n_estimators=self.param_dict['n_estimators'])

    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def fit(self):
        self.model.fit(
            X=self.dataset.ft_mat[self.dataset.train_index],
            y=self.dataset.label_mat[self.dataset.train_index]
        )

    def evaluate(self):
        train_pred = self.model.predict(self.dataset.ft_mat[self.dataset.train_index])
        valid_pred = self.model.predict(self.dataset.ft_mat[self.dataset.valid_index])
        test_primary_pred = self.model.predict(self.dataset.ft_mat[self.dataset.test_is_primary_idx])
        test_transfer_pred = self.model.predict(self.dataset.ft_mat[self.dataset.test_is_transfer_idx])

        print(evaluate_multi_cls(train_pred, self.dataset.label_mat[self.dataset.train_index]))
        print(evaluate_multi_cls(valid_pred, self.dataset.label_mat[self.dataset.valid_index]))
        print(evaluate_multi_cls(test_primary_pred, self.dataset.label_mat[self.dataset.test_is_primary_idx]))
        print(evaluate_multi_cls(test_transfer_pred, self.dataset.label_mat[self.dataset.test_is_transfer_idx]))


if __name__ == '__main__':
    param_dict = {
        'seed': 2,
        'train_split_param': [0.9, 0.1],
        'ft_stand': True,
        'n_estimators': 500
    }
    trainer = LRTrainer(**param_dict)
    trainer.fit()
    trainer.evaluate()
