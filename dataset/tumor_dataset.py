#!/usr/bin/env
# coding:utf-8

"""
Created on 2020/12/7 下午5:02

base Info
"""
__author__ = 'xx'
__version__ = '1.0'

import torch
import numpy as np
from collections import Counter
from dataset.dataset_utils import train_test_split
import os.path as osp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os.path as osp

current_path = osp.dirname(osp.realpath(__file__))
dataset_dir = 'v10'
train_data_file = 'dna_rna_methy_train.csv'
test_data_file = 'dna_rna_methy_test.csv'


class TumorDataset(object):
    def __init__(self,
                 shuffle=True,
                 split_param=[0.9, 0.05, 0.05],
                 ft_stand=True):

        self.shuffle = shuffle
        self.split_param = split_param
        self.ft_stand = ft_stand

        self.dataset_info = {}
        self.ft_mat = None
        self.label_mat = None
        self.raw_label = None
        self.ft_size = None
        self.label_size = None
        self.ft_names = None
        self.label_coding = None
        self.all_sample_idx_str = None
        self.record_num = None
        self.total_num = None
        self.dataset_name = 'Tumor_v10'
        self.col_name = None

        self.train_index = None
        self.valid_index = None
        self.test_index = None

        self.nonmetastatic_index = None
        self.metastatic_index = None

        self.load_dataset()
        if self.ft_stand:
            self.ft_standardization()

    def load_dataset(self):
        train_df = pd.read_csv(osp.join(current_path, dataset_dir, train_data_file))
        # print(train_df)
        extra_test_df = pd.read_csv(osp.join(current_path, dataset_dir, test_data_file))
        # print(list(train_df.columns))
        # print(list(extra_test_df.columns))
        df = pd.concat([train_df, extra_test_df])
        self.nonmetastatic_index = np.arange(train_df.shape[0])
        self.metastatic_index = np.arange(extra_test_df.shape[0]) + train_df.shape[0]
        self.all_idx = np.hstack((self.nonmetastatic_index, self.metastatic_index))

        self.raw_label = df['cancer_type'].to_numpy()
        self.all_sample_idx_str = df['sample'].to_numpy()

        df = df.drop(['cancer_type', 'sample', 'is_primary'], axis=1)
        self.ft_mat = df.to_numpy()
        self.ft_names = df.columns.to_numpy()

        # print(np.unique(self.raw_label[self.nonmetastatic_index]))
        # print(np.unique(self.raw_label[self.metastatic_index]))
        # same label

        enc = LabelEncoder()
        enc.fit(self.raw_label)
        self.label_mat = enc.transform(self.raw_label)
        self.label_coding = enc.classes_

        train_index, valid_index, test_index = train_test_split(
            self.nonmetastatic_index.shape[0], self.split_param, shuffle=self.shuffle)

        self.train_index = self.nonmetastatic_index[train_index]
        self.valid_index = self.nonmetastatic_index[valid_index]
        self.test_index = self.nonmetastatic_index[test_index]

        self.label_size = np.max(self.label_mat) + 1
        self.ft_size = self.ft_mat.shape[1]
        self.total_num = self.ft_mat.shape[0]

        # self.label_size = np.max(self.label_mat) + 1
        # self.ft_size = self.ft_mat.shape[1]
        # self.total_num = self.ft_mat.shape[0]
        # exit()

        msg = "dataset name is {}, ft size is {}, record_num is {}, label size is {}".format(
            self.dataset_name, self.ft_size, self.record_num, self.label_size
        )
        print(msg)


    def ft_standardization(self):
        self.stand_scaler = StandardScaler()
        self.stand_scaler.fit(self.ft_mat)
        self.ft_mat = self.stand_scaler.transform(self.ft_mat)
        # print(self.stand_scaler.mean_)
        # print(self.stand_scaler.var_)

    def to_tensor(self, device):
        self.ft_mat = torch.FloatTensor(self.ft_mat).to(device)
        self.label_mat = torch.LongTensor(self.label_mat).to(device)

    def generate_dataset_info(self):
        self.dataset_info['ft_dim'] = self.ft_mat.shape[-1]
        self.dataset_info['label_num'] = max(self.label_mat) + 1




if __name__ == '__main__':
    dataset = TumorDataset()