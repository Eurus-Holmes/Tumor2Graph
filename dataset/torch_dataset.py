#!/usr/bin/env
# coding:utf-8

"""
Created on 2020/12/7 下午7:38

base Info
"""
__author__ = 'xx'
__version__ = '1.0'
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, dataset, split_type):
        self.dataset = dataset
        self.split_type = split_type
        assert split_type in ['train', 'valid', 'test', 'metastatic']
        if self.split_type == 'train':
            self.select_idx = self.dataset.train_index
        elif self.split_type == 'valid':
            self.select_idx = self.dataset.valid_index
        elif self.split_type == 'test':
            self.select_idx = self.dataset.test_index
        else:
            self.select_idx = self.dataset.metastatic_index

    def __len__(self):
        return self.select_idx.shape[0]

    def __getitem__(self, idx):
        ft = self.dataset.ft_mat[idx]
        label = self.dataset.label_mat[idx]
        return ft, label

