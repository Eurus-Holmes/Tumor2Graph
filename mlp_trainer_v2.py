#!/usr/bin/env
# coding:utf-8

"""
Created on 2020/12/7 下午5:38

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from dataset.torch_dataset_v2 import TorchDataset
from dataset.tumor_dataset_v2 import TumorDataset
from metrics.evaluate_cls import evaluate_multi_cls
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, **param_dict):
        super(Model, self).__init__()
        self.param_dict = param_dict
        self.input_dim = self.param_dict['ft_dim']
        self.out_dim = self.param_dict['label_num']
        self.h_dim = self.param_dict['h_dim']
        self.dropout_num = self.param_dict['dropout_num']
        self.layer_num = self.param_dict['layer_num']

        self.layer_list = nn.ModuleList()
        for idx in range(self.layer_num):
            in_size = self.h_dim
            out_size = self.h_dim
            if idx == 0:
                in_size = self.input_dim
            if idx == self.layer_num - 1:
                out_size = self.out_dim
            layer = nn.Linear(in_size, out_size)
            self.layer_list.append(layer)
        self.mlp_activation = nn.ELU()

    def forward(self, node_ft):
        H = node_ft
        for idx in range(self.layer_num):
            H = self.layer_list[idx](H)
            if idx != self.layer_num - 1:
                H = self.mlp_activation(H)
                H = F.dropout(H, p=self.dropout_num)
        H = F.log_softmax(H, dim=-1)
        return H


model_save_dir = 'save_model_param'
current_path = osp.dirname(osp.realpath(__file__))


class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = TumorDataset(
            train_split_param=self.param_dict['train_split_param'],
            ft_stand=self.param_dict['ft_stand']
        )
        self.dataset.generate_dataset_info()
        self.param_dict.update(self.dataset.dataset_info)
        # self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        # self.save_model_path = osp.join(current_path, model_save_dir, self.trainer_info)
        self.loss_op = torch.nn.NLLLoss()
        self.build_model()

    def build_model(self):
        self.model = Model(**self.param_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.min_dif = -1e10

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def iteration(self, epoch, dataloader, is_training=False):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        all_pred = []
        all_label = []
        all_loss = []
        for ft_mat, label_mat in dataloader:
            ft_mat = ft_mat.cuda().float()
            label_mat = label_mat.cuda().long()
            pred = self.model(ft_mat)
            if is_training:
                # print(pred.size(), label_mat.size())
                c_loss = self.loss_op(pred, label_mat)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)

                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                loss = c_loss + param_l2_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                all_loss.append(loss.detach().to('cpu').item())

            max_value, max_pos = torch.max(pred, dim=1)
            pred = max_pos.detach().to('cpu').numpy()
            label_mat = label_mat.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            all_label = np.hstack([all_label, label_mat])

        return all_pred, all_label, all_loss

    def print_res(self, res_list, epoch):
        train_acc, valid_acc, test_primary_acc, test_transfer_acc, train_macro_f1, \
        valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1 = res_list

        msg_log = 'Epoch: {:03d}, Acc Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f}, Test transfer: {:.4f} ' \
                  'Macro F1 Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f}, Test transfer: {:.4f}'.format(
            epoch, train_acc, valid_acc, test_primary_acc, test_transfer_acc, train_macro_f1, \
        valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1 )
        print(msg_log)

    def start(self, display=True):
        train_dataset = TorchDataset(dataset=self.dataset, split_type='train')
        train_dataloader = DataLoader(train_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        valid_dataset = TorchDataset(self.dataset, split_type='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        test_primary_dataset = TorchDataset(self.dataset, split_type='test_is_primary')
        test_primary_dataloader = DataLoader(test_primary_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        test_transfer_dataset = TorchDataset(self.dataset, split_type='test_is_transfer')
        test_transfer_dataloader = DataLoader(test_transfer_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            train_pred, train_label, train_loss = self.iteration(epoch=epoch, dataloader=train_dataloader, is_training=True)
            train_acc, train_micro_f1, train_macro_f1, train_micro_p, train_micro_r = evaluate_multi_cls(train_pred, train_label)

            valid_pred, valid_label, valid_loss = self.iteration(epoch=epoch, dataloader=valid_dataloader, is_training=False)
            valid_acc, valid_micro_f1, valid_macro_f1, valid_micro_p, valid_micro_r = evaluate_multi_cls(valid_pred, valid_label)

            test_primary_pred, test_primary_label, test_primary_loss = self.iteration(epoch=epoch, dataloader=test_transfer_dataloader, is_training=False)
            test_primary_acc, test_primary_micro_f1, test_primary_macro_f1, test_primary_micro_p, test_primary_micro_r = evaluate_multi_cls(test_primary_pred, test_primary_label)

            test_transfer_pred, test_transfer_label, test_transfer_loss = \
                self.iteration(epoch=epoch, dataloader=test_primary_dataloader, is_training=False)
            test_transfer_acc, test_transfer_micro_f1, test_transfer_macro_f1, test_transfer_micro_p, test_transfer_micro_r\
                = evaluate_multi_cls(test_transfer_pred, test_transfer_label)

            res_list = [
                train_acc, valid_acc, test_primary_acc, test_transfer_acc, train_macro_f1, \
                valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1
            ]

            if valid_acc > self.min_dif:
                self.min_dif = valid_acc
                self.best_res = res_list
                self.best_epoch = epoch
                # save model
                # save_complete_model_path = osp.join(current_path, model_save_dir, self.trainer_info + '_complete.pkl')
                # torch.save(self.model, save_complete_model_path)
                same_model_param_path = osp.join(current_path, model_save_dir, self.trainer_info + '_param.pkl')
                torch.save(self.model.state_dict(), same_model_param_path)

            if display:
                self.print_res(res_list, epoch)

            if epoch % 50 == 0 and epoch > 0:
                print('Best res')
                self.print_res(self.best_res, self.best_epoch)


if __name__ == '__main__':
    param_dict = {
        'seed': 3,
        'train_split_param': [0.9, 0.1],
        'ft_stand': True,
        'dropout_num': 0.3,
        'layer_num': 4,
        'epoch_num': 200,
        'lr': 1e-4,
        'param_l2_coef': 1e-3,
        'batch_size': 128,
        'h_dim': 512
    }
    trainer = Trainer(**param_dict)
    trainer.start()







