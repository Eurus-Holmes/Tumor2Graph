#!/usr/bin/env
# coding:utf-8

"""
Created on 2020/11/12 上午11:17

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch
import torch.nn as nn
import torch.nn.functional as F


class DropEdge(nn.Module):
    def __init__(self, p=0.3):
        super(DropEdge, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def forward(self, adj):
        node_num = adj.size()[0]
        identity_mat = torch.eye(node_num).to(adj.device)
        # print(adj)
        without_self_loop_adj = adj - identity_mat
        # print(without_self_loop_adj)
        without_self_loop_adj = self.dropout(without_self_loop_adj)
        return without_self_loop_adj + identity_mat



if __name__ == '__main__':
    adj = torch.ones(10)
    dropedge = DropEdge(p=0.5)
    print(dropedge(adj))