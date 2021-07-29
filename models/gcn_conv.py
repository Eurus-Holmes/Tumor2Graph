#!/usr/bin/env
# coding:utf-8
"""
Created on 2020/8/25 10:55

base Info
"""
__author__ = 'xx'
__version__ = '1.0'


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

def get_laplace_mat(adj_mat, type='sym', add_i=False, degree_version='v1', requires_grad=True):
    if type == 'sym':
        # Symmetric normalized Laplacian
        if add_i is True:
            adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        else:
            adj_mat_hat = adj_mat
        # adj_mat_hat = adj_mat_hat[adj_mat_hat > 0]
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5, degree_version=degree_version)
        # print(degree_mat_hat.dtype, adj_mat_hat.dtype)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        # print(laplace_mat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)

    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)

        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)

    else:
        exit('error laplace mat type')

    if requires_grad:
        return laplace_mat
    else:
        return laplace_mat.detach()


# @torch.no_grad()
def get_degree_mat(adj_mat, pow=1, degree_version='v1'):
    degree_mat = torch.eye(adj_mat.size()[0]).to(adj_mat.device)

    if degree_version == 'v1':
        degree_list = torch.sum((adj_mat > 0), dim=1).float()
    elif degree_version == 'v2':
        # adj_mat_hat = adj_mat.data
        # adj_mat_hat[adj_mat_hat < 0] = 0
        adj_mat_hat = F.relu(adj_mat)
        degree_list = torch.sum(adj_mat_hat, dim=1).float()
    else:
        exit('error degree_version ' + degree_version)
    degree_list = torch.pow(degree_list, pow)
    degree_mat = degree_mat * degree_list
    # degree_mat = torch.pow(degree_mat, pow)
    # degree_mat[degree_mat == float("Inf")] = 0
    # degree_mat.requires_grad = False
    # print('degree_mat = ', degree_mat)
    return degree_mat


class GCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 init_type='v1',
                 degree_version='v1'
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.degree_version = degree_version
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )

        if init_type == 'v1':
            # bound = (1/in_channels)**0.5
            bound = (1 / in_channels)
            nn.init.uniform_(self.weight, -bound, bound)
            if bias is True:
                self.bias = Parameter(torch.Tensor(out_channels))
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            nn.init.xavier_normal_(self.weight)
            if bias is True:
                self.bias = Parameter(torch.Tensor(out_channels))
                nn.init.zeros_(self.bias)

    def forward(self, node_state, adj_mat):
        adj_mat = get_laplace_mat(adj_mat, type='sym', degree_version=self.degree_version)
        # print('adj_mat.grad = ', adj_mat.grad)
        node_state = torch.mm(adj_mat, node_state)
        node_state = torch.mm(node_state, self.weight)
        if self.bias is not None:
            node_state = node_state + self.bias
        return node_state


class BatchGCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 init_type='v1',
                 degree_version='v1'):
        super(BatchGCNConv, self).__init__()
        self.gcn_layer = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            init_type=init_type,
            degree_version=degree_version)

    def forward(self, batch_ft_mat, batch_adj):
        batch_size = batch_ft_mat.size()[0]
        res_mat = []

        for idx in range(batch_size):
            ft_mat = self.gcn_layer(batch_ft_mat[idx], batch_adj[idx]).unsqueeze(0)
            res_mat.append(ft_mat)

        ft_mat = torch.cat(res_mat, dim=0)
        return ft_mat