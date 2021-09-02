import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from dataset.torch_dataset_vgcn import TorchDataset
from dataset.tumor_dataset_vgcn import TumorDataset
from metrics.evaluate_cls import evaluate_multi_cls
from models.gcn_conv import GCNConv
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.metrics import confusion_matrix
import pandas as pd


# 2 layer mlp
# translayer -> adj
# 2 layer gcn
# linear pred


class Model(nn.Module):
    def __init__(self, **param_dict):
        super(Model, self).__init__()
        self.param_dict = param_dict
        self.input_dim = self.param_dict['ft_dim']
        self.out_dim = self.param_dict['label_num']
        self.h_dim = self.param_dict['h_dim']
        self.dropout_num = self.param_dict['dropout_num']
        self.add_res = self.param_dict['add_res']

        self.linear1 = nn.Linear(self.input_dim, self.h_dim)
        self.linear2 = nn.Linear(self.h_dim, self.h_dim)

        self.adj_trans_linear = nn.Linear(self.h_dim, self.h_dim)
        self.gcn_layer1 = GCNConv(self.h_dim, self.h_dim)
        self.gcn_layer2 = GCNConv(self.h_dim, self.h_dim)

        self.linear_pred = nn.Linear(self.h_dim, self.out_dim)
        self.activation = nn.ELU()
        self.dropout_layer = nn.Dropout(p=self.dropout_num)

    def forward(self, node_ft):

        res_mat = torch.zeros(node_ft.size()[0], self.h_dim).to(node_ft.device)

        node_ft = self.activation(self.linear1(node_ft))
        node_ft = self.dropout_layer(node_ft)

        node_ft = self.activation(self.linear2(node_ft))
        node_ft = self.dropout_layer(node_ft)

        res_mat += node_ft

        # adj
        trans_adj_ft = self.adj_trans_linear(node_ft)
        trans_adj_ft = torch.tanh(trans_adj_ft)
        w = torch.norm(trans_adj_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        adj = torch.mm(trans_adj_ft, trans_adj_ft.t()) / w_mat

        node_ft = self.activation(self.gcn_layer1(node_ft, adj))
        node_ft = self.dropout_layer(node_ft)
        res_mat += node_ft

        node_ft = self.activation(self.gcn_layer2(node_ft, adj))
        node_ft = self.dropout_layer(node_ft)
        res_mat += node_ft

        if self.add_res:
            node_embedding = res_mat
            pred = self.linear_pred(res_mat)
        else:
            node_embedding = node_ft
            pred = self.linear_pred(node_ft)
        pred = F.log_softmax(pred, dim=-1)
        return pred, adj, node_embedding


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
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'],
                                                         self.param_dict['batch_size'])
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
            pred, adj, node_embedding = self.model(ft_mat)
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
                adj_l1_loss = self.param_dict['adj_loss_coef'] * torch.norm(adj)
                loss = c_loss + param_l2_loss + adj_l1_loss
                # print('c_loss = ', c_loss.detach().to('cpu').item(),
                #       ' adj_l1_loss = ', adj_l1_loss.detach().to('cpu').item(),
                #       ' param_l2_loss = ', param_l2_loss.detach().to('cpu').item()
                #       )
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
        train_acc, valid_acc, test_primary_acc, test_transfer_acc, \
        train_macro_f1, valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1, \
        train_micro_p, valid_micro_p, test_primary_micro_p, test_transfer_micro_p, \
        train_micro_r, valid_micro_r, test_primary_micro_r, test_transfer_micro_r = res_list

        """
        msg_log = 'Epoch: {:03d}, Acc Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f}, Test transfer: {:.4f} ' \
                  'Macro F1 Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f}, Test transfer: {:.4f} ' \
                  'Micro P Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f}, Test transfer: {:.4f} ' \
                  'Micro R Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f}, Test transfer: {:.4f} '.format(
            epoch, train_acc, valid_acc, test_primary_acc, test_transfer_acc, \
            train_macro_f1, valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1, \
            train_micro_p, valid_micro_p, test_primary_micro_p, test_transfer_micro_p, \
            train_micro_r, valid_micro_r, test_primary_micro_r, test_transfer_micro_r)
        """
        msg_log = 'Epoch: {:03d}, Acc Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f} ' \
                  'Macro F1 Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f} ' \
                  'Micro P Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f} ' \
                  'Micro R Train: {:.4f}, Val: {:.4f}, Test primary: {:.4f} '.format(
            epoch, train_acc, valid_acc, test_primary_acc, \
            train_macro_f1, valid_macro_f1, test_primary_macro_f1, \
            train_micro_p, valid_micro_p, test_primary_micro_p, \
            train_micro_r, valid_micro_r, test_primary_micro_r)
        print(msg_log)

    def start(self, display=True):
        train_dataset = TorchDataset(dataset=self.dataset, split_type='train')
        train_dataloader = DataLoader(train_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        valid_dataset = TorchDataset(self.dataset, split_type='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.param_dict['batch_size'], shuffle=True)

        test_primary_dataset = TorchDataset(self.dataset, split_type='test_is_primary')
        test_primary_dataloader = DataLoader(test_primary_dataset, batch_size=self.param_dict['batch_size'],
                                             shuffle=True)

        test_transfer_dataset = TorchDataset(self.dataset, split_type='test_is_transfer')
        test_transfer_dataloader = DataLoader(test_transfer_dataset, batch_size=self.param_dict['batch_size'],
                                              shuffle=True)

        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            train_pred, train_label, train_loss = self.iteration(epoch=epoch, dataloader=train_dataloader,
                                                                 is_training=True)
            train_acc, train_micro_f1, train_macro_f1, train_micro_p, train_micro_r = evaluate_multi_cls(train_pred,
                                                                                                         train_label)

            valid_pred, valid_label, valid_loss = self.iteration(epoch=epoch, dataloader=valid_dataloader,
                                                                 is_training=False)
            valid_acc, valid_micro_f1, valid_macro_f1, valid_micro_p, valid_micro_r = evaluate_multi_cls(valid_pred,
                                                                                                         valid_label)

            test_primary_pred, test_primary_label, test_primary_loss = self.iteration(epoch=epoch,
                                                                                      dataloader=test_primary_dataloader,
                                                                                      is_training=False)
            test_primary_acc, test_primary_micro_f1, test_primary_macro_f1, test_primary_micro_p, test_primary_micro_r \
                = evaluate_multi_cls(test_primary_pred, test_primary_label)

            test_transfer_pred, test_transfer_label, test_transfer_loss = \
                self.iteration(epoch=epoch, dataloader=test_transfer_dataloader, is_training=False)
            test_transfer_acc, test_transfer_micro_f1, test_transfer_macro_f1, test_transfer_micro_p, test_transfer_micro_r \
                = evaluate_multi_cls(test_transfer_pred, test_transfer_label)

            res_list = [
                train_acc, valid_acc, test_primary_acc, test_transfer_acc,
                train_macro_f1, valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1,
                train_micro_p, valid_micro_p, test_primary_micro_p, test_transfer_micro_p,
                train_micro_r, valid_micro_r, test_primary_micro_r, test_transfer_micro_r
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

    def whole_graph_evaluate(self):
        # load_params
        model_param_path = osp.join(current_path, model_save_dir, self.trainer_info + '_param.pkl')
        print('load params ', model_param_path)
        self.model.load_state_dict(torch.load(model_param_path))
        self.dataset.to_tensor(self.device)


        self.model.eval()

        train_pred_prob, adj, train_embedding = self.model(self.dataset.ft_mat[self.dataset.train_index])
        max_value, train_pred = torch.max(train_pred_prob, dim=1)
        train_embedding = train_embedding.detach().to('cpu').numpy()

        np.save(
            osp.join(current_path, 'save_embedding_new_new', self.trainer_info + '_train_embedding.npy'),
            train_embedding
        )

        train_pred = train_pred.detach().to('cpu').numpy()
        train_label = self.dataset.label_mat[self.dataset.train_index].detach().to('cpu').numpy()
        print('train_pred = ', train_pred)
        print('train_label = ', train_label)

        cm = confusion_matrix(train_label, train_pred)
        print("confusion_matrix: ", cm)
        np.savetxt(self.trainer_info + '_train_cm.csv', cm, delimiter=',')

        test_pred_prob, adj, test_embedding = self.model(self.dataset.ft_mat[self.dataset.test_is_primary_idx])
        max_value, test_pred = torch.max(test_pred_prob, dim=1)
        test_embedding = test_embedding.detach().to('cpu').numpy()

        np.save(
            osp.join(current_path, 'save_embedding_new_new', self.trainer_info + '_test_embedding.npy'),
            test_embedding
        )

        test_pred = test_pred.detach().to('cpu').numpy()
        test_label = self.dataset.label_mat[self.dataset.test_is_primary_idx].detach().to('cpu').numpy()
        print('test_pred = ', test_pred)
        print('test_label = ', test_label)

        # file1 = open('pred.txt', 'w')
        # file1.write(str(test_pred))
        # file1.close()
        #
        # file2 = open('label.txt', 'w')
        # file2.write(str(test_label))
        # file2.close()


        cm = confusion_matrix(test_label, test_pred)
        print("confusion_matrix: ", cm)
        np.savetxt(self.trainer_info+'_test_cm.csv', cm, delimiter=',')

        all_pred_prob, adj, node_embedding = self.model(self.dataset.ft_mat)
        max_value, all_pred = torch.max(all_pred_prob, dim=1)

        all_pred = all_pred.detach().to('cpu').numpy()
        all_label = self.dataset.label_mat.detach().to('cpu').numpy()

        train_acc, train_micro_f1, train_macro_f1, train_micro_p, train_micro_r = \
            evaluate_multi_cls(all_pred[self.dataset.train_index], all_label[self.dataset.train_index])

        valid_acc, valid_micro_f1, valid_macro_f1, valid_micro_p, valid_micro_r = \
            evaluate_multi_cls(all_pred[self.dataset.valid_index], all_label[self.dataset.valid_index])

        test_primary_acc, test_primary_micro_f1, test_primary_macro_f1, test_primary_micro_p, test_primary_micro_r = \
            evaluate_multi_cls(all_pred[self.dataset.test_is_primary_idx], all_label[self.dataset.test_is_primary_idx])

        test_transfer_acc, test_transfer_micro_f1, test_transfer_macro_f1, test_transfer_micro_p, test_transfer_micro_r = \
            evaluate_multi_cls(all_pred[self.dataset.test_is_transfer_idx],
                               all_label[self.dataset.test_is_transfer_idx])

        print("*" * 10)
        print(train_micro_p, valid_micro_p, test_primary_micro_p, test_transfer_micro_p,
              train_micro_r, valid_micro_r, test_primary_micro_r, test_transfer_micro_r)
        print("*" * 10)

        res_list = [
            train_acc, valid_acc, test_primary_acc, test_transfer_acc,
            train_macro_f1, valid_macro_f1, test_primary_macro_f1, test_transfer_macro_f1,
            train_micro_p, valid_micro_p, test_primary_micro_p, test_transfer_micro_p,
            train_micro_r, valid_micro_r, test_primary_micro_r, test_transfer_micro_r
        ]

        print('whole_graph_evaluate')
        self.print_res(res_list, 0)


if __name__ == '__main__':
    for seed in [2, 3, 5]:
        param_dict = {
            'seed': seed,
            'train_split_param': [0.95, 0.05],
            'ft_stand': False,
            'dropout_num': 0.3,
            'layer_num': 4,
            'epoch_num': 200,
            'lr': 5e-4,
            'param_l2_coef': 1e-2,
            'batch_size': 1024,
            'h_dim': 512,
            'adj_loss_coef': 1e-2,
            'add_res': True,
        }
        trainer = Trainer(**param_dict)
        trainer.start()
        # trainer.whole_graph_evaluate()

