from dataset.abs_dataset_cls_v2 import AbsDataset
from dataset.abs_dataset_pdb import PDBDataset
import numpy as np
import random
import torch
import math
import os.path as osp

class ExtraEvaluate(object):
    def __init__(self, model_file_path, param_dict, model):
        self.param_dict = param_dict

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hiv_dataset = AbsDataset(
            max_antibody_len=self.param_dict['max_antibody_len'],
            max_virus_len=self.param_dict['max_virus_len'],
            train_split_param=self.param_dict['hot_data_split'],
            label_type=self.param_dict['label_type'],
            kmer_min_df=self.param_dict['kmer_min_df'],
            reprocess=False
        )  # same param with training
        self.hiv_dataset.to_tensor(self.device)
        self.model = model
        #
        print(self.hiv_dataset.dataset_info)
        self.param_dict = param_dict
        self.param_dict.update(self.hiv_dataset.dataset_info)

        # model_file_path = osp.join(current_path, select_model_param_dir, model_complete_file_name)
        self.model = self.model(**self.param_dict).to(self.device)
        self.model.load_state_dict(torch.load(model_file_path))
        self.model.eval()

    def extra_test_with_graph(self, extra_dataset):
        hiv_antibody_graph_node = self.hiv_dataset.protein_ft_dict['antibody_kmer_whole'][
            self.hiv_dataset.known_antibody_idx]
        hiv_virus_graph_node = self.hiv_dataset.protein_ft_dict['virus_kmer_whole'][self.hiv_dataset.known_virus_idx]

        extra_antibody_graph_node = extra_dataset.protein_ft_dict['antibody_kmer_whole']
        extra_virus_graph_node = extra_dataset.protein_ft_dict['virus_kmer_whole']

        # print('hiv_virus_graph_node = ', hiv_virus_graph_node.size())
        # print('extra_antibody_graph_node = ', extra_antibody_graph_node.size())
        antibody_graph_node_ft = torch.cat([hiv_antibody_graph_node, extra_antibody_graph_node], dim=0)
        virus_graph_node_ft = torch.cat([hiv_virus_graph_node, extra_virus_graph_node], dim=0)

        all_pred = []
        pair_num = extra_dataset.virus_index_in_pair.shape[0]
        for i in range(math.ceil(pair_num / self.param_dict['batch_size'])):
            right_bound = min((i + 1) * self.param_dict['batch_size'], pair_num + 1)

            antibody_idx = extra_dataset.antibody_index_in_pair[i * self.param_dict['batch_size']: right_bound]
            virus_idx = extra_dataset.virus_index_in_pair[i * self.param_dict['batch_size']: right_bound]

            batch_antibody_node_idx_in_graph = antibody_idx + self.hiv_dataset.known_antibody_idx.shape[0]
            batch_virus_node_idx_in_graph = virus_idx + self.hiv_dataset.known_virus_idx.shape[0]

            batch_antibody_amino_ft = extra_dataset.protein_ft_dict['antibody_amino_num'][antibody_idx]
            batch_virus_amino_ft = extra_dataset.protein_ft_dict['virus_amino_num'][virus_idx]


            ft_dict = {
                'antibody_graph_node_kmer_ft': antibody_graph_node_ft,
                'virus_graph_node_kmer_ft': virus_graph_node_ft,
                # 'antibody_graph_node_pssm_ft': antibody_graph_node_pssm_ft,
                # 'virus_graph_node_pssm_ft': virus_graph_node_pssm_ft,
                'antibody_amino_ft': batch_antibody_amino_ft,
                'virus_amino_ft': batch_virus_amino_ft,
                'antibody_idx': batch_antibody_node_idx_in_graph,
                'virus_idx': batch_virus_node_idx_in_graph
            }

            pred, antibody_adj, virus_adj, pair_ft = self.model(**ft_dict)
            pred = pred.view(-1)
            pred = pred.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])

        print('select_type :',  extra_dataset.dataset_select_type)
        # print('all_pred = ', all_pred)
        # print('all_label_mat = ', extra_dataset.all_label_mat)
        # evaluate

        # print('ave value = ', all_pred[all_pred>0.5].shape[0] / extra_dataset.all_label_mat.shape[0])

        TP = all_pred[all_pred > 0.5].shape[0]
        P = extra_dataset.all_label_mat.shape[0]
        recall_dict[extra_dataset.dataset_select_type[0]] = [
            extra_dataset.dataset_select_type[0], TP, P, TP / P
        ]  # TP, P, recall
        print(extra_dataset.dataset_select_type[0], TP, P, TP / P)
