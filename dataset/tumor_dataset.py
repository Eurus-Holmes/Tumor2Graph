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
#train_data_file = 'dna_rna_methy_train.csv'
#test_data_file = 'dna_rna_methy_test.csv'
data_file = "df_all_new2.csv"

class TumorDataset(object):
    def __init__(self,
                 shuffle=True,
                 train_split_param=[0.9, 0.1],
                 ft_stand=True):

        self.shuffle = shuffle
        self.train_split_param = train_split_param
        self.ft_stand = ft_stand

        self.dataset_info = {}
        self.ft_mat = None
        self.img_name = None
        self.img_mat = None
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
        self.test_is_primary_idx = None
        self.test_is_transfer_idx = None

        self.raw_train_index = None
        self.raw_test_index = None

        self.load_dataset()
        if self.ft_stand:
            self.ft_standardization()

    def load_dataset(self):
#        train_df = pd.read_csv(osp.join(current_path, dataset_dir, train_data_file))
#        test_df = pd.read_csv(osp.join(current_path, dataset_dir, test_data_file))
        df_all = pd.read_csv(osp.join(current_path, dataset_dir, data_file))
        # print(list(train_df.columns))
        # print(list(extra_test_df.columns))
        train_df = df_all[df_all["split_type"]=="primary_train"]
        valid_df = df_all[df_all["split_type"]=="primary_valid"]
        test_df = df_all[df_all["split_type"]=="primary_test"]
        transfer_df = df_all[df_all["split_type"].isna()]
        test_df = pd.concat([test_df,transfer_df])
        print(train_df.shape)
        print(test_df.shape)
        
#        df = pd.concat([train_df, test_df])
#        self.raw_label = df['cancer_type'].to_numpy()
        
        self.train_index = np.arange(train_df.shape[0])
        self.valid_index = np.arange(valid_df.shape[0]) + train_df.shape[0]
        self.raw_test_index = np.arange(test_df.shape[0]) + train_df.shape[0]+valid_df.shape[0]
        # self.all_idx = np.hstack((self.raw_train_index, self.raw_test_index))
        df_all = pd.concat([train_df,valid_df,test_df])
        self.raw_label = df_all["num_label"].values
        self.all_sample_idx_str = df_all['sample'].to_numpy()
        # raw_train -> train_idx, valid_idx
        # raw_test -> test_is_primary_idx,  test_not_primary_idx
#        self.train_split_param.append(0)
#        train_index, valid_index, _ = train_test_split(
#            self.raw_train_index.shape[0], self.train_split_param, shuffle=self.shuffle)
#        self.train_index = self.raw_train_index[train_index]
#        self.valid_index = self.raw_train_index[valid_index]
#
        sample_is_primary = test_df['is_primary'].to_numpy()
#        # print('sample_is_primary = ', sample_is_primary)
        self.test_is_primary_idx = self.raw_test_index[sample_is_primary=='primary']
        self.test_is_transfer_idx = self.raw_test_index[sample_is_primary=='transfer']
        print("self.test_is_primary_idx.shape: ", self.test_is_primary_idx.shape)
        print("self.test_is_transfer_idx.shape: ", self.test_is_transfer_idx.shape)
#        # (1545,)(362, )(1907, )
        # print(self.test_is_primary_idx.shape, self.test_not_primary_idx.shape, self.raw_test_index.shape)
#        print(df.shape)
        df = df_all.drop(['cancer_type', 'sample', 'is_primary',"image_name","split_type","num_label"], axis=1)
        print(df.shape)
        self.ft_mat = df.to_numpy()
        self.img_name = df_all["image_name"].values
        self.ft_names = df.columns.to_numpy()
        
        # print(np.unique(self.raw_label[self.nonmetastatic_index]))
        # print(np.unique(self.raw_label[self.metastatic_index]))
        # same label

        enc = LabelEncoder()
        enc.fit(self.raw_label)
        self.label_mat = enc.transform(self.raw_label)
        self.label_coding = enc.classes_

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
    print(dataset.label_coding)
