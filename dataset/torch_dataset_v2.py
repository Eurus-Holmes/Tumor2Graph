import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
class TorchDataset(Dataset):
    def __init__(self, dataset, split_type):
        self.dataset = dataset
        self.split_type = split_type
        self.folder_path ="/mnt/lustreold/chenfeiyang/image_deconvolved"
        assert split_type in ['train', 'valid', 'test_is_primary', 'test_is_transfer']
        if self.split_type == 'train':
            self.select_idx = self.dataset.train_index
        elif self.split_type == 'valid':
            self.select_idx = self.dataset.valid_index
        elif self.split_type == 'test_is_primary':
            self.select_idx = self.dataset.test_is_primary_idx
        else:
            self.select_idx = self.dataset.test_is_transfer_idx
        self.image_name = self.dataset.img_name
    def __len__(self):
        return self.select_idx.shape[0]

    def __getitem__(self, idx):
        
        img_name = self.image_name[self.select_idx[idx]]
        self.image_list = []
        image_size = 32
#        for name in img_name:
        
#            img_name = os.listdir(folder_path)
        name_path = os.path.join(self.folder_path,img_name+".npy")
        image = np.load(name_path)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        result = np.zeros(image.shape,dtype=np.float32)
        cv2.normalize(image, result, 0, 1, cv2.NORM_MINMAX,dtype =cv2.CV_32F )
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        mean = image_size*[image_size*[mean]]
        std = image_size*[image_size*[std]]
        image = (result - mean) / std
        
        
        
        ft = self.dataset.ft_mat[self.select_idx[idx]]
        img = image
        label = self.dataset.label_mat[self.select_idx[idx]]
        
        return ft,img, label
