import os
import torch 

import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from data.my_dataset import MyDataset

class AugmentedDataModule(pl.LightningDataModule):
    '''
    Data Module for the augmented data prepared by Speech_Augment
    '''

    def __init__(self, root, batch_size=32):
        self.batch_size = batch_size
        self.root = root
        self.prepare_data_per_node = False

        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def setup(self, stage=None):
        self.train_augme = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.train_augme = self.train_augme[self.train_augme['augmented'] == True]
        self.train_augme = MyDataset(self.train_augme)
        self.train_clean = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.train_clean = self.train_clean[self.train_clean['augmented'] == False]
        self.train_clean = MyDataset(self.train_clean)

    def collate_fn(self, batch):
        path, audio1, label = zip(*batch)
        audio2 = [self.train_clean.get_specific_audio('_'.join(x.split('_')[:-1]) + '.' + x.split('.')[-1]) for x in path]

        print(f'audio1: {audio1}')
        print(f'audio2: {audio2}')

        return [audio2, audio1, label]
    
    def train_dataloader(self) :
        train_dl = DataLoader(self.train_augme, batch_size=self.batch_size, collate_fn=self.collate_fn)

        return train_dl