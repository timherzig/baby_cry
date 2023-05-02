import os
import torch 

import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from data.my_dataset import MyDataset

class AugmentedDataModule(pl.LightningDataModule):
    '''
    Data Module for the augmented data prepared by Speech Augment
    '''

    def __init__(self, root, batch_size=32):
        self.batch_size = batch_size
        self.root = root

    # def prepare_data(self) -> None:
    
    def setup(self):
        self.train_augme = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.train_augme = self.train_augme[self.train_augme['augmented'] == True]
        self.train_augme = MyDataset(self.train_augme)
        self.train_clean = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.train_clean = self.train_clean[self.train_clean['augmented'] == False]
        self.train_clean = MyDataset(self.train_clean)

    def collate_fn(self, batch):
        path, audio1, label = zip(*batch)
        audio2 = [self.train_clean.get_specific_audio('_'.join(x.split('_')[:-1]) + '.' + x.split('.')[-1]) for x in path]

        return torch.tensor(audio2), torch.tensor(audio1), torch.tensor(label)
    
    def train_dataloader(self) :
        train_dl = DataLoader(self.train_augme, batch_size=self.batch_size, collate_fn=self.collate_fn)

        return train_dl