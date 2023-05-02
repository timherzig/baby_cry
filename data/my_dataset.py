import torchaudio

import pandas as pd

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio, sr = torchaudio.load(row['path'])
        label = row['label']

        return row['path'], audio, label

    def __len__(self):
        return len(self.df)
    
    def get_specific_audio(self, x):
        # path = self.df.loc[self.df['path'] == x].at['path']
        # print(path)
        audio, sr = torchaudio.load(x)

        return audio