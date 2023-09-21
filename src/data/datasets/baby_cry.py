import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class BabyCryDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.data = pd.read_csv(os.path.join(self.config.data.path, f"{split}.csv"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
