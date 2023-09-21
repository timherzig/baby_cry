from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets.baby_cry import BabyCryDataset


class DataModule(LightningDataModule):
    def __init__(self, config, num_workers=0):
        super().__init__()
        self.config = config
        self.dataset = config.data.dataset
        self.num_workers = num_workers

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.collate_fn = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.dataset == "baby_cry":
            if stage == "fit":
                self.data_train = BabyCryDataset(self.config, "train")
                self.data_val = BabyCryDataset(self.config, "val")
            elif stage == "test":
                self.data_test = BabyCryDataset(self.config, "test")
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.config.train_args.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.config.train_args.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.config.train_args.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
