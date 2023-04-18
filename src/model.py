import pytorch_lightning as pl

class model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def configure_optimizers(self):

    def train_dataloader(self):

    def val_dataloader(self):

    def training_step(self, batch, batch_idx):

    def validation_step(self, batch, batch_idx, dataloader_idx=0):