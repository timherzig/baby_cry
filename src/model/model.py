import pytorch_lightning as pl

from src.model.jdc.jdc import JDC
from src.model.trill.trill import TRILL


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.model.name == "trill":
            self.base_model = TRILL(self.config)
        elif self.config.model.name == "jdc":
            self.base_model = JDC(self.config)

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass
