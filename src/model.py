import torch

import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

class Whisper_Encoder(pl.LightningModule):
    '''
    Simple model implementation for testing purposes

    model2 gets augmented data and is trained to match the encoding whisper produces on clean data
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        # self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config.model.feature_extractor)

        self.model1 = WhisperForAudioClassification.from_pretrained(self.config.model.model).encoder
        self.model1._freeze_parameters
        self.model2 = WhisperForAudioClassification.from_pretrained(self.config.model.model).encoder

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters, lr=1e-3)
        return opt

    def training_step(self, batch, batch_idx):
        x1, x2, _ = batch
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        loss = F.mse_loss(out2, out1)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x1, x2, _ = batch
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        loss = F.mse_loss(out2, out1)
        return loss
    
    def forward(self, x):
        return self.model2.forward(x)