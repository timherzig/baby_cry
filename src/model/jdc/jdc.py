import torch
import torch.nn as nn


class JDC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(self.config.model.pretrained, str):
            self.base_jdc = torch.load(self.config.model.pretrained)
        else:
            raise NotImplementedError

        # TODO: Stack/extend the output in a temporal domain to match the input of the potential BiLSTM

    def forward(self, x):
        pass
