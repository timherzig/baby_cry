import torch
import torch.nn as nn


class TRILL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(self.config.model.pretrained, str):
            self.base_trill = torch.load(self.config.model.pretrained)
        else:
            raise NotImplementedError

    def forward(self, x):
        pass
