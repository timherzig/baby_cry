import torch

from torch import nn

from src.models.jdc.model import JDCNet


class BabyCryNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.architecture = config.model.architecture

        if self.architecture.lower() == "jdc":
            self.base_model = JDCNet(
                num_class=config.model.num_class,
            )
            params = torch.load(config.model.pretrained_path, map_location="cpu")["net"]
            self.base_model.load_state_dict(params)
        else:
            raise NotImplementedError(
                f"Model architecture {self.architecture} not implemented."
            )

    def forward(self, x):
        return self.base_model(x)


def get_model(config):
    return BabyCryNet(config)
