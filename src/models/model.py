import torch

from torch import nn

from src.models.jdc.model import JDCNet


class BabyCryNet(nn.Module):
    def __init__(self, config, load_pretrained=True):
        super().__init__()
        self.config = config

        self.architecture = config.model.architecture

        if self.architecture.lower() == "jdc":
            self.base_model = JDCNet(
                num_class=config.model.num_classes,
            )
            if load_pretrained:
                params = torch.load(config.model.pretrained, map_location="cpu")["net"]
                self.base_model.load_state_dict(params)
            # # freeze base model
            # for param in self.base_model.parameters():
            #     param.requires_grad = False
        else:
            raise NotImplementedError(
                f"Model architecture {self.architecture} not implemented."
            )

        self.linear = nn.Linear(
            in_features=31,
            out_features=config.model.num_classes,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pitch_prediction = self.base_model(x)
        pitch_prediction = pitch_prediction.squeeze()
        pitch_prediction = self.linear(pitch_prediction).squeeze()
        return self.sigmoid(pitch_prediction)


def get_model(config):
    return BabyCryNet(config)
