import torch

from torch import nn
import matplotlib.pyplot as plt

from src.models.jdc.model import JDCNet


class BabyCryNet(nn.Module):
    def __init__(self, config, load_pretrained=True):
        super().__init__()
        self.config = config
        self.architecture = config.model.architecture

        if self.architecture.lower() == "jdc":
            self.base_model = JDCNet(
                num_class=config.model.num_classes, seq_len=config.model.seq_len
            )
            if load_pretrained:
                params = torch.load(config.model.pretrained, map_location="cpu")["net"]
                self.base_model.load_state_dict(params)
            # freeze base model
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(
                f"Model architecture {self.architecture} not implemented."
            )

        # TODO: add classifier head (RNN/GNU/BI-LSTM)
        self.lstm_classifier = nn.LSTM(
            input_size=2,
            hidden_size=config.model.bilstm.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(config.model.bilstm.hidden_size * 2, 1)

    def forward(self, x):
        # (b, 1, 192, 80) -> (b, 192) (Pitch sequence prediction/estimation)
        pitch_prediction, _, _ = self.base_model(x)

        # (b, 192) -> (b, 192, 2) (Delta features)
        delta = pitch_prediction[:, 1:] - pitch_prediction[:, :-1]
        delta = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1)
        pitch_prediction = torch.stack([pitch_prediction, delta], dim=2)

        # (b, 192, 2) -> (b)  (Binary classification)
        classifier_prediction, _ = self.lstm_classifier(pitch_prediction)
        classifier_prediction = classifier_prediction[:, -1, :]
        classifier_prediction = self.fc(classifier_prediction).squeeze(1)
        classifier_prediction = torch.sigmoid(classifier_prediction)

        return classifier_prediction


def get_model(config):
    return BabyCryNet(config)
