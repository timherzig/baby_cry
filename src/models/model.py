import torch

from torch import nn
from torch.nn import functional as F
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

        # self.p_dropout = nn.Dropout(p=self.config.dropout)

        # self.p_fc = nn.Linear(
        #     in_features=self.config.model.seq_len,
        #     out_features=self.config.model.seq_len,
        #     bias=False,
        # )

        # self.p_bn = nn.BatchNorm1d(self.config.model.seq_len, affine=False)

        # self.d_dropout = nn.Dropout(p=self.config.dropout)

        # self.d_fc = nn.Linear(
        #     in_features=self.config.model.seq_len,
        #     out_features=self.config.model.seq_len,
        #     bias=False,
        # )

        # self.d_bn = nn.BatchNorm1d(self.config.model.seq_len, affine=False)

        self.lstm_dropout = nn.Dropout(p=self.config.dropout)

        self.lstm_classifier = nn.LSTM(
            input_size=2,
            num_layers=config.model.bilstm.num_layers,
            hidden_size=config.model.bilstm.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(
                config.model.bilstm.hidden_size * 2 * config.model.bilstm.num_layers,
                config.model.bilstm.hidden_size * 2 * config.model.bilstm.num_layers,
                bias=True,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.bilstm.hidden_size * 2 * config.model.bilstm.num_layers,
                2,
                bias=False,
            ),
        )

    def forward(self, x):
        # (b, nmels, 192, 80) -> (b, nmels, 192) (Pitch sequence prediction/estimation)
        pitch_prediction = torch.zeros(
            (x.shape[0], x.shape[1], self.config.model.seq_len)
        ).to(x.device)
        for i in range(x.shape[1]):
            pitch, _, _ = self.base_model(x[:, i, :, :].unsqueeze(1))
            pitch_prediction[:, i, :] = pitch

        pitch_prediction /= (
            300  # normalize pitch (found 300 to be roughly the max pitch value)
        )

        # (b, nmels, 192) -> (b, 2, nmels, 192) (Delta features)
        delta = pitch_prediction[:, :, 1:] - pitch_prediction[:, :, :-1]
        delta = torch.cat([torch.zeros_like(delta[:, :, :1]), delta], dim=2)

        # pitch_prediction = self.p_dropout(pitch_prediction)
        # pitch_prediction = self.p_fc(pitch_prediction).transpose(1, 2)
        # pitch_prediction = self.p_bn(pitch_prediction).transpose(1, 2)

        # delta = self.d_dropout(delta)
        # delta = self.d_fc(delta).transpose(1, 2)
        # delta = self.d_bn(delta).transpose(1, 2)

        # (b, nmels, 192) -> (b, 2, nmels, 192) (Pitch prediction + Delta features)
        pitch_prediction = torch.stack([pitch_prediction, delta], dim=1)

        # (b, 2, nmels, 192) -> (b, 2, nmels * 192) -> (b, nmels * 192, 2)
        pitch_prediction = pitch_prediction.flatten(start_dim=2, end_dim=3)
        pitch_prediction = pitch_prediction.transpose(1, 2)

        # (b, nmels * 192, 2) -> (b, 2)  (Binary classification)
        pitch_prediction = self.lstm_dropout(pitch_prediction)
        _, (classifier_prediction, _) = self.lstm_classifier(pitch_prediction)
        classifier_prediction = classifier_prediction.transpose(0, 1).flatten(
            start_dim=1
        )
        classifier_prediction = self.fc(classifier_prediction)

        return classifier_prediction


def get_model(config):
    return BabyCryNet(config)
