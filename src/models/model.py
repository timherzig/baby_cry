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

        self.lstm_classifier = nn.LSTM(
            input_size=2,
            num_layers=config.model.bilstm.num_layers,
            hidden_size=config.model.bilstm.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(
            config.model.bilstm.hidden_size * 2 * config.model.bilstm.num_layers,
            2,
            bias=False,
        )

        # self.softmax = nn.Softmax(dim=1)

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
        pitch_prediction = torch.stack([pitch_prediction, delta], dim=1)

        # (b, 2, nmels, 192) -> (b, nmels, 2, 192) -> (b, nmels * 192, 2)
        pitch_prediction = pitch_prediction.flatten(start_dim=2, end_dim=3)
        pitch_prediction = pitch_prediction.transpose(1, 2)

        # (b, nmels * 192, 2) -> (b, 2)  (Binary classification)
        _, (classifier_prediction, _) = self.lstm_classifier(pitch_prediction)
        classifier_prediction = classifier_prediction.transpose(0, 1).flatten(
            start_dim=1
        )
        classifier_prediction = self.fc(classifier_prediction)

        return classifier_prediction


def get_model(config):
    return BabyCryNet(config)
