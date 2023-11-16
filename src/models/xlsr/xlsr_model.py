from transformers import Wav2Vec2ForXVector, Wav2Vec2Config


class XLSRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
