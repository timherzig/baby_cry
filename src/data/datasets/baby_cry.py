import os
import librosa

import numpy as np
import pandas as pd

from madmom.audio.signal import *

from torch.utils.data import Dataset


def spec_extraction(file_name, win_size):
    # print(file_name)

    x_test = []
    # y, sr = librosa.load(file_name, sr=8000)
    # madmom.Signal() is faster than librosa.load()
    y = Signal(file_name, sample_rate=8000, dtype=np.float32, num_channels=1)
    S = librosa.core.stft(y, n_fft=1024, hop_length=80 * 1, win_length=1024)

    x_spec = np.abs(S)
    x_spec = librosa.core.power_to_db(x_spec, ref=np.max)
    x_spec = x_spec.astype(np.float32)
    num_frames = x_spec.shape[1]

    # for padding
    padNum = num_frames % win_size
    if padNum != 0:
        len_pad = win_size - padNum
        padding_feature = np.zeros(shape=(513, len_pad))
        x_spec = np.concatenate((x_spec, padding_feature), axis=1)
        num_frames = num_frames + len_pad

    for j in range(0, num_frames, win_size):
        x_test_tmp = x_spec[:, range(j, j + win_size)].T
        x_test.append(x_test_tmp)
    x_test = np.array(x_test)

    # for normalization

    x_train_mean = np.load(
        "model/melodyExtraction_JDC/x_data_mean_total_31.npy"
    )  # TODO: correct paths
    x_train_std = np.load("model/melodyExtraction_JDC/x_data_std_total_31.npy")
    x_test = (x_test - x_train_mean) / (x_train_std + 0.0001)
    x_test = x_test[:, :, :, np.newaxis]

    return x_test, x_spec


class BabyCryDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.data = pd.read_csv(os.path.join(self.config.data.path, f"{split}.csv"))
        self._extraction = False
        if self.config.model.name == "jdc":
            self.spec_extraction = True
        self.spec = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.spec_extraction:
            path = self.data.iloc[idx]["path"]
            y = self.data.iloc[idx]["label"]

            x, _ = spec_extraction(path, self.config.model.jdc.win_size)
        else:
            x = librosa.load(self.data.iloc[idx]["path"], sr=16000)[0]
            y = self.data.iloc[idx]["label"]

            if self.spec:
                # x = librosa.feature.melspectrogram(
                #     x, sr=16000, n_mels=self.input_shape[1], fmax=8000
                # )
                pass  # TODO: implement

        return x, y
