import os
import math
import torch
import random
import torchaudio
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf

from torch.utils.data import Dataset, DataLoader

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {"n_fft": 2048, "win_length": 1200, "hop_length": 300}
MEL_PARAMS = {"n_mels": 80, "n_fft": 2048, "win_length": 1200, "hop_length": 300}


class BabyCryDatasetMelspec(Dataset):
    def __init__(self, config, split, sr=24000, val_test=False, verbose=True):
        self.config = config
        self.split = split
        self.root_dir = config.data.root_dir

        path = os.path.join(self.root_dir, self.split + ".csv")
        self.df = pd.read_csv(path)

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = self.config.data.augmentation and (not val_test)
        self.max_mel_length = 192  # 513  # 192
        self.mean, self.std = -4, 4

        self.verbose = verbose

        self.zero_value = -10
        self.bad_F0 = 5

    def __len__(self):
        return len(self.df.index)

    def path_to_mel_and_label(self, path):
        wave_tensor = self._load_tensor(path)

        # use pyworld to get F0
        output_file = path + "_f0.npy"
        # check if the file exists
        if os.path.isfile(output_file):  # if exists, load it directly
            f0 = np.load(output_file)
        else:  # if not exist, create F0 file
            if self.verbose:
                print("Computing F0 for " + path + "...")
            x = wave_tensor.numpy().astype("double")
            frame_period = MEL_PARAMS["hop_length"] * 1000 / self.sr
            _f0, t = pw.harvest(x, self.sr, frame_period=frame_period)
            if sum(_f0 != 0) < self.bad_F0:  # this happens when the algorithm fails
                _f0, t = pw.dio(
                    x, self.sr, frame_period=frame_period
                )  # if harvest fails, try dio
            f0 = pw.stonemask(x, _f0, t, self.sr)
            # save the f0 info for later use
            np.save(output_file, f0)

        f0 = torch.from_numpy(f0).float()

        if self.data_augmentation:
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)

        f0_zero = f0 == 0

        #######################################
        # You may want your own silence labels here
        # The more accurate the label, the better the results
        is_silence = torch.zeros(f0.shape)
        is_silence[f0_zero] = 1
        #######################################

        mel_concat = torch.zeros(
            (
                math.ceil(mel_length / self.max_mel_length),
                mel_tensor.size(0),
                self.max_mel_length,
            )
        )
        f0_concat = torch.zeros(
            (math.ceil(mel_length / self.max_mel_length), self.max_mel_length)
        )
        is_silence_concat = torch.zeros(
            (
                math.ceil(mel_length / self.max_mel_length),
                self.max_mel_length,
            )
        )

        for i in range(math.ceil(mel_length / self.max_mel_length)):
            start_point = i * self.max_mel_length
            end_point = (i + 1) * self.max_mel_length
            if end_point > mel_length:
                end_point = mel_length

            mel_concat[i, :, : end_point - start_point] = mel_tensor[
                :, start_point:end_point
            ]
            f0_concat[i, : end_point - start_point] = f0[start_point:end_point]
            is_silence_concat[i, : end_point - start_point] = is_silence[
                start_point:end_point
            ]

        # if mel_length > self.max_mel_length:
        #     random_start = np.random.randint(0, mel_length - self.max_mel_length)
        #     mel_tensor = mel_tensor[
        #         :, random_start : random_start + self.max_mel_length
        #     ]
        #     f0 = f0[random_start : random_start + self.max_mel_length]
        #     is_silence = is_silence[random_start : random_start + self.max_mel_length]

        if torch.any(torch.isnan(f0)):  # failed
            f0[torch.isnan(f0)] = self.zero_value  # replace nan value with 0

        return mel_concat, f0_concat, is_silence_concat  # mel_tensor, f0, is_silence

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        mel_tensor, f0, is_silence = self.path_to_mel_and_label(item["path"])
        label = 0 if item["label"] == "G" else 1
        return mel_tensor, f0, is_silence, label

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def get_split(self):
        neg = len(self.df[self.df["label"] == "G"])
        pos = len(self.df[self.df["label"] == "J"])

        # return [0.7, 0.3]
        return [pos / (pos + neg), neg / (pos + neg)]  # [percent of J, percent of G]


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        nmels = max([x[0].size(0) for x in batch])
        mels = torch.zeros(
            (batch_size, nmels, MEL_PARAMS["n_mels"], self.max_mel_length)
        ).float()
        f0s = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        is_silences = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).float()

        for bid, (mel, f0, is_silence, label) in enumerate(batch):
            mel_size = mel.size(0)
            mels[bid, :mel_size, :, :] = mel
            f0s[bid, :mel_size, :] = f0
            is_silences[bid, :mel_size, :] = is_silence

            labels[bid] = label

        return mels, f0s, is_silences, labels


def get_dataloaders(config, device, collate_config={}):
    if config.data.data_type == "mel-spectogram":
        train_dataset = BabyCryDatasetMelspec(config, "train", verbose=False)
        val_dataset = BabyCryDatasetMelspec(config, "val", verbose=False)
        test_dataset = BabyCryDatasetMelspec(config, "test", verbose=False)

        collate_fn = Collater(**collate_config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != "cpu"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != "cpu"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != "cpu"),
        )
    else:
        raise NotImplementedError(f"Data type {config.data.data_type} not implemented.")

    split = train_dataset.get_split()

    return train_loader, val_loader, test_loader, split
