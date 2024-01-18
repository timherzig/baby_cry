import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.data.data import get_dataloaders
from src.models.jdc.model import JDCNet


def pitch_distribution_for_dataset(config, set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if set == "train":
        loader, _, _, split = get_dataloaders(config, device)
    elif set == "val":
        _, loader, _, split = get_dataloaders(config, device)
    elif set == "test":
        _, _, loader, split = get_dataloaders(config, device)

    Net = JDCNet(num_class=config.model.num_classes, seq_len=config.model.seq_len)
    params = torch.load(config.model.pretrained, map_location="cpu")["net"]
    Net.load_state_dict(params)

    for param in Net.parameters():
        param.requires_grad = False

    Net = Net.to(device)

    all_G_pitch = []
    all_J_pitch = []
    longest_seq = 0

    for batch in tqdm(loader):
        samples, _, _, labels = batch
        samples = samples.to(device)
        labels = labels.to(device)

        output = torch.zeros(
            (samples.shape[0], samples.shape[1], config.model.seq_len)
        ).to(samples.device)
        for i in range(samples.shape[1]):
            pitch, _, _ = Net(samples[:, i, :, :].unsqueeze(1))
            output[:, i, :] = pitch

        output /= 300  # normalize pitch (found 300 to be roughly the max pitch value)

        l = labels[:, 0].bool()

        output = output.reshape(
            [output.shape[0], config.model.seq_len * output.shape[1]]
        )

        for i in range(output.shape[0]):
            if l[i]:
                all_G_pitch.append(output[i].cpu().numpy())
            else:
                all_J_pitch.append(output[i].cpu().numpy())

            if output.shape[1] > longest_seq:
                longest_seq = output.shape[1]

    for i in range(len(all_G_pitch)):
        all_G_pitch[i] = np.pad(
            all_G_pitch[i], (0, longest_seq - all_G_pitch[i].shape[0]), "constant"
        )
    for i in range(len(all_J_pitch)):
        all_J_pitch[i] = np.pad(
            all_J_pitch[i], (0, longest_seq - all_J_pitch[i].shape[0]), "constant"
        )

    all_G_pitch = np.array(all_G_pitch)
    all_J_pitch = np.array(all_J_pitch)

    print(all_G_pitch.shape)
    print(all_J_pitch.shape)

    mean_G_pitch = all_G_pitch.mean(axis=0)
    mean_J_pitch = all_J_pitch.mean(axis=0)

    std_dev_G_pitch = all_G_pitch.std(axis=0)
    std_dev_J_pitch = all_J_pitch.std(axis=0)

    plt.plot(mean_G_pitch, label="mean G")
    plt.fill_between(
        np.arange(longest_seq),
        (mean_G_pitch - std_dev_G_pitch),
        (mean_G_pitch + std_dev_G_pitch),
        alpha=0.3,
    )

    plt.plot(mean_J_pitch, label="mean J")
    plt.fill_between(
        np.arange(longest_seq),
        (mean_J_pitch - std_dev_J_pitch),
        (mean_J_pitch + std_dev_J_pitch),
        alpha=0.3,
    )

    plt.legend()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(
        f"figs/pitch_distribution_{set}_{config.data.root_dir.split('/')[-1]}.png"
    )
    plt.clf()


def pitch_distribution(config):
    pitch_distribution_for_dataset(config, "train")
    pitch_distribution_for_dataset(config, "val")
    pitch_distribution_for_dataset(config, "test")
