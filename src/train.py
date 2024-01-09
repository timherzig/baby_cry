import os
import time
import torch
import shutil
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, accuracy_score
from src.data.data import get_dataloaders
from src.models.model import get_model


def train(config, config_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists("./trained_models/"):
        os.makedirs("./trained_models/")

    train_loader, val_loader, test_loader, split = get_dataloaders(config, device)
    print(f"Train class weights: {split}")

    Net = get_model(config).to(device)

    num_total_learnable_params = sum(
        p.numel() for p in Net.parameters() if p.requires_grad
    )
    print(f"Number of learnable parameters: {num_total_learnable_params}")

    optimizer = optim.Adam(Net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_type = "binary_cross_entropy"  # {binary_cross_entropy}

    print(
        f"Training data: {config.data.root_dir} ver. {config.data.version}, data type {config.data.data_type}."
    )

    num_epochs = config.num_epochs
    train_loss_per_epoch = torch.zeros(num_epochs)
    val_loss_per_epoch = torch.zeros(num_epochs)
    val_f1_per_epoch = torch.zeros(num_epochs)
    val_acc_per_epoch = torch.zeros(num_epochs)
    best_val_loss = float("inf")

    time_name = time.ctime()
    time_name = time_name.replace(" ", "_")
    time_name = time_name.replace(":", "_")

    path = f"./trained_models/{config.data.root_dir.split('/')[-1].split('.')[0]}_{config.data.version}/{config.model.architecture}/{config_name}"

    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

    if config_name == "default":
        path = "./trained_models/debug"
        shutil.rmtree(path, ignore_errors=True)

    log_path = f"{path}/log"
    save_path = f"{path}/save"

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    OmegaConf.save(config, f"{path}/config.yaml")

    f = open(f"{log_path}/{time_name}.csv", "w+")

    print("Training started...")
    best_model_save_path = ""

    for epoch in range(num_epochs):
        Net.train()
        t = time.time()
        total_train_loss = 0
        train_counter = 0

        for batch in tqdm(train_loader):
            train_counter += 1
            samples, _, _, labels = batch
            samples = samples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if loss_type == "binary_cross_entropy":
                output = Net(samples)
                loss = F.binary_cross_entropy(output, labels)  # TODO: add class weights
            else:
                raise NotImplementedError(f"Loss type {loss_type} not implemented.")

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss_per_epoch[epoch] = total_train_loss / train_counter

        total_val_loss = 0
        total_f1_counter = 0
        total_acc_counter = 0
        val_counter = 0
        for batch in tqdm(val_loader):
            val_counter += 1
            samples, _, _, labels = batch
            samples = samples.to(device)
            labels = labels.to(device)

            if loss_type == "binary_cross_entropy":
                output = Net(samples)
                loss = F.binary_cross_entropy(output, labels)
                print(f"max value output {torch.max(output)}")
                o = (output > 0.5).float()
                o = o.clone().cpu().detach().numpy()
                l = labels.clone().cpu().detach().numpy()
                f1 = f1_score(l, o)
                acc = accuracy_score(l, o)
            else:
                raise NotImplementedError(f"Loss type {loss_type} not implemented.")

            total_val_loss += loss.item()
            total_f1_counter += f1
            total_acc_counter += acc

        val_loss_per_epoch[epoch] = total_val_loss / val_counter
        val_f1_per_epoch[epoch] = total_f1_counter / val_counter
        val_acc_per_epoch[epoch] = total_acc_counter / val_counter
        if val_loss_per_epoch[epoch] < best_val_loss:
            best_val_loss = val_loss_per_epoch[epoch]
            net_str = f"{config.model.architecture}_epoch_{epoch}_train-loss_{train_loss_per_epoch[epoch]:.4f}_val-loss_{val_loss_per_epoch[epoch]:.4f}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": Net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss_per_epoch,
                },
                f"{save_path}/{net_str}",
            )
            best_model_save_path = f"{save_path}/{net_str}"

        elapsed = time.time() - t

        print_str = f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss_per_epoch[epoch]:.4f} | Val loss: {val_loss_per_epoch[epoch]:.4f} | Val f1: {val_f1_per_epoch[epoch]:.4f} | Val acc: {val_acc_per_epoch[epoch]:.4f} | Time: {elapsed:.2f} s"
        print(print_str)

        df = pd.DataFrame([print_str])
        df.to_csv(
            f"{log_path}/{time_name}.csv",
            sep=" ",
            mode="a",
            header=False,
            index=False,
        )

        scheduler.step()

    plt.plot(train_loss_per_epoch, label="train")
    plt.plot(val_loss_per_epoch, label="val")
    plt.legend()
    plt.savefig(f"{log_path}/train_plot.png")

    f.close()

    print("Training finished.")
    print(f"Best model saved at: {best_model_save_path}")
