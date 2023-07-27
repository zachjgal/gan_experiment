from typing import Tuple

import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from generator import Generator
from discriminator import Discriminator
from constants import nz, num_classes


class _MyDataset(Dataset):
    """Dataset adapter so that pytorch can understand how to process our data"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_data_loader(**kwargs):
    """Loads the dataset and performs pre-processing"""
    data = loadmat('emnist-balanced.mat')
    data = data['dataset']
    train_data = data['train'][0, 0]['images'][0, 0]
    train_labels = data['train'][0, 0]['labels'][0, 0]
    train_labels = train_labels - 10
    desired_labels = set(range(26))
    filtered_train_data = []
    filtered_train_labels = []
    for i in range(len(train_data)):
        c = train_labels[i][0]
        if c not in desired_labels:
            continue
        else:
            filtered_train_labels.append(train_labels[i])
            filtered_train_data.append(train_data[i])
    train_data = np.array(filtered_train_data)
    train_labels = np.array(filtered_train_labels)
    train_data = train_data.reshape((-1, 28, 28), order='F')
    normalized_train_data = [img / 255.0 for img in train_data]
    normalized_train_data = np.array(normalized_train_data, dtype=np.float32)
    dataset = _MyDataset(normalized_train_data, train_labels)
    data_loader = DataLoader(dataset, **kwargs)
    return data_loader


TRAINED_G_PATH = Path("./output/results_2023-07-25_15-45-51/2023-07-25_20-17-07_epoch_36/G_2023-07-25_20-17-07.pth")
TRAINED_D_PATH = Path("./output/results_2023-07-25_15-45-51/2023-07-25_20-17-07_epoch_36/D_2023-07-25_20-17-07.pth")

def get_trained_models() -> Tuple[Generator, Discriminator]:
    """Load a trained generator and discriminator"""
    device = torch.device("cpu")
    G = Generator(nz, num_classes)
    D = Discriminator(num_classes)

    G.load_state_dict(torch.load(TRAINED_G_PATH))
    D.load_state_dict(torch.load(TRAINED_D_PATH))
    return G, D

