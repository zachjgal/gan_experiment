import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class _MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_data_loader(**kwargs):
    # Load the data
    data = loadmat('emnist-letters.mat')
    data = data['dataset']
    # Access the training data and labels
    train_data = data['train'][0, 0]['images'][0, 0]
    train_labels = data['train'][0, 0]['labels'][0, 0]
    # Reshape the data to 28x28 and rotate it
    train_data = train_data.reshape((-1, 28, 28), order='F')
    # Flip and rotate image because apparently the things are flipped
    train_data = np.rot90(train_data, axes=(1, 2))
    # Subtract 1 from labels since Python uses 0-based indexing
    train_labels = train_labels - 1
    # Normalize train data to 0-1 values
    normalized_train_data = [img / 255.0 for img in train_data]
    # Convert the list back to numpy array and then to PyTorch tensor of float32 type
    # which happens automatically
    normalized_train_data = np.array(normalized_train_data, dtype=np.float32)
    dataset = _MyDataset(normalized_train_data, train_labels)
    data_loader = DataLoader(dataset, **kwargs)
    return data_loader