import os
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def load_data_CIFAR10(cid):
    """Load CIFAR-10 (training and test set)."""
    try:
        data_dir = '/datasets/'
        if not os.path.exists(data_dir):
            raise ValueError(f"Required files do not exist, path: {data_dir}")
    except:
        data_dir = '../datasets/'
        print("found data", data_dir)
        if not os.path.exists(data_dir):
            raise ValueError(f"Required files do not exist, path: {data_dir}")
    
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = CIFAR10(data_dir, download=True, transform=trf)

    cid = int(cid)
    
    datasets_noniid: List[Subset] = []
    
    distribute_noniid(100, 0.5, 41, dataset, datasets_noniid)

    # Randomly split the dataset into 80% train / 20% test 
    # by subsetting the transformed train and test datasets
    train_size = 0.8
    indices = list(range(int(len(datasets_noniid[cid]))))
    split = int(train_size * len(datasets_noniid[cid]))
    np.random.shuffle(indices)

    train_data_cid = Subset(datasets_noniid[cid], indices=indices[:split])
    test_data_cid = Subset(datasets_noniid[cid], indices=indices[split:])

    print("Train/test sizes: {}/{}".format(len(train_data_cid), len(test_data_cid)))

    batch_size = 32
    num_workers = 2
    train_loader = DataLoader(
        train_data_cid, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(
        test_data_cid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    # print("# OF SAMPLES")
    # print(len(train_loader))
    # print(len(test_loader))
    
    return train_loader, test_loader

def distribute_noniid(num_clients, beta, seed, trainset, datasets):

    """Distribute dataset in non-iid manner."""
    print(trainset)
    labels = np.array([label for _, label in trainset])
    min_size = 0
    np.random.seed(seed)

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(np.max(labels) + 1):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            # Balance
            proportions = np.array(
                [
                    p * (len(idx_j) < labels.shape[0] / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        # net_dataidx_map[j] = np.array(idx_batch[j])
        datasets.append(Subset(trainset, np.array(idx_batch[j])))


class CIFAR10_Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
