
import os
import gdown

# from PIL import Image
# from PIL.Image import Image as ImageType
# from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import zipfile
import requests

def downloadEUROSAT():
    """
    Download and extract the EuroSAT dataset if not already present
    """
    #  Download compressed dataset
    url = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip'
    save_path = "../../datasets/EuroSAT.zip"
    if not os.path.exists(save_path):
        print("Downloading Zip", save_path)
        r = requests.get(url, stream=True)
        chunk_size = 128
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    # Decompress dataset

    folder_name = "../../datasets/EuroSAT"
    if not os.path.exists(folder_name):
        print('Extracting EuroSAT data...')
        with zipfile.ZipFile("../../datasets/EuroSAT.zip", 'r') as zip_ref:
            zip_ref.extractall("../../datasets/EuroSAT")
        print('EuroSAT dataset extracted in ../../datasets/EuroSAT')


    if os.path.exists(save_path):
        os.remove(save_path)


class EuroSAT(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # Apply image transformations 
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        # Get class label
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)


class EuroSATNet(nn.Module):
    """Simple CNN model for EuroSATNet."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_EUROSAT(cid):
    """Load EUROSAT.py (training and test set)."""
    try:
        data_dir = '/datasets/EuroSAT/2750/'
        if not os.path.exists(data_dir):
            raise ValueError(f"Required files do not exist, path: {data_dir}")
    except:
        data_dir = '../datasets/EuroSAT/2750/'
        print("found data", data_dir)
        if not os.path.exists(data_dir):
            raise ValueError(f"Required files do not exist, path: {data_dir}")
    
    dataset = datasets.ImageFolder(data_dir)
    class_names = dataset.classes
    print("Class names: {}".format(class_names))
    print("Total number of classes: {}".format(len(class_names)))

    # Appropriate transforsm to make more data
    input_size = 224
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    train_data = EuroSAT(dataset, train_transform)
    # test_data = EuroSAT(dataset, test_transform)

    # Randomly split the dataset into 80% train / 20% test 
    # by subsetting the transformed train and test datasets
    # train_size = 0.8
    # indices = list(range(int(len(dataset))))
    # split = int(train_size * len(dataset))
    # np.random.shuffle(indices)

    # train_data = Subset(train_data, indices=indices[:split])
    # test_data = Subset(test_data, indices=indices[split:])
    # print("Train/test sizes: {}/{}".format(len(train_data), len(test_data)))

    # num_workers = 2
    # batch_size = 32

    cid = int(cid)
    
    datasets_noniid: List[Subset] = []
    
    distribute_noniid(100, 0.5, 41, train_data, datasets_noniid)

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

if __name__ == "__main__":
    downloadEUROSAT()
