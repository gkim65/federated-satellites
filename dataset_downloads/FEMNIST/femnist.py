# Modified from code by Alexandru-Andrei Iacob (aai30@cam.ac.uk) and Lorenzo Sani (ls985@cam.ac.uk, lollonasi97@gmail.com)

import csv
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

import subprocess
import gdown

from PIL import Image
from PIL.Image import Image as ImageType
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

def download_femnist():
    """
    Download and extract the femnist dataset if not already present
    """
    #  Download compressed dataset
    print(Path('./femnist.tar.gz').exists())
    if not Path('./femnist.tar.gz').exists():
        file_id = '1MeYgT9qQp973EP7kXnM9KTDGo1-qeuC6'
        gdown.download(
            f'https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}',
            './femnist.tar.gz',
        )

    # Decompress dataset
    if not Path('./femnist').exists():
        print('Extracting FEMNIST data...')
        subprocess.run('tar -xzf ./femnist.tar.gz'.split(), check=True, capture_output=True)
        print('FEMNIST dataset extracted in ./femnist/data')


class FemnistDataset(Dataset):
    """Class to load the FEMNIST dataset."""

    def __init__(
        self,
        client: int,
        split: str = "train",
        transform: Optional[Callable[[ImageType], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
    ) -> None:
        """Initialize the FEMNIST dataset.

        Args:
            client (int): client to get file mapping for
            split (str): split of the dataset to load, train or test.
            transform (Optional[Callable[[ImageType], Any]], optional):
                    transform function to be applied to the ImageType object.
            target_transform (Optional[Callable[[int], Any]], optional):
                    transform function to be applied to the label.
        """
        # TODO: find ways to make this not changed for my mac
        self.data_dir = Path('/datasets/femnist/data') #Path('FEMNIST_tests/femnist/data')
        self.client = client
        self.mapping_dir = Path('/datasets/femnist/client_data_mappings/fed_natural') #Path('FEMNIST_tests/femnist/client_data_mappings/fed_natural')
        self.split = split


        self.data: Sequence[tuple[str, int]] = self._load_dataset()
        self.transform: Callable[[ImageType], Any] | None = transform
        self.target_transform: Callable[[int], Any] | None = target_transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get a sample.

        Args:
            index (_type_): index of the sample.

        Returns
        -------
            Tuple[Any, Any]: couple (sample, label).
        """
        sample_path, label = self.data[index]

        # Convert to the full path
        try:
            full_sample_path: Path = self.data_dir / self.split / sample_path
            if not full_sample_path.exists():
                raise ValueError(f"Required files do not exist, path: {full_sample_path}")
        except:
            self.data_dir = Path('../datasets/femnist/data')
            full_sample_path: Path =  self.data_dir / self.split / sample_path
            if not full_sample_path.exists():
                raise ValueError(f"Required files do not exist, path: {full_sample_path}")
        img: ImageType = Image.open(full_sample_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        """Get the length of the dataset as number of samples.

        Returns
        -------
            int: the length of the dataset.
        """
        return len(self.data)

    def _load_dataset(self) -> Sequence[tuple[str, int]]:
        """Load the paths and labels of the partition.

        Preprocess the dataset for faster future loading
        If opened for the first time

        Raises
        ------
            ValueError: raised if the mapping file doesn't exist

        Returns
        -------
            Sequence[Tuple[str, int]]:
                partition asked as a sequence of couples (path_to_file, label)
        """
        try:
            preprocessed_path: Path = (self.mapping_dir / self.client / self.split).with_suffix(".pt")
            if not preprocessed_path.exists():
                raise ValueError(f"Required files do not exist, path: {preprocessed_path}")
        except:
            self.mapping_dir = Path('../datasets/femnist/client_data_mappings/fed_natural')
            preprocessed_path: Path = (self.mapping_dir / self.client / self.split).with_suffix(".pt")
            if not preprocessed_path.exists():
                raise ValueError(f"Required files do not exist, path: {preprocessed_path}")

        if preprocessed_path.exists():
            return torch.load(preprocessed_path)
        else:
            try:
                csv_path = (self.mapping_dir / self.client / self.split).with_suffix(".csv")
                if not csv_path.exists():
                    raise ValueError(f"Required files do not exist, path: {csv_path}")
            except:
                self.mapping_dir = Path('../datasets/femnist/client_data_mappings/fed_natural')
                csv_path = (self.mapping_dir / self.client / self.split).with_suffix(".csv")
                if not csv_path.exists():
                    raise ValueError(f"Required files do not exist, path: {csv_path}")
            

            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Ignore header
                next(csv_reader)

                # Extract the samples and the labels
                partition: Sequence[tuple[str, int]] = [
                    (sample_path, int(label_id))
                    for _, sample_path, _, label_id in csv_reader
                ]

                # Save for future loading
                torch.save(partition, preprocessed_path)
                return partition

class FemnistNet(nn.Module):
    """Simple CNN model for FEMNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_FEMNIST(cid):
  """Load FEMNIST (training and test set)."""
  trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_loader = DataLoader(
        FemnistDataset(client=cid, split='train', transform=ToTensor()),
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

  val_loader = DataLoader(
        FemnistDataset(client=cid, split='test', transform=ToTensor()),
        batch_size=32,
        shuffle=False,
        drop_last=False
  )

  return train_loader, val_loader 
  

if __name__ == "__main__":
    download_femnist()
