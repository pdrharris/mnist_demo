"""
This module contains a custom PyTorch dataset class for loading the MNIST dataset.

Attributes:
    MnistDataset: A custom PyTorch dataset class for loading the MNIST dataset.
"""
from typing import Dict
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class MnistDataset(Dataset):
    """
    A custom PyTorch dataset class for loading the MNIST dataset.

    The dataset expects to find a directories under data_root called 'train' and 'test'.
    These will contain the train or the test data.

    The dataset also expects to find a file called 'train.csv' or 'test.csv', as appropriate,
    in the specified data directory.
        The 'train.csv' or 'test.csv' file should contain a column called 'png' that contains
        the name of each image file in the dataset.
        The 'train.csv' or 'test.csv' should also contain a column called 'label'
        that contains the correct label for each image in the dataset.

    See mnist_dataset.ipynb for more details on this class and the csv files.

    Args:
        data_root (Path or str): The root directory containing the dataset.
        mode (str): The mode of the dataset (either 'train' or 'test').

    Attributes:
        data_path (Path): The path to the directory for this dataset (the train or the test dataset).
        image_info (pd.DataFrame): A Pandas DataFrame containing information about the images in the dataset.

    Methods:
        __len__: Returns the number of images in the dataset.
        __getitem__: Returns the image and label data for the specified index.

    Examples:
        train_dataset = MnistDataset('../data, 'train')
        len(train_dataset)
            [returns the length of the dataset]
        train_dataset[0]
            [returns the image and label data for the first image in the dataset]
    """

    def __init__(self, data_root: Path, mode: str) -> None:
        """
        Initializes a new instance of the MnistDataset class.

        Args:
            data_root (Path or str): The root directory containing the dataset.
            mode (Path): The mode of the dataset (either 'train' or 'test').
        """
        data_root = Path(data_root) # If data_root is a string, convert it to a Path object
        self.data_path = data_root / mode
        self.image_info = pd.read_csv(self.data_path / f'{mode}.csv')

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_info)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the image and label data for the specified index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the image and label data.
        """
        row = self.image_info.iloc[idx]
        image = read_image(str(self.data_path / row.png)) / 255.0
        label = torch.tensor(row.label, dtype=torch.long)
        return {'x': image, 'Y': label}
