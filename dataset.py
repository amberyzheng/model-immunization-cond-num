import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import random

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Subset
from sklearn.preprocessing import LabelEncoder
import scipy.io as sio
from timm.data import create_dataset, create_loader

from pdb import set_trace as stx


import itertools

def combined_loader(loader1, loader2):
    """
    Combines two loaders. The shorter one restarts once exhausted.
    """
    iter1 = iter(loader1)
    iter2 = iter(loader2)
    while True:
        try:
            batch1 = next(iter1)
        except StopIteration:
            break  # Exit when loader1 is exhausted

        try:
            batch2 = next(iter2)
        except StopIteration:
            iter2 = iter(loader2)
            batch2 = next(iter2)
        
        yield batch1, batch2




class CombinedLoader:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.len1 = len(loader1)
        self.len2 = len(loader2)

    def __iter__(self):
        return combined_loader(self.loader1, self.loader2)

    def __len__(self):
        return self.len2
    

class BinaryDataset(Dataset):
    """Custom Dataset to filter and relabel for binary classification."""
    def __init__(self, dataset, target_digits, max_samples_per_digit=None):
        """
        Args:
            dataset (Dataset): The original MNIST dataset.
            target_digits (list): List of two target digits to include (e.g., [3, 7]).
            max_samples_per_digit (int, optional): Maximum number of samples per digit. Defaults to the smaller digit count.
        """
        if len(target_digits) != 2:
            raise ValueError("target_digits must contain exactly two digits.")
        self.dataset = dataset
        self.target_digits = target_digits
        self.label_map = {target_digits[0]: 0, target_digits[1]: 1}  # Map target digits to 0 and 1

        # Group indices by label
        digit_indices = defaultdict(list)
        for i, (_, label) in enumerate(dataset):
            if label in target_digits:
                digit_indices[label].append(i)

        # Balance the dataset by limiting the number of samples per digit
        digit1, digit2 = target_digits
        num_samples = max_samples_per_digit or min(len(digit_indices[digit1]), len(digit_indices[digit2]))
        self.indices_digit1 = random.sample(digit_indices[digit1], num_samples)
        self.indices_digit2 = random.sample(digit_indices[digit2], num_samples)

        # Combine indices and shuffle
        self.indices = list(zip(self.indices_digit1, self.indices_digit2))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get paired indices for the two digits
        idx1, idx2 = self.indices[idx]
        image1, label1 = self.dataset[idx1]
        image2, label2 = self.dataset[idx2]

        # Map labels to 0 and 1
        label1 = self.label_map[label1]
        label2 = self.label_map[label2]

        return (image1, label1), (image2, label2)





class TabularDataset(Dataset):
    def __init__(self, d1_path, d2_path, features_columns, d1_label_column, d2_label_column):
        """
        Args:
            d1_path (str): Path to the D1 CSV file.
            d2_path (str): Path to the D2 CSV file.
            features_columns (list): List of feature column names.
            d1_label_column (str): Name of the target column in D1 (y1).
            d2_label_column (str): Name of the target column in D2 (y2).
        """
        # Load datasets
        self.d1_data = pd.read_csv(d1_path)
        self.d2_data = pd.read_csv(d2_path)

        # Preprocess the features and labels
        self.features_columns = features_columns
        self.d1_label_column = d1_label_column
        self.d2_label_column = d2_label_column

        self.d1_data, self.d1_mean, self.d1_std, self.d1_label_mean, self.d1_label_std = self._preprocess(self.d1_data, d1_label_column)
        self.d2_data, self.d2_mean, self.d2_std, self.d2_label_mean, self.d2_label_std = self._preprocess(self.d2_data, d2_label_column)

        # Convert to PyTorch tensors
        self.d1_features = torch.tensor(self.d1_data[features_columns].values, dtype=torch.float32)
        self.d1_labels = torch.tensor(self.d1_data[d1_label_column].values, dtype=torch.float32)

        self.d2_features = torch.tensor(self.d2_data[features_columns].values, dtype=torch.float32)
        self.d2_labels = torch.tensor(self.d2_data[d2_label_column].values, dtype=torch.float32)

    def _preprocess(self, data, label_column):
        """Handle missing values, encode categorical columns, and normalize numerical features and labels."""
        # Fill missing values
        data = data.fillna(0)

        # Encode categorical columns
        for col in data.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        # Normalize numerical features
        numerical_cols = data[self.features_columns].select_dtypes(include=["number"])
        mean = numerical_cols.mean()
        std = numerical_cols.std().replace(0, 1)  # Replace 0 std with 1 to avoid division by 0
        data[self.features_columns] = (data[self.features_columns] - mean) / std

        # Normalize labels
        label_mean = data[label_column].mean()
        label_std = data[label_column].std() or 1  # Avoid division by zero
        data[label_column] = (data[label_column] - label_mean) / label_std

        return data, mean, std, label_mean, label_std

    def __len__(self):
        # Use the smaller dataset size for pairing
        return min(len(self.d1_features), len(self.d2_features))

    def __getitem__(self, idx):
        # Pair the features and labels from D1 and D2
        feature1 = self.d1_features[idx]
        label1 = self.d1_labels[idx]

        feature2 = self.d2_features[idx]
        label2 = self.d2_labels[idx]

        return (feature1, label1), (feature2, label2)

    def denormalize_labels(self, y, dataset="d1"):
        """
        Denormalize labels back to original scale.
        Args:
            y (torch.Tensor): Normalized labels.
            dataset (str): Specify which dataset ("d1" or "d2").
        Returns:
            torch.Tensor: Denormalized labels.
        """
        if dataset == "d1":
            return y * self.d1_label_std + self.d1_label_mean
        elif dataset == "d2":
            return y * self.d2_label_std + self.d2_label_mean
        else:
            raise ValueError("Invalid dataset. Use 'd1' or 'd2'.")


from PIL import Image


class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2, max_samples_per_class=None):
        """
        Args:
            dataset1: The first dataset (ImageNet).
            dataset2: The second dataset (Stanford Cars or Country211).
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.max_samples_per_class = max_samples_per_class

    def __len__(self):
        # Use the maximum length of the datasets to ensure all data is used
        if self.max_samples_per_class:
            return max(self.len1, self.len2)
        return min(self.len1, self.len2)

    def __getitem__(self, idx):
        """
        Returns:
            X1, y1 (from dataset1) and X2, y2 (from dataset2).
        """
        # Wrap-around indexing to use all data from both datasets
        X1, y1 = self.dataset1[idx % self.len1]
        X2, y2 = self.dataset2[idx % self.len2]
        return (X1, y1), (X2, y2)



class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_name, data_dir=None, target_classes=None, max_samples_per_class=None, d1_path=None, d2_path=None, d2_name=None):
        """
        Args:
            batch_size (int): Batch size for training and validation.
            dataset_name (str): Name of the dataset ('mnist', 'imagenet', or 'tabular').
            data_dir (str, optional): Directory path for datasets (required for ImageNet).
            target_classes (list, optional): List of two target classes for binary classification.
            max_samples_per_class (int, optional): Maximum samples to include for each class.
            d1_path (str, optional): Path to the D1 CSV file.
            d2_path (str, optional): Path to the D2 CSV file.
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.target_classes = target_classes
        self.max_samples_per_class = max_samples_per_class
        self.d1_path = d1_path
        self.d2_path = d2_path
        self.d2_name = d2_name

        # Define normalizations for each dataset
        self.normalizations = {
            "mnist": transforms.Normalize((0.1307,), (0.3081,)),
            "imagenet": transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        }

        # Define transformations
        self.transforms = {
            "mnist": {
                "train": transforms.Compose([transforms.ToTensor(), self.normalizations["mnist"]]),
                "val": transforms.Compose([transforms.ToTensor(), self.normalizations["mnist"]]),
            },
            "imagenet": {
                "train": transforms.Compose([
                    transforms.RandomResizedCrop(
                        size=(224, 224), scale=(0.5, 1), 
                        interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    self.normalizations["imagenet"]
                ]),
                "val": transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        self.normalizations["imagenet"]]),
                    },      
        }

    def prepare_data(self):
        if self.dataset_name == "mnist":
            datasets.MNIST(root="./data", train=True, download=True)
        elif self.dataset_name == "imagenet":
            if self.d2_name == "country211":
                datasets.Country211(root=self.d2_path, split="train", download=True)
            if self.d2_name == "cars":
                # Assumes the Stanford Cars dataset is manually downloaded
                if not os.path.exists(os.path.join(self.d2_path, "cars_train")):
                    raise FileNotFoundError("Stanford Cars dataset is not found. Please download it first.")
        elif self.dataset_name == "tabular":
            if not self.d1_path or not self.d2_path:
                raise ValueError("For tabular datasets, d1_path and d2_path must be specified.")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def setup(self, stage=None):
        if self.dataset_name == "mnist":
            self.train_dataset = datasets.MNIST(root="./data", train=True, transform=self.transforms["mnist"]["train"])
            self.val_dataset = datasets.MNIST(root="./data", train=False, transform=self.transforms["mnist"]["val"])
        elif self.dataset_name == "imagenet":
            self.d1_train = create_dataset("imagenet", root=self.d1_path, split="train")
            self.d1_val = create_dataset("imagenet", root=self.d1_path, split="val")
            self.d2_train = create_dataset("image_folder", root=self.d2_path, split="train")
            self.d2_val = create_dataset("image_folder", root=self.d2_path, split="val")

            # Pair the datasets
            self.train_dataset = PairedDataset(self.d1_train, self.d2_train, self.max_samples_per_class)
            self.val_dataset = PairedDataset(self.d1_val, self.d2_val)

        elif self.dataset_name == "tabular":
            d1_data = pd.read_csv(self.d1_path)
            self.features_columns = [col for col in d1_data.columns if col not in ["LotArea", "SalePrice"]]
            full_dataset = TabularDataset(
                self.d1_path, self.d2_path, self.features_columns, "LotArea", "SalePrice"
            )
            train_indices = list(range(200))
            test_indices = list(range(200, 300))
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, test_indices)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        if self.target_classes:
            train_dataset = BinaryDataset(self.train_dataset, self.target_classes, max_samples_per_digit=self.max_samples_per_class)
            val_dataset = BinaryDataset(self.val_dataset, self.target_classes, max_samples_per_digit=self.max_samples_per_class)

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

    def train_dataloader(self):
        if self.dataset_name == "imagenet":
            loader1 = create_loader(self.d1_train, 
                            batch_size=self.batch_size, 
                            input_size=(3, 224, 224), 
                            is_training=True, 
                            mean=self.normalizations["imagenet"].mean, 
                            std=self.normalizations["imagenet"].std, 
                            interpolation='bicubic', 
                            num_workers=4)
            loader2 = create_loader(self.d2_train, 
                            batch_size=self.batch_size, 
                            input_size=(3, 224, 224), 
                            is_training=True, 
                            mean=self.normalizations["imagenet"].mean, 
                            std=self.normalizations["imagenet"].std, 
                            interpolation='bicubic', 
                            num_workers=4)
        
            self.train_loader = CombinedLoader(loader1, loader2)

        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

        return self.train_loader



    def val_dataloader(self):
        if self.dataset_name == "imagenet":
            return create_loader(self.val_dataset, batch_size=self.batch_size, input_size=(3, 224, 224), is_training=False, mean=self.normalizations["imagenet"].mean, std=self.normalizations["imagenet"].std, num_workers=4)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def get_full_data(self, train=True):

        dataset = self.train_dataset if train else self.val_dataset

        X1, y1, X2, y2 = [], [], [], []

        for (image1, label1), (image2, label2) in dataset:
            X1.append(image1)
            y1.append(label1)
            X2.append(image2)
            y2.append(label2)

        X1 = torch.stack(X1)
        y1 = torch.tensor(y1)
        X2 = torch.stack(X2)
        y2 = torch.tensor(y2)

        return (X1, y1), (X2, y2)
    
    def get_sampled_data(self, k, train=True):
        """
        Randomly samples k data points from the dataset without loading all data into memory.

        Args:
            k (int): Number of samples to draw.
            train (bool): If True, samples from the training dataloader; otherwise from the validation dataloader.

        Returns:
            X1, y1, X2, y2: Sampled tensors.
        """
        loader = self.train_loader if train else self.val_loader  # Select appropriate loader
        
        sampled_X1, sampled_y1, sampled_X2, sampled_y2 = [], [], [], []
        count = 0

        # Shuffle the dataloader
        loader = iter(loader)  # Create an iterator from the loader

        for (X1, y1), (X2, y2) in loader:
            sampled_X1.append(X1)
            sampled_y1.append(y1)
            sampled_X2.append(X2)
            sampled_y2.append(y2)
            
            count += X1.size(0)
            if count >= k:  # Stop when we collect k samples
                break

        # Concatenate and trim to exactly k samples
        X1 = torch.cat(sampled_X1, dim=0)[:k]
        y1 = torch.cat(sampled_y1, dim=0)[:k]
        X2 = torch.cat(sampled_X2, dim=0)[:k]
        y2 = torch.cat(sampled_y2, dim=0)[:k]

        return (X1, y1), (X2, y2)
            
