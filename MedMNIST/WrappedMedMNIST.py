import os
import numpy as np
from medmnist import INFO
from medmnist import __dict__ as medmnist_dict
from torchvision import transforms
import torch


class WrappedMedMNIST:
    """Wrapper class for MedMNIST datasets with standardized loading and preprocessing."""

    # List of supported MedMNIST datasets (single-label, multi-class)
    _compatible_datasets = [
        'pathmnist', 
        'dermamnist', 
        'octmnist', 
        'bloodmnist', 
        'tissuemnist',
        'organamnist', 
        'organcmnist', 
        'organsmnist'
    ]

    # List of all available MedMNIST datasets (full catalog)
    _all_datasets = list(INFO.keys())

    def __init__(self, name: str, flatten: bool = True, normalize: bool = True,
                 download: bool = True, full_split: bool = True):
        """
        Initialize a WrappedMedMNIST dataset loader (only for single-label, multi-class datasets).

        Args:
            name (str): The name of the MedMNIST dataset.
            flatten (bool): If True, flatten each image to 1D.
            normalize (bool): If True, apply standard normalization to the images.
            download (bool): If True, download the dataset if not already present.
            full_split (bool): If True, load train, val and test splits; otherwise, load only train.

        Raises:
            ValueError: If the dataset is not supported or does not meet the single-label, multi-class criteria.
        """
        WrappedMedMNIST._ensure_root_dir()
        
        if name not in WrappedMedMNIST._compatible_datasets:
            raise ValueError(f"Dataset '{name}' is not supported. Use WrappedMedMNIST.available_datasets() to check available datasets.")
        
        self.name = name
        self.flatten = flatten

        info = INFO[name]
        self.n_classes = len(info['label'])
        self.task = info['task']
        #dataset_class = info['python_class']
        dataset_class = medmnist_dict[info['python_class']]

        if self.task != 'multi-class' or self.n_classes <= 2:
            raise ValueError(f"Dataset '{name}' must be multi-class with more than two classes.")

        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize(mean=[.5], std=[.5]))
        transform = transforms.Compose(transform_list)

        splits = ['train', 'val', 'test'] if full_split else ['train']

        self._datasets = [
            dataset_class(
                root='./MedMNIST/medmnist_data',
                split=split,
                transform=transform,
                download=download
            )
            for split in splits
        ]

        self.data = self._prepare_data()

    def _prepare_data(self) -> dict:
        """
        Load and preprocess data from all splits.

        Returns:
            dict: A dictionary with keys 'data' and 'target' containing image arrays and labels.
        """
        WrappedMedMNIST._ensure_root_dir()
        images, labels = [], []
        for dataset in self._datasets:
            for img_tensor, label in dataset:
                img_np = img_tensor.numpy()
                if self.flatten:
                    img_np = img_np.flatten()
                else:
                    img_np = np.transpose(img_np, (1, 2, 0))  # Keep channel last format
                images.append(img_np)
                # Handle single-label and one-hot label cases
                if isinstance(label, (int, np.integer)):
                    labels.append(int(label))
                else:
                    labels.append(int(label[0]))
        return {
            'data': np.array(images),
            'target': np.array(labels)
        }

    @staticmethod
    def available_datasets() -> list:
        """
        Return the list of supported MedMNIST datasets (single-label, multi-class).

        Returns:
            list: List of compatible dataset names.
        """
        return WrappedMedMNIST._compatible_datasets

    @staticmethod
    def all_datasets() -> list:
        """
        Return the full list of datasets available in MedMNIST.

        Returns:
            list: List of all dataset names supported by MedMNIST.
        """
        WrappedMedMNIST._ensure_root_dir()
        return WrappedMedMNIST._all_datasets

    @staticmethod
    def load_any(name: str, flatten: bool = True, normalize: bool = True,
                 download: bool = True, full_split: bool = True) -> dict:
        """
        Load any MedMNIST dataset by name, regardless of task type.

        Args:
            name (str): Dataset name.
            flatten (bool): If True, flatten each image to 1D.
            normalize (bool): If True, apply standard normalization.
            download (bool): If True, download if not already present.
            full_split (bool): If True, use all splits; otherwise only 'train'.

        Returns:
            dict: Dictionary with 'data' and 'target' arrays.

        Raises:
            ValueError: If the dataset name is not valid.
        """
        WrappedMedMNIST._ensure_root_dir()
        
        if name not in WrappedMedMNIST._all_datasets:
            raise ValueError(f"Dataset '{name}' not found. Use WrappedMedMNIST.all_datasets() to list all available datasets.")

        info = INFO[name]
        dataset_class = medmnist_dict[info["python_class"]]
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize(mean=[.5], std=[.5]))
        transform = transforms.Compose(transform_list)

        splits = ['train', 'val', 'test'] if full_split else ['train']

        datasets = [
            dataset_class(
                root='./medmnist_data',
                split=split,
                transform=transform,
                download=download
            )
            for split in splits
        ]

        images, labels = [], []
        for dataset in datasets:
            for img_tensor, label in dataset:
                img_np = img_tensor.numpy()
                if flatten:
                    img_np = img_np.flatten()
                else:
                    img_np = np.transpose(img_np, (1, 2, 0))
                images.append(img_np)
                labels.append(WrappedMedMNIST._process_label(label))

        return {
            'data': np.array(images),
            'target': np.array(labels)
        }
        
    @staticmethod
    def _ensure_root_dir(root: str = './MedMNIST/medmnist_data') -> None:
        """
        Ensure that the specified root directory exists. Creates the directory if it does not exist.

        Args:
            root (str): Path to the root directory where MedMNIST data will be stored. Defaults to './medmnist_data'.

        Returns:
            None

        Raises:
            OSError: If the directory cannot be created due to OS-related errors (e.g., permission denied).
        """
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            
    @staticmethod
    def _process_label(label):
        """
        Convert the input label to a standardized format (int or numpy.ndarray).

        Args:
            label (int, torch.Tensor, np.ndarray): The label to process.

        Returns:
            int or np.ndarray: The processed label in a consistent format.

        Raises:
            TypeError: If the label type is unsupported.
        """
        if isinstance(label, int):
            return label
        elif isinstance(label, torch.Tensor):
            return label.numpy()
        elif isinstance(label, np.ndarray):
            return label
        else:
            raise TypeError(f"Unsupported label type: {type(label)}")
