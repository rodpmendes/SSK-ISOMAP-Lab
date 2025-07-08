"""Noise effects for dataset perturbation.

This module implements various noise addition techniques for data augmentation
and robustness testing in machine learning experiments.
"""

from typing import Callable, Dict
from skimage.util import random_noise
import numpy as np

def gaussian(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """Add Gaussian noise to the dataset.

    Args:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Dataset with Gaussian noise applied.
    """

    noise = np.random.normal(0, scale=scale_mag, size=dataset.shape)
    return dataset + noise

def salt_pepper(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """Add Salt & Pepper noise to the dataset.

    Args:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Proportion of elements affected by salt and pepper noise.

    Returns:
        np.ndarray: Dataset with Salt & Pepper noise applied.
    """

    noise = random_noise(dataset, mode='s&p', amount=scale_mag)
    return dataset + noise

def poison(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """Inject extreme values (poison noise) into random samples.

    Args:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Fraction of samples to poison (between 0 and 1).

    Returns:
        np.ndarray: Dataset with poison noise applied.
    """
    dataset_with_noise = dataset.copy()
    n_samples = dataset.shape[0]
    n_poison = int(scale_mag * n_samples)
    poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    max_val = np.max(dataset)
    noise = np.random.uniform(low=2 * max_val, high=3 * max_val, size=dataset[poison_indices].shape)
    dataset_with_noise[poison_indices] = noise
    return dataset_with_noise

def speckle(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """
    Add Speckle noise (multiplicative noise) to the dataset.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Intensity of speckle noise.
    
    Returns:
        np.ndarray: Dataset with Speckle noise added.
    """
    
    noise = np.random.randn(*dataset.shape)
    dataset_with_noise = dataset + dataset * noise * scale_mag
    return dataset_with_noise

def uniform(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """
    Add Uniform noise to the dataset.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Range of uniform noise (-scale_mag to +scale_mag).
    
    Returns:
        np.ndarray: Dataset with Uniform noise added.
    """
    
    noise = np.random.uniform(-scale_mag, scale_mag, size=dataset.shape)
    dataset_with_noise = dataset.copy() + noise
    return dataset_with_noise

def swap(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """Randomly swap feature values within the dataset.

    Args:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Proportion of total elements to swap.

    Returns:
        np.ndarray: Dataset with swapped feature values.
    """
    
    dataset_with_noise = dataset.copy()
    n_samples, n_features = dataset.shape
    n_swaps = int(scale_mag * n_samples * n_features)

    for _ in range(n_swaps):
        i1, j1 = np.random.randint(n_samples), np.random.randint(n_features)
        i2, j2 = np.random.randint(n_samples), np.random.randint(n_features)
        dataset_with_noise[i1, j1], dataset_with_noise[i2, j2] = (
            dataset_with_noise[i2, j2],
            dataset_with_noise[i1, j1],
        )

    return dataset_with_noise

def masking(dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """Apply masking noise by setting random values to zero.

    Args:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Proportion of elements to mask (between 0 and 1).

    Returns:
        np.ndarray: Dataset with masked values.
    """
    
    dataset_with_noise = dataset.copy()
    n_samples, n_features = dataset.shape
    n_mask = int(scale_mag * n_samples * n_features)

    indices = np.unravel_index(
        np.random.choice(n_samples * n_features, n_mask, replace=False),
        (n_samples, n_features),
    )
    dataset_with_noise[indices] = 0
    return dataset_with_noise

def apply_noise_type(noise_type: str, dataset: np.ndarray, scale_mag: float) -> np.ndarray:
    """Apply the specified noise type to the dataset.

    Args:
        noise_type (str): Noise type name. Options: ['gaussian', 'salt_pepper', 'poison', 'speckle', 'uniform', 'swap', 'masking'].
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Noise intensity or proportion.

    Returns:
        np.ndarray: Dataset with the specified noise applied.

    Raises:
        ValueError: If an invalid noise type is provided.
    """
    
    noise_functions: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
        'gaussian': gaussian,
        'salt_pepper': salt_pepper,
        'poison': poison,
        'speckle': speckle,
        'uniform': uniform,
        'swap': swap,
        'masking': masking,
    }

    if noise_type not in noise_functions:
        raise ValueError(
            f"Invalid noise type '{noise_type}'. Available types are: {list(noise_functions.keys())}"
        )

    return noise_functions[noise_type](dataset, scale_mag)