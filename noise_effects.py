from skimage.util import random_noise
import numpy as np

def gaussian(dataset, scale_mag):
    """
    Add Gaussian noise to the dataset.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Standard deviation of the Gaussian noise.
    
    Returns:
        np.ndarray: Dataset with Gaussian noise added.
    """
    
    # Generate noise
    noise = np.random.normal(0, scale=scale_mag, size=dataset.shape)
    
    # Apply noise in feature level
    dataset_with_noise = dataset.copy() + noise
    
    return dataset_with_noise

def salt_pepper(dataset, scale_mag):
    """
    Add Salt & Pepper noise to the dataset.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Amount of noise to apply (between 0 and 1).
    
    Returns:
        np.ndarray: Dataset with Salt & Pepper noise added.
    """
    
    # Apply noise salt and pepper
    noise = random_noise(dataset, mode='s&p', amount=scale_mag)
    
    # Apply noise in feature level
    dataset_with_noise = dataset.copy() + noise
    return dataset_with_noise

def poison(dataset, scale_mag):
    """
    Add Poison noise by injecting extreme values in random samples.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Fraction of samples to poison (between 0 and 1).
    
    Returns:
        np.ndarray: Dataset with Poison noise applied.
    """
    
    dataset_with_noise = dataset.copy()
    n_samples = dataset.shape[0]
    
    # Define quantos dados serão "envenenados"
    n_poison = int(scale_mag * n_samples)
    
    # Seleciona índices aleatórios para aplicar o ruído
    poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    
    # Aplica ruído extremo (valores fora da distribuição) nas amostras selecionadas
    max_val = np.max(dataset)
    min_val = np.min(dataset)
    noise = np.random.uniform(low=2 * max_val, high=3 * max_val, size=dataset[poison_indices].shape)
    
    dataset_with_noise[poison_indices] = noise
    return dataset_with_noise

def speckle(dataset, scale_mag):
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

def uniform(dataset, scale_mag):
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

def swap(dataset, scale_mag):
    """
    Randomly swap values between features in the dataset.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Proportion of swaps relative to total elements.
    
    Returns:
        np.ndarray: Dataset with randomly swapped feature values.
    """
    
    dataset_with_noise = dataset.copy()
    n_samples, n_features = dataset.shape
    n_swaps = int(scale_mag * n_samples * n_features)
    
    for _ in range(n_swaps):
        i1, j1 = np.random.randint(n_samples), np.random.randint(n_features)
        i2, j2 = np.random.randint(n_samples), np.random.randint(n_features)
        dataset_with_noise[i1, j1], dataset_with_noise[i2, j2] = dataset_with_noise[i2, j2], dataset_with_noise[i1, j1]
    
    return dataset_with_noise

def masking(dataset, scale_mag):
    """
    Apply masking noise by setting random values to zero.
    
    Parameters:
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Proportion of values to be masked.
    
    Returns:
        np.ndarray: Dataset with masking noise applied.
    """
    
    dataset_with_noise = dataset.copy()
    n_samples, n_features = dataset.shape
    n_mask = int(scale_mag * n_samples * n_features)
    
    indices = np.unravel_index(np.random.choice(n_samples * n_features, n_mask, replace=False),
                               (n_samples, n_features))
    dataset_with_noise[indices] = 0  # ou np.nan se preferir
    return dataset_with_noise

def apply_noise_type(noise_type, dataset, scale_mag):
    """
    Apply the specified type of noise to the dataset.
    
    Parameters:
        noise_type (str): One of ['gaussian', 'salt_pepper', 'poison', 'speckle', 
                                 'uniform', 'swap', 'masking'].
        dataset (np.ndarray): Original dataset.
        scale_mag (float): Magnitude or intensity of the noise.
    
    Returns:
        np.ndarray: Noisy dataset.
    """
    
    switch = {
        'gaussian': gaussian,
        'salt_pepper': salt_pepper,
        'poision': poison,
        'speckle': speckle,
        'uniform': uniform,
        'swap': swap,
        'masking': masking
    }
    func = switch.get(noise_type, lambda: print("Invalid noise type"))
    
    return func(dataset, scale_mag)