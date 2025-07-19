#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-Supervised K-ISOMAP via Complex Network Properties
Python code for the Experimental Setup to reproduce the results of the paper
"""

# Imports
import warnings
import numpy as np
import scipy as sp
from sklearn import preprocessing
import json
from typing import Any, List, Tuple
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from classification_utils import Clustering
from noise_effects import apply_noise_type

from dr_methods import Isomap
from dr_methods import KIsomap
from dr_methods import SSKIsomap
from dr_methods import UMAP
from dr_methods import TSNE
from dr_methods import LocallyLinearEmbedding
from dr_methods import SpectralEmbedding
from dr_methods import KernelPCA
from dr_methods import LinearDiscriminantAnalysis
from dr_methods import PLSRegression

from config import *

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')


"""
  initiate pipeline code
"""
def preprocess_data(dataset_config: DatasetConfig, dataset_data: Any, dataset_target: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess dataset by encoding categories, handling NaNs, scaling,
    and applying sample and dimensionality reductions.

    This includes:
    - Sparse to dense conversion
    - Categorical encoding
    - NaN handling
    - Standard scaling
    - Sample reduction (n)
    - Dimensionality reduction (m)
    - Printing n, m, c before and after reduction

    Args:
        dataset_config (DatasetConfig): Dataset configuration with reduction flags.
        dataset_data (Any): Raw dataset features.
        dataset_target (Any): Target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed features and target labels.
    """
    # Track original dataset shape
    n_original = dataset_data.shape[0]
    m_original = dataset_data.shape[1]
    c_original = len(np.unique(dataset_target))

    # Handle sparse to dense
    if sp.sparse.issparse(dataset_data):
        dataset_data = dataset_data.todense()
    dataset_data = np.asarray(dataset_data)

    # Encode pandas categorical columns
    if hasattr(dataset_data, 'select_dtypes'):
        cat_cols = dataset_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            dataset_data[col] = le.fit_transform(dataset_data[col].astype(str))
        dataset_data = dataset_data.to_numpy()
        
    # Encode numpy object columns
    if dataset_data.dtype.kind in {'O', 'U', 'S'}:
        encoded_data = []
        for col in dataset_data.T:
            try:
                col = col.astype(float)
            except ValueError:
                le = LabelEncoder()
                col = le.fit_transform(col.astype(str))
            encoded_data.append(col)
        dataset_data = np.array(encoded_data).T

    # Replace NaNs
    dataset_data = np.nan_to_num(dataset_data)

    # Standardize
    dataset_data = preprocessing.scale(dataset_data).astype(np.float64)

    # Sample reduction (reduce n)
    if dataset_config.reduce_samples:
        dataset_data, _, dataset_target, _ = train_test_split(
            dataset_data,
            dataset_target,
            train_size=dataset_config.sample_percentage,
            random_state=42
        )

    # Dimensionality reduction (reduce m) with automatic check
    if dataset_config.reduce_dim:
        max_components = min(dataset_config.num_features, dataset_data.shape[1])
        pca = PCA(n_components=max_components)
        dataset_data = pca.fit_transform(dataset_data)

    # Track reduced dataset shape
    n_reduced = dataset_data.shape[0]
    m_reduced = dataset_data.shape[1]
    c_reduced = len(np.unique(dataset_target))

    #print(f"Dataset: {dataset_config.name}")
    #print(f"n original: {n_original}, n reduced: {n_reduced}")
    #print(f"m original: {m_original}, m reduced: {m_reduced}")
    #print(f"c original: {c_original}, c reduced: {c_reduced}")
    #print("-" * 50)

    # Encode target labels if needed
    if hasattr(dataset_target, 'unique'):
        le_target = LabelEncoder()
        dataset_target = le_target.fit_transform(dataset_target)
    else:
        dataset_target = np.array(dataset_target)

    return dataset_data, dataset_target

def normalize_metric(metric_list: List[float], split_sizes: List[int]) -> List[List[float]]:
    """Normalize a combined metric list and split it back into method-specific segments.

    Args:
        metric_list (List[float]): Combined metric values for all methods.
        split_sizes (List[int]): Number of metric values per method.

    Returns:
        List[List[float]]: Normalized metric values split per method.
    """
    metric_array = np.array(metric_list)
    min_val, max_val = np.min(metric_array), np.max(metric_array)
    if max_val > min_val:
        normalized = (metric_array - min_val) / (max_val - min_val)
    else:
        normalized = metric_array
    result = []
    start = 0
    for size in split_sizes:
        result.append(normalized[start:start + size].tolist())
        start += size
    return result


def RunExperiments(cfg: ExperimentConfig) -> None:
    """Run dimensionality reduction and clustering experiments on all configured datasets.

    Args:
        cfg (ExperimentConfig): Full experimental configuration.
    """
    results = {}
    
    for dataset in cfg.datasets:
        X = dataset.data
        dataset_data, dataset_target = preprocess_data(dataset, X['data'], X['target'])
        dataset_name = dataset.name
        n, m, c = dataset_data.shape[0], dataset_data.shape[1], len(np.unique(dataset_target))
        nn = round(np.sqrt(n))
        print(f'Running experiments for {dataset_name} | n={n} | m={m} | c={c} | k-NN={nn}')

        continue
    
        # Define metrics to be used
        metrics = ['ri', 'ch', 'fm', 'v', 'dbs', 'ss']
        
        # Define centrality selection modes and supervision proportions
        selection_modes = ["betweenness_centrality", "degree_centrality", "closeness_centrality", "eigenvector_centrality"]
        proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Start with RAW and ISOMAP
        methods = ['RAW', 'ISOMAP']

        # Append all SSKISOMAP variants in the desired order
        for selection_mode in selection_modes:
            for proportion in proportions:
                method_key = f'SSKISOMAP_{selection_mode}_p{int(proportion * 100)}'
                methods.append(method_key)

        # Append the remaining dimensionality reduction methods
        methods.extend(['KISOMAP', 'UMAP', 'TSNE', 'LLE', 'SE', 'KPCA', 'LDA', 'PLS'])

        # Initialize results dictionary for all methods and metrics
        results_per_method = {method: {metric: [] for metric in metrics} for method in methods}
        
        
        cluster_type = cfg.clustering_config.chosen_cluster.value

        for magnitude in cfg.noise_config.get_magnitudes():
            current_data = dataset_data.copy()
            if cfg.noise_config.should_apply_noise():
                current_data = apply_noise_type(cfg.noise_config.noise_type, current_data, magnitude)

            # RAW data clustering
            L = Clustering(current_data.T, dataset_target, f'RAW {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['RAW'][metric].append(L[idx])

            # ISOMAP
            iso_data = Isomap(n_neighbors=nn, n_components=2).fit_transform(current_data).T
            L = Clustering(iso_data, dataset_target, f'ISOMAP {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['ISOMAP'][metric].append(L[idx])

            # KISOMAP (select best option per metric)
            kiso_results = {metric: [] for metric in metrics}
            for option in range(11):
                try:
                    kiso_data = KIsomap(current_data, nn, 2, option)
                    if kiso_data.any():
                        L_kiso = Clustering(kiso_data.T, dataset_target, f'KISOMAP {dataset_name} option={option}', cluster_type)
                        for idx, metric in enumerate(metrics):
                            kiso_results[metric].append(L_kiso[idx])
                except Exception as e:
                    print(f"KISOMAP option {option} failed: {e}")
            for metric in metrics:
                if kiso_results[metric]:
                    best_idx = np.argmax(kiso_results[metric])
                    results_per_method['KISOMAP'][metric].append(kiso_results[metric][best_idx])


            
            # SSKISOMAP Variations Loop
            for selection_mode in selection_modes:
                for proportion in proportions:
                    try:
                        sskiso_data = SSKIsomap(
                            current_data, nn, 2, dataset_target,
                            proportion=proportion,
                            selection_mode=selection_mode
                        ).T

                        DR_method = f'SSKISOMAP_{selection_mode}_p{int(proportion * 100)}_{dataset_name}'
                        L = Clustering(sskiso_data, dataset_target, DR_method, cluster_type)

                        # Store each variation as a separate "method" in the results dictionary
                        method_key = f'SSKISOMAP_{selection_mode}_p{int(proportion * 100)}'

                        # Ensure the method_key exists
                        if method_key not in results_per_method:
                            results_per_method[method_key] = {metric: [] for metric in metrics}

                        # Append clustering metrics for this run
                        for idx, metric in enumerate(metrics):
                            results_per_method[method_key][metric].append(L[idx])

                    except Exception as e:
                        print(f"Error in SSKISOMAP with mode={selection_mode}, proportion={proportion}: {e}")


            # UMAP
            umap_data = UMAP(n_components=2).fit_transform(current_data).T
            L = Clustering(umap_data, dataset_target, f'UMAP {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['UMAP'][metric].append(L[idx])

            # TSNE
            tsne_data = TSNE(n_components=2).fit_transform(current_data).T
            L = Clustering(tsne_data, dataset_target, f'TSNE {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['TSNE'][metric].append(L[idx])

            # LLE
            lle_data = LocallyLinearEmbedding(n_components=2).fit_transform(current_data).T
            L = Clustering(lle_data, dataset_target, f'LLE {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['LLE'][metric].append(L[idx])

            # Spectral Embedding (SE)
            se_data = SpectralEmbedding(n_components=2).fit_transform(current_data).T
            L = Clustering(se_data, dataset_target, f'SE {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['SE'][metric].append(L[idx])

            # Kernel PCA (KPCA)
            kpca_data = KernelPCA(n_components=2).fit_transform(current_data).T
            L = Clustering(kpca_data, dataset_target, f'KPCA {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['KPCA'][metric].append(L[idx])

            # LDA
            n_components_lda = c - 1 if c > 2 else 1
            lda_data = LinearDiscriminantAnalysis(n_components=n_components_lda).fit_transform(current_data, dataset_target).T
            L = Clustering(lda_data, dataset_target, f'LDA {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['LDA'][metric].append(L[idx])

            # PLS
            pls_data = PLSRegression(n_components=c - 1).fit_transform(current_data, dataset_target)[0].T
            L = Clustering(pls_data, dataset_target, f'PLS {dataset_name}', cluster_type)
            for idx, metric in enumerate(metrics):
                results_per_method['PLS'][metric].append(L[idx])

        # Normalize results
        results_normalized = {method: {} for method in methods}
        for metric in metrics:
            combined_data = []
            split_sizes = []
            for method in methods:
                combined_data.extend(results_per_method[method][metric])
                split_sizes.append(len(results_per_method[method][metric]))
            normalized = normalize_metric(combined_data, split_sizes)
            for idx, method in enumerate(methods):
                results_normalized[method][metric] = normalized[idx]

        # Save both raw and normalized results per dataset
        results[dataset_name] = results_per_method
        results[f'{dataset_name}_norm'] = results_normalized

        # Merge with previous results and save
        try:
            with open(cfg.file_results, 'r') as f:
                previous_results = json.load(f)
        except FileNotFoundError:
            previous_results = {}

        results.update(previous_results)
        os.makedirs(os.path.dirname(cfg.file_results), exist_ok=True)
        with open(cfg.file_results, 'w') as f:
            json.dump(results, f)




def SetExperimentalSetup() -> ExperimentConfig:
    """
    Creates and returns the complete experimental configuration using lazy loading for datasets.
    """
    cfg = ExperimentConfig(
        datasets=load_all_datasets(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        file_results='results/json/dataset_results_all.json',
    )
    return cfg

def SetExperimentalSetupSmall() -> ExperimentConfig:
    """
    Creates and returns the small experimental configuration using lazy loading for datasets.
    """
    cfg = ExperimentConfig(
        datasets=load_small_datasets(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        file_results='results/json/dataset_results_small.json',
    )
    return cfg

def SetExperimentalSetupSmallNotReduced() -> ExperimentConfig:
    """
    Creates and returns the small not reduced experimental configuration using lazy loading for datasets.
    """
    cfg = ExperimentConfig(
        datasets=load_small_datasets_not_reduced(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        file_results='results/json/dataset_results_small_not_reduced.json',
    )
    return cfg

def SetExperimentalSetupValidation() -> ExperimentConfig:
    """
    Creates and returns an experimental configuration for validation purposes,
    using lazy loading for a single dataset and with noise disabled.

    Returns:
        ExperimentConfig: Configuração do experimento com um dataset e ruído desabilitado.
    """
    cfg = ExperimentConfig(
        datasets=load_tree_datasets(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        file_results='results/json/dataset_results_one.json',
    )
    return cfg

def SetExperimentalSetupSSKVariation() -> ExperimentConfig:
    """
    Creates and returns the small experimental configuration using lazy loading for datasets.
    """
    cfg = ExperimentConfig(
        datasets=load_small_datasets_not_reduced(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        file_results='results/json/dataset_results_ssk_variation_controled_scaling_all.json',
    )
    return cfg

def SetPaperSetup() -> ExperimentConfig:
    """
    Creates and returns the paper experimental configuration using lazy loading for datasets.
    """
    cfg = ExperimentConfig(
        datasets=load_paper_datasets(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        #file_results='results/json/dataset_results_paper_setup_min_max.json',
        #file_results='results/json/dataset_results_paper_setup_min_max_v2.json',
        #file_results='results/json/dataset_results_paper_setup.json',
        file_results='results/json/dataset_results_MedMNIST.json',
    )
    return cfg

def SetMedMNISTSetup() -> ExperimentConfig:
    """
    Creates and returns the paper experimental configuration using lazy loading for datasets.
    """
    cfg = ExperimentConfig(
        datasets=load_medMNIST_datasets(),
        noise_config=NoiseConfig(apply_noise=False, noise_type=NoiseType.NONE, max_std_dev=1.0),
        file_results='results/json/dataset_results_MedMNIST.json',
    )
    return cfg


def GenerateOverviewExperimentalConfig(cfg: ExperimentConfig) -> List[OverviewExperimentConfig]:
    """
        Generate an overview of experimental configuration for each dataset.

    Args:
        cfg (ExperimentConfig): Full experimental configuration containing datasets.
        
    Returns:
        List[OverviewExperimentConfig]: A list with metadata overview for each dataset.
    """
    results = []
    
    for dataset in cfg.datasets:
        X = dataset.data
        dataset_data, dataset_target = preprocess_data(dataset, X['data'], X['target'])
        dataset_name = dataset.name
        n, m, c = dataset_data.shape[0], dataset_data.shape[1], len(np.unique(dataset_target))
        nn = round(np.sqrt(n))
        
        #print(f'Running experiments for {dataset_name} | n={n} | m={m} | c={c} | k-NN={nn}')
        #print("-" * 50)
        
        print(f"n={n:04} | m={m:04} | c={c:04} | k-NN={nn:04} | {dataset_name}")
        
        overview = OverviewExperimentConfig(
            dataset_name=dataset_name,
            n=n,
            m=m,
            c=c,
            nn=nn
        )
        results.append(overview)
    
    return OverviewExperimentConfig  
        
def main():
    #cfg = SetExperimentalSetupValidation()
    #cfg = SetExperimentalSetup()
    #cfg = SetExperimentalSetupSmall()
    #cfg = SetExperimentalSetupSmallNotReduced()
    #cfg = SetExperimentalSetupSSKVariation()
    
    #cfg = SetPaperSetup()
    #GenerateOverviewExperimentalConfig(cfg)
    
    cfg = SetMedMNISTSetup()
    RunExperiments(cfg)

if __name__ == "__main__":
    main()
