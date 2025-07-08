from sklearn.feature_selection import VarianceThreshold

def reduce_features_variance_threshold(dataset_data: np.ndarray, target_num_features: int) -> np.ndarray:
    """Reduces the number of features (m) using Variance Threshold feature selection.

    Args:
        dataset_data (np.ndarray): Original feature matrix of shape (n_samples, n_features).
        target_num_features (int): Desired number of features after reduction.

    Returns:
        np.ndarray: Reduced feature matrix with shape (n_samples, target_num_features).
    """
    selector = VarianceThreshold()
    filtered_data = selector.fit_transform(dataset_data)

    # If still more features than desired, apply PCA as fallback
    if filtered_data.shape[1] > target_num_features:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_num_features)
        reduced_data = pca.fit_transform(filtered_data)
    else:
        reduced_data = filtered_data

    return reduced_data
