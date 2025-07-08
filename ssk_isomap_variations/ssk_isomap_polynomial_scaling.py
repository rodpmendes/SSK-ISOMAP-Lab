import numpy as np
import sklearn.neighbors as sknn
from numpy.linalg import norm
import networkx as nx
import classification_utils as clustering
import scipy as sp
from typing import Tuple, List

def select_edges(cov_matrix: np.ndarray, proportion: float, selection_mode: str) -> List[Tuple[int, int]]:
    """
    Selects a subset of edges based on a given network centrality measure.

    Args:
        cov_matrix (np.ndarray): Covariance matrix (graph adjacency weights).
        proportion (float): Proportion of edges to select.
        selection_mode (str): Centrality measure for selection.

    Returns:
        List[Tuple[int, int]]: List of selected edge tuples (node indices).
    """
    G = nx.from_numpy_array(cov_matrix)

    if selection_mode == "betweenness_centrality":
        centrality = nx.edge_betweenness_centrality(G)
    elif selection_mode == "degree_centrality":
        centrality = {edge: G.degree(edge[0]) + G.degree(edge[1]) for edge in G.edges}
    elif selection_mode == "closeness_centrality":
        centrality = {edge: nx.closeness_centrality(G, edge[0]) + nx.closeness_centrality(G, edge[1]) for edge in G.edges}
    elif selection_mode == "eigenvector_centrality":
        try:
            eig_cent = nx.eigenvector_centrality_numpy(G)
            centrality = {edge: eig_cent[edge[0]] + eig_cent[edge[1]] for edge in G.edges}
        except nx.NetworkXException as e:
            print(f"Eigenvector centrality failed: {e}")
            centrality = {edge: 0 for edge in G.edges}
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    # Sort edges by centrality descending
    sorted_edges = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    # Select top proportion
    num_edges_to_select = max(1, int(len(sorted_edges) * proportion))

    # Extract only edge tuples (without the centrality values)
    selected_edges = [edge for edge, _ in sorted_edges[:num_edges_to_select]]

    return selected_edges


# SSK-ISOMAP implementation
def SSKIsomap(dados, k, d, target, prediction_mode="GMM", proportion=0.1, selection_mode: str = "betweenness_centrality") -> np.ndarray:
    """
    Semi-Supervised K-Isomap using patch-wise tangent spaces and edge reweighting based on network centrality.

    Parameters:
        dados (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of neighbors for KNN graph.
        d (int): Number of output dimensions.
        target (np.ndarray): Ground truth labels used for semi-supervised learning.
        prediction_mode (str, optional): Label estimation method for unlabeled data points.
            Options include 'GMM' and 'DBSCAN'. Default is 'GMM'.
        proportion (float, optional): Proportion of top-centrality edges to be reweighted
            based on label information. Range: 0.0 to 1.0. Default is 0.1.
        selection_mode (str, optional): Centrality measure used for edge selection.
            Options: 'betweenness_centrality', 'degree_centrality', 'closeness_centrality',
            'eigenvector_centrality'. Default is 'betweenness_centrality'.

    Returns:
        np.ndarray: Reduced data representation of shape (n_samples, d).
    """
    
    if not clustering.is_valid_mode(prediction_mode):
        print('prediction not implemented')
        return
    
    n = dados.shape[0]
    m = dados.shape[1]
    # Matrix to store the tangent spaces
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            matriz_pcs[i, :, :] = Wpca
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                B[i, j] = np.linalg.norm(delta)
                
    # Apply Gaussian Mixture Model (GMM) or DBSCAN or another mode implemented 
    self_labels = clustering.predict(prediction_mode, dados, target)
    
    #Semi-Supervised
    supervised_edges = select_edges(B, proportion, selection_mode)
    for tup in supervised_edges:
        i, j = tup
        
        delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
        
        
        
        ##---
        # Polynomial scaling parameters
        alpha = 0.5   # Same class attraction strength
        gamma = 2.0   # Different class repulsion strength
        epsilon = 1e-6  # Small value to avoid zero weights
        min_weight = 1e-6  # Minimum allowed weight to keep graph connected

        for (i, j) in supervised_edges:
            dist = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
            
            # Proteção contra valores negativos ou zero
            dist = np.abs(dist)
            dist_value_min = max(np.min(dist), epsilon)
            dist_value_max = max(np.max(dist), epsilon)
            
            # Normalização opcional (se quiser, pode comentar se não quiser usar)
            # max_dist_in_patch = np.max(dist)
            # if max_dist_in_patch > 0:
            #     dist_value_min /= max_dist_in_patch
            #     dist_value_max /= max_dist_in_patch

            # Aplicar a função polinomial com floor para evitar zero
            if self_labels[i] == self_labels[j]:
                weight = dist_value_min ** alpha
                B[i, j] = max(weight, min_weight)
            else:
                weight = dist_value_max ** gamma
                B[i, j] = max(weight, min_weight)

            # Log opcional para debugar (remova depois de validar)
            # print(f"Edge ({i},{j}): dist_min={dist_value_min:.6f}, dist_max={dist_value_max:.6f}, weight={B[i,j]:.6f}")
        ##---
        
           
    # Computes geodesic distances in B
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])  
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    return output