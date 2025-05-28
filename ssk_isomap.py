import numpy as np
import sklearn.neighbors as sknn
from numpy.linalg import norm
import networkx as nx
import utils.clustering as clustering

def select_edges(cov_matrix, proportion=.1, selection_mode="betweenness_centrality"):
    """
    Selects a proportion of edges from a graph based on a centrality criterion.

    Parameters:
        cov_matrix (np.ndarray): Matrix representing edge weights (e.g., patch distances).
        proportion (float): Proportion of top edges to select.
        selection_mode (str): Criterion to use ('betweenness_centrality').

    Returns:
        list: List of selected edges with highest centrality.
    """
    
    if selection_mode=="betweenness_centrality":
        G = nx.from_numpy_matrix(cov_matrix)
        edge_betweenness = nx.edge_betweenness_centrality(G)
        #for edge, centrality in edge_betweenness.items():
        #    print(f"Betweenness Centrality = {centrality:.6f} - Aresta {edge}")
        
        # betweenness centrality em ordem decrescente
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

        # número de arestas a serem selecionadas (proporção do total)
        num_edges_to_select = int(len(sorted_edges) * proportion)

        # Seleciona a proporção do parâmetro com maiores betweenness centrality
        selected_edges = sorted_edges[:num_edges_to_select]
    else:
        selected_edges = None

    return selected_edges

# SSK-ISOMAP implementation
def SSKIsomap(dados, k, d, target, prediction_mode="GMM", proportion=0.1):
    """
    Semi-Supervised K-Isomap using patch-wise tangent spaces and edge reweighting based on network centrality.

    Parameters:
        dados (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of neighbors for KNN graph.
        d (int): Number of output dimensions.
        target (np.ndarray): Ground truth labels used for semi-supervised learning.
        prediction_mode (str): Mode for label propagation or clustering (e.g., 'GMM').
        proportion (float): Proportion of top-centrality edges to semi-supervise.

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
    selected_edges = select_edges(B, proportion, "betweenness_centrality")
    for tup in selected_edges:
        i, j = tup[0]
        delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                
        if self_labels[i] == self_labels[j]:
            B[i, j] = min(delta)
        else:
            B[i, j] = max(delta) 
            
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