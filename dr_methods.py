import numpy as np
import sklearn.neighbors as sknn
from numpy.linalg import norm
import networkx as nx
from sklearn.manifold import Isomap
from ssk_isomap import SSKIsomap
from umap import UMAP                 # install with: pip install umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression


#################################################################################
# Fast K-ISOMAP implementation - consumes more memory, but it is a lot faster
# Pre-allocate matrices to speed up performance
#################################################################################
def KIsomap(dados, k, d, option, alpha=0.5):
    # Number of samples and features  
    n = dados.shape[0]
    m = dados.shape[1]
    # Matrix to store the principal components for each neighborhood
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:                   # Treat isolated points
            matriz_pcs[i, :, :] = np.eye(m)     # Eigenvectors in columns
        else:
            # Get the neighboring samples
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]                 
            # Projection matrix
            Wpca = maiores_autovetores  # Eigenvectors in columns
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                
                ##### Functions of the principal curvatures (definition of the metric)
                # We must choose one single option for each execution
                if option == 0:
                    B[i, j] = norm(delta)                  # metric A0 - Norms of the principal curvatures
                elif option == 1:
                    B[i, j] = delta[0]                     # metric A1 - Curvature of the first principal component
                elif option == 2:
                    B[i, j] = delta[-1]                    # metric A2 - Curvature of the last principal component
                elif option == 3:
                    B[i, j] = (delta[0] + delta[-1])/2     # metric A3 - Average between the curvatures of first and last principal components
                elif option == 4:
                    B[i, j] = np.sum(delta)/len(delta)     # metric A4 - Mean curvature
                elif option == 5:
                    B[i, j] = max(delta)                   # metric A5 - Maximum curvature
                elif option == 6:
                    B[i, j] = min(delta)                   # metric A6 - Minimum curvature
                elif option == 7:
                    B[i, j] = min(delta)*max(delta)        # metric A7 - Product between minimum and maximum curvatures
                elif option == 8:
                    B[i, j] = max(delta) - min(delta)      # metric A8 - Difference between maximum and minimum curvatures
                elif option == 9:
                    B[i, j] = 1 - np.exp(-delta.mean())     # metric A9 - Negative exponential kernel
                else:
                    B[i, j] = ((1-alpha)*A[i, j]/sum(A[i, :]) + alpha*norm(delta))      # alpha = 0 => regular ISOMAP, alpha = 1 => K-ISOMAP 
                
    # Computes geodesic distances using the previous selected metric
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)    
    # Remove infs and nans from B (if the graph is not connected)
    maximo = np.nanmax(B[B != np.inf])   
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = np.linalg.eig(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)    
    # Return the low dimensional coordinates
    return output.real



