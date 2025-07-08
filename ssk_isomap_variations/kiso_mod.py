import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from numpy.linalg import norm, svd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def gerar_swiss_roll(n=1000, noise=0.05):
    X, _ = make_swiss_roll(n_samples=n, noise=noise)
    return X

def triedro_frenet(p1, p2, p3):
    T = (p2 - p1)
    T /= norm(T)
    N = (p3 - p2) - T * np.dot(p3 - p2, T)
    N /= norm(N)
    B = np.cross(T, N)
    return np.vstack([T, N, B])

def calcular_distancia_so3(A, B):
    R = A.T @ B
    u, _, vt = svd(R)
    R_ortho = u @ vt
    angle = np.arccos((np.trace(R_ortho) - 1) / 2)
    return np.abs(angle)

def calcular_distancia_se(W1, W2, c1, c2, beta=1.0):
    rot = calcular_distancia_so3(W1, W2)
    trans = norm(c1 - c2)
    return rot + beta * trans

def KIsomap_mod(dados, k=10, d=3, option=10):
    n, m = dados.shape
    knnGraph = kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()

    matriz_pcs = np.zeros((n, m, m))
    centros = np.zeros((n, m))

    for i in range(n):
        vizinhos = A[i, :].nonzero()[0]
        if len(vizinhos) < 3:
            matriz_pcs[i] = np.eye(m)
            centros[i] = dados[i]
        else:
            amostras = dados[vizinhos]
            centros[i] = amostras.mean(axis=0)
            cov = np.cov(amostras.T)
            _, vecs = np.linalg.eigh(cov)
            matriz_pcs[i] = vecs[:, ::-1]

    B = A.copy()
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                if option == 10:
                    viz_i = A[i].nonzero()[0]
                    if len(viz_i) >= 3:
                        p1, p2, p3 = dados[viz_i[:3]]
                        triedro = triedro_frenet(p1, p2, p3)
                        curvatura = norm(triedro[1])  # norma da normal
                        B[i, j] = curvatura
                    else:
                        B[i, j] = A[i, j]
                elif option == 11:
                    B[i, j] = calcular_distancia_so3(matriz_pcs[i], matriz_pcs[j])
                elif option == 12:
                    B[i, j] = calcular_distancia_se(matriz_pcs[i], matriz_pcs[j], centros[i], centros[j])
                elif option == 13:
                    alpha = 1
                    delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                    B[i, j] = ((1-alpha)*A[i, j]/sum(A[i, :]) + alpha*norm(delta))      # alpha = 0 => regular ISOMAP, alpha = 1 => K-ISOMAP 
                elif option == 14:
                    alpha = 0
                    delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                    B[i, j] = ((1-alpha)*A[i, j]/sum(A[i, :]) + alpha*norm(delta))      # alpha = 0 => regular ISOMAP, alpha = 1 => K-ISOMAP 

    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)
    H = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * H @ (D**2) @ H
    vals, vecs = np.linalg.eigh(K)
    idx = np.argsort(vals)[::-1]
    return vecs[:, idx[:d]] * np.sqrt(vals[idx[:d]])

# === Execução para comparação ===
X = gerar_swiss_roll()

embedding_frenet = KIsomap_mod(X, k=10, d=3, option=10)
embedding_so3 = KIsomap_mod(X, k=10, d=3, option=13)
embedding_se3 = KIsomap_mod(X, k=10, d=3, option=14)

# Agora você pode plotar cada um:
# embedding_frenet, embedding_so3, embedding_se3














def plot_embeddings(original, fren, so3, se3):
    fig = plt.figure(figsize=(18, 5))

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=5)
    ax.set_title('Swiss Roll Original')

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    ax.scatter(fren[:, 0], fren[:, 1], fren[:, 2], c='red', s=5)
    ax.set_title('K-Isomap - Frenet-Serret (option=10)')

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    ax.scatter(so3[:, 0], so3[:, 1], so3[:, 2], c='green', s=5)
    ax.set_title('K-Isomap')

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    ax.scatter(se3[:, 0], se3[:, 1], se3[:, 2], c='purple', s=5)
    ax.set_title('Regular Isomap')

    plt.tight_layout()
    plt.show()
    
    a = 'visualization here'
    
    

# Executar visualização com os embeddings gerados
plot_embeddings(X, embedding_frenet, embedding_so3, embedding_se3)
