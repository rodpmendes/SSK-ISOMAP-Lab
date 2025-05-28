#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Semi-Supervised K-ISOMAP via Complex Network Properties

Python code for the Experimental Setup to reproduce the results of the paper

"""

# Imports
import sys
import time
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from numpy import sqrt
from sklearn import preprocessing
import json

from sklearn.decomposition import PCA
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


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Plot the scatterplots dor the 2D output data
def PlotaDados(dados, labels, metodo):
    nclass = len(np.unique(labels))
    if metodo == 'LDA':
        if nclass == 2:
            return -1
    # Encode the labels as integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     
    # Map labels to numbers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    rotulos = np.array(rotulos)
    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']
    plt.figure(1)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()


#%%%%%%%%%%%%%%%%%%%%  Data loading

# OpenML datasets
datasets = [
            
            #To perform the experiments according to the article, uncomment the desired sets of datasets
            
            {"db": skdata.load_digits(), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='wine-quality-white', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='wine-quality-red', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='glass', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='ecoli', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='vowel', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='collins', version=4), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='energy-efficiency', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='balance-scale', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='diabetes', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='mfeat-karhunen', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='grub-damage', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='banknote-authentication', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='vehicle', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='ionosphere', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='wall-robot-navigation', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='CIFAR_10_small', version=1), "reduce_samples": True, "percentage":.2, "reduce_dim":False, "num_features": 30},             # 20% of the samples and 30-D
            # {"db": skdata.fetch_openml(name='artificial-characters', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},      # 25% of the samples
            # {"db": skdata.fetch_openml(name='waveform-5000', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='nursery', version=1), "reduce_samples": True, "percentage":.3, "reduce_dim":False, "num_features": 0},                     # 30% of the samples
            # {"db": skdata.fetch_openml(name='eye_movements', version=1), "reduce_samples": True, "percentage":.3, "reduce_dim":False, "num_features": 0},               # 30% of the samples
            # {"db": skdata.fetch_openml(name='zoo', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='thyroid-dis', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='one-hundred-plants-shape', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            
            #-------
            # {"db": skdata.fetch_openml(name='servo', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='car-evaluation', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='breast-tissue', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='Engine1', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='xd6', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='heart-h', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='steel-plates-fault', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='PhishingWebsites', version=1), "reduce_samples": True, "percentage":.1, "reduce_dim":False, "num_features": 0},           # 10% of the samples
            # {"db": skdata.fetch_openml(name='satimage', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                  # 25% of the samples
            # {"db": skdata.fetch_openml(name='led24', version=1), "reduce_samples": True, "percentage":.20, "reduce_dim":False, "num_features": 0},                     # 20% of the samples
            # {"db": skdata.fetch_openml(name='hayes-roth', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='rabe_131', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='prnn_synth', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='visualizing_environmental', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='diggle_table_a2', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='newton_hema', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='wisconsin', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='fri_c4_250_100', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='conference_attendance', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='tic-tac-toe', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='qsar-biodeg', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='spambase', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                  # 25% of the samples
            # {"db": skdata.fetch_openml(name='cmc', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            # {"db": skdata.fetch_openml(name='heart-statlog', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
            #----
            # {"db": skdata.fetch_openml(name='cnae-9', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 50},                     # 50-D
            # {"db": skdata.fetch_openml(name='AP_Breast_Kidney', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 500},          # 500-D
            # {"db": skdata.fetch_openml(name='AP_Endometrium_Breast', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 400},     # 400-D
            # {"db": skdata.fetch_openml(name='AP_Ovary_Lung', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},             # 100-D
            # {"db": skdata.fetch_openml(name='OVA_Uterus', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                # 100-D
            # {"db": skdata.fetch_openml(name='micro-mass', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                # 100-D
            # {"db": skdata.fetch_openml(name='har', version=1), "reduce_samples": True, "percentage":0.1, "reduce_dim":True, "num_features": 100},                      # 10%  of the samples and 100-D
            # {"db": skdata.fetch_openml(name='eating', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                    # 100-D
            # {"db": skdata.fetch_openml(name='oh5.wc', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 40},                     # 40-D
            # {"db": skdata.fetch_openml(name='leukemia', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 40},                   # 40-D
            # {"db": skdata.fetch_openml(name='pendigits', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                 # 25% of the samples
            # {"db": skdata.fetch_openml(name='mnist_784', version=1), "reduce_samples": False, "percentage":.5, "reduce_dim":True, "num_features": 50},                 # 50-D and 50% of the samples
            # {"db": skdata.fetch_openml(name='Fashion-MNIST', version=1), "reduce_samples": False, "percentage":.5, "reduce_dim":True, "num_features": 100},             # 100-D and 50% of the samples
]

plot_results = False
apply_noise = False    

# Noise parameters
noise_config = ['none', 'gaussian', 'salt_pepper', 'poison']
noise_type = noise_config[0]
if apply_noise:
    # Standard deviation (spread or “width”) of the distribution. Must be non-negative
    data_std_dev = 1 # for normalized data base
    
    # Define magnitude
    magnitude = np.linspace(0, data_std_dev, 11)
else:
    magnitude = np.linspace(0, 0, 1)

# Clustering parameters
cluster_config = ['gmm', 'kmeans', 'agglomerative']
CLUSTER = cluster_config[0]

# File result parameters
file_results = 'dataset_results.json'
results = {}

for dataset in datasets:
    
    X = dataset["db"]
    raw_data = X['data']
    dataset_data = X['data']
    dataset_target = X['target']
    dataset_name = X['details']['name']

    # Convert labels to integers
    label_list = []
    for x in dataset_target:
        if x not in label_list:  
            label_list.append(x)     
            
    # Map labels to respective numbers
    labels = []
    for x in dataset_target:  
        for i in range(len(label_list)):
            if x == label_list[i]:  
                labels.append(i)
    dataset_target = np.array(labels)

    # Some adjustments are require in opnML datasets
    # Categorical features must be encoded manually
    if type(dataset_data) == sp.sparse._csr.csr_matrix:
        dataset_data = dataset_data.todense()
        dataset_data = np.asarray(dataset_data)

    if not isinstance(dataset_data, np.ndarray):
        cat_cols = dataset_data.select_dtypes(['category']).columns
        dataset_data[cat_cols] = dataset_data[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dataset_data = dataset_data.to_numpy()

    # To remove NaNs
    dataset_data = np.nan_to_num(dataset_data)

    # Data standardization (to deal with variables having different units/scales)
    dataset_data = preprocessing.scale(dataset_data).astype(np.float64)

    # OPTIONAL: set this flag to True to reduce the number of samples
    reduce_samples = dataset["reduce_samples"]
    reduce_dim = dataset["reduce_dim"]

    if not reduce_samples and not reduce_dim:
        raw_data = dataset_data

    if reduce_samples:
        percentage = dataset["percentage"]
        dataset_data, garbage, dataset_target, garbage_t = train_test_split(dataset_data, dataset_target, train_size=percentage, random_state=42)
        raw_data = dataset_data

    # OPTIONAL: set this flag to True to reduce the dimensionality with PCA prior to metric learning
    if reduce_dim:
        num_features = dataset["num_features"]
        raw_data = dataset_data
        dataset_data = PCA(n_components=num_features).fit_transform(dataset_data)
    
    # Number of samples, features and classes
    n = dataset_data.shape[0]
    m = dataset_data.shape[1]
    c = len(np.unique(dataset_target))


    # Number of neighbors in KNN graph (patch size)
    nn = round(sqrt(n))                 
        
    # Number of neighbors
    nn = round(sqrt(n))     # Número de vizinhos = raiz quadrada de n
    
    #print()
    #print('Number of samples (n): ', n)
    #print('Number of features (m): ', m)
    #print('Number of classes (c): ', c)
    #print('Number of Neighbors in k-NN graph (k): ', nn)
    #print()
    #print('Press enter to continue...')
    #input()


    # RAW results
    ri_raw, ch_raw, fm_raw, v_raw, dbs_raw, ss_raw = [], [], [], [], [], []
    ri_raw_norm, ch_raw_norm, fm_raw_norm, v_raw_norm, dbs_raw_norm, ss_raw_norm = [], [], [], [], [], []
        
    # ISOMAP results
    ri_iso, ch_iso, fm_iso, v_iso, dbs_iso, ss_iso = [], [], [], [], [], []
    ri_iso_norm, ch_iso_norm, fm_iso_norm, v_iso_norm, dbs_iso_norm, ss_iso_norm = [], [], [], [], [], []
        
    # K-ISOMAP results
    ri_kiso, ch_kiso, fm_kiso, v_kiso, dbs_kiso, ss_kiso = [], [], [], [], [], []
    ch_kiso_norm, ri_kiso_norm, fm_kiso_norm, v_kiso_norm, dbs_kiso_norm, ss_kiso_norm = [], [], [], [], [], []
    ri_best_metric, ch_best_metric, fm_best_metric, v_best_metric, dbs_best_metric, ss_best_metric = [], [], [], [], [], []
    
    # SSK-ISOMAP results
    ri_sskiso, ch_sskiso, fm_sskiso, v_sskiso, dbs_sskiso, ss_sskiso = [], [], [], [], [], []
    ch_sskiso_norm, ri_sskiso_norm, fm_sskiso_norm, v_sskiso_norm, dbs_sskiso_norm, ss_sskiso_norm = [], [], [], [], [], []
    
    # UMAP results
    ri_umap, ch_umap, fm_umap, v_umap, dbs_umap, ss_umap = [], [], [], [], [], []
    ri_umap_norm, ch_umap_norm, fm_umap_norm, v_umap_norm, dbs_umap_norm, ss_umap_norm = [], [], [], [], [], []
    
    # TSNE results
    ri_tsne, ch_tsne, fm_tsne, v_tsne, dbs_tsne, ss_tsne = [], [], [], [], [], []
    ri_tsne_norm, ch_tsne_norm, fm_tsne_norm, v_tsne_norm, dbs_tsne_norm, ss_tsne_norm = [], [], [], [], [], []
    
    # LocallyLinearEmbedding results
    ri_lle, ch_lle, fm_lle, v_lle, dbs_lle, ss_lle = [], [], [], [], [], []
    ri_lle_norm, ch_lle_norm, fm_lle_norm, v_lle_norm, dbs_lle_norm, ss_lle_norm = [], [], [], [], [], []
    
    # SpectralEmbedding results
    ri_se, ch_se, fm_se, v_se, dbs_se, ss_se = [], [], [], [], [], []
    ri_se_norm, ch_se_norm, fm_se_norm, v_se_norm, dbs_se_norm, ss_se_norm = [], [], [], [], [], []
    
    # KernelPCA results
    ri_kpca, ch_kpca, fm_kpca, v_kpca, dbs_kpca, ss_kpca = [], [], [], [], [], []
    ri_kpca_norm, ch_kpca_norm, fm_kpca_norm, v_kpca_norm, dbs_kpca_norm, ss_kpca_norm = [], [], [], [], [], []
    
    # LDA results
    ri_lda, ch_lda, fm_lda, v_lda, dbs_lda, ss_lda = [], [], [], [], [], []
    ri_lda_norm, ch_lda_norm, fm_lda_norm, v_lda_norm, dbs_lda_norm, ss_lda_norm = [], [], [], [], [], []
    
    # PLS results
    ri_pls, ch_pls, fm_pls, v_pls, dbs_pls, ss_pls = [], [], [], [], [], []
    ri_pls_norm, ch_pls_norm, fm_pls_norm, v_pls_norm, dbs_pls_norm, ss_pls_norm = [], [], [], [], [], []
    
    for r in range(len(magnitude)):
            # Computes the results for all 10 curvature based metrics
            start = time.time()
            
            if apply_noise:
                dataset_data = apply_noise_type(noise_type, dataset_data, magnitude[r])
                raw_data = apply_noise_type(noise_type, dataset_data, magnitude[r])
            
            
            ############## RAW DATA
            print(dataset_name + ' RAW DATA result')
            print('-----------------')
            DR_method = 'RAW ' + dataset_name + ' cluster=' + CLUSTER
            if reduce_dim:
                L_ = Clustering(dataset_data.T, dataset_target, DR_method, CLUSTER)
            else:
                L_ = Clustering(raw_data.T, dataset_target, DR_method, CLUSTER)
            ri_raw.append(L_[0])
            ch_raw.append(L_[1])
            fm_raw.append(L_[2])
            v_raw.append(L_[3])
            dbs_raw.append(L_[4])
            ss_raw.append(L_[5])
            
            
            ############## Regular ISOMAP 
            print(dataset_name + ' ISOMAP result')
            print('---------------')
            model = Isomap(n_neighbors=nn, n_components=2)
            isomap_data = model.fit_transform(dataset_data)
            isomap_data = isomap_data.T
            DR_method = 'ISOMAP ' + dataset_name + ' cluster=' + CLUSTER
            L_iso = Clustering(isomap_data, dataset_target, DR_method, CLUSTER)
            ri_iso.append(L_iso[0])
            ch_iso.append(L_iso[1])
            fm_iso.append(L_iso[2])
            v_iso.append(L_iso[3])
            dbs_iso.append(L_iso[4])
            ss_iso.append(L_iso[5])
            
            ############## K-ISOMAP 
            ri, ch, fm, v, dbs, ss = [], [], [], [], [], []
            for i in range(11):
                DR_method = 'K-ISOMAP ' + dataset_name + ' option=' + str(i) + ' cluster=' + CLUSTER + ' mag=' + str(r)
                
                try:
                    dados_kiso = KIsomap(dataset_data, nn, 2, i)
                except Exception as e:
                    print(DR_method + " -------- def KIsomap error:", e)
                    dados_kiso = []
                    
                if dados_kiso.any():
                    L_kiso = Clustering(dados_kiso.T, dataset_target, DR_method, CLUSTER)
                    ri.append(L_kiso[0])
                    ch.append(L_kiso[1])
                    fm.append(L_kiso[2])
                    v.append(L_kiso[3])
                    dbs.append(L_kiso[4])
                    ss.append(L_kiso[5])
            finish = time.time()
            print(dataset_name + ' K-ISOMAP time: %f s' %(finish - start))
            print()
            
            # Find best result in terms of Rand index of metric function
            ri_star = max(ri)
            ri_kiso.append(ri_star)
            ri_best_metric.append(ri.index(ri_star))
                        
            # Find best result in terms of Calinski Harabasz Score of metric function
            ch_star = max(ch)
            ch_kiso.append(ch_star)
            ch_best_metric.append(ch.index(ch_star))
            
            # Find best result in terms of Fowlkes Mallows Score of metric function
            fm_star = max(fm)
            fm_kiso.append(fm_star)
            fm_best_metric.append(fm.index(fm_star))
            
            # Find best result in terms of V measure of metric function
            v_star = max(v)
            v_kiso.append(v_star)
            v_best_metric.append(v.index(v_star))
            
            # Find best result in terms of Davies Bouldin Score
            dbs_star = max(dbs)
            dbs_kiso.append(dbs_star)
            dbs_best_metric.append(dbs.index(dbs_star))
            
            # Find best result in terms of Silhouette Score
            ss_star = max(ss)
            ss_kiso.append(ss_star)
            ss_best_metric.append(ss.index(ss_star))
            
            
            ############## SSK-ISOMAP
            print(dataset_name + ' SSK-ISOMAP result')
            print('---------------')
            model = SSKIsomap(dataset_data, nn, 2, dataset_target)
            sskiso_data = model.fit_transform(dataset_data)
            sskiso_data = sskiso_data.T
            DR_method = 'SSK-ISOMAP ' + dataset_name + ' cluster=' + CLUSTER
            L_sskiso = Clustering(sskiso_data, dataset_target, DR_method, CLUSTER)
            ri_sskiso.append(L_sskiso[0])
            ch_sskiso.append(L_sskiso[1])
            fm_sskiso.append(L_sskiso[2])
            v_sskiso.append(L_sskiso[3])
            dbs_sskiso.append(L_sskiso[4])
            ss_sskiso.append(L_sskiso[5])
            
            
            ############## UMAP
            print(dataset_name + ' UMAP result')
            print('---------------')
            model = UMAP(n_components=2)
            umap_data = model.fit_transform(dataset_data)
            umap_data = umap_data.T
            DR_method = 'UMAP ' + dataset_name + ' cluster=' + CLUSTER
            L_umap = Clustering(umap_data, dataset_target, DR_method, CLUSTER)
            ri_umap.append(L_umap[0])
            ch_umap.append(L_umap[1])
            fm_umap.append(L_umap[2])
            v_umap.append(L_umap[3])
            dbs_umap.append(L_umap[4])
            ss_umap.append(L_umap[5])
            
            
            ############## T-SNE
            print(dataset_name + ' T-SNE result')
            print('---------------')
            model = TSNE(n_components=2)
            tsne_data = model.fit_transform(dataset_data)
            tsne_data = tsne_data.T
            DR_method = 'T-SNE ' + dataset_name + ' cluster=' + CLUSTER
            L_tsne = Clustering(tsne_data, dataset_target, DR_method, CLUSTER)
            ri_tsne.append(L_tsne[0])
            ch_tsne.append(L_tsne[1])
            fm_tsne.append(L_tsne[2])
            v_tsne.append(L_tsne[3])
            dbs_tsne.append(L_tsne[4])
            ss_tsne.append(L_tsne[5])
            
            
            ############## LLE LocallyLinearEmbedding
            print(dataset_name + ' LocallyLinearEmbedding result')
            print('---------------')
            model = LocallyLinearEmbedding(n_components=2)
            lle_data = model.fit_transform(dataset_data)
            lle_data = lle_data.T
            DR_method = 'LocallyLinearEmbedding ' + dataset_name + ' cluster=' + CLUSTER
            L_lle = Clustering(lle_data, dataset_target, DR_method, CLUSTER)
            ri_lle.append(L_lle[0])
            ch_lle.append(L_lle[1])
            fm_lle.append(L_lle[2])
            v_lle.append(L_lle[3])
            dbs_lle.append(L_lle[4])
            ss_lle.append(L_lle[5])
            
            
            ############## SpectralEmbedding
            print(dataset_name + ' SpectralEmbedding result')
            print('---------------')
            model = SpectralEmbedding(n_components=2)
            se_data = model.fit_transform(dataset_data)
            se_data = se_data.T
            DR_method = 'SpectralEmbedding ' + dataset_name + ' cluster=' + CLUSTER
            L_se = Clustering(se_data, dataset_target, DR_method, CLUSTER)
            ri_se.append(L_se[0])
            ch_se.append(L_se[1])
            fm_se.append(L_se[2])
            v_se.append(L_se[3])
            dbs_se.append(L_se[4])
            ss_se.append(L_se[5])
            
            
            ############## KernelPCA
            print(dataset_name + ' KernelPCA result')
            print('---------------')
            model = KernelPCA(n_components=2)
            kpca_data = model.fit_transform(dataset_data)
            kpca_data = kpca_data.T
            DR_method = 'KernelPCA ' + dataset_name + ' cluster=' + CLUSTER
            L_kpca = Clustering(kpca_data, dataset_target, DR_method, CLUSTER)
            ri_kpca.append(L_kpca[0])
            ch_kpca.append(L_kpca[1])
            fm_kpca.append(L_kpca[2])
            v_kpca.append(L_kpca[3])
            dbs_kpca.append(L_kpca[4])
            ss_kpca.append(L_kpca[5])
            
            
            ############## LDA
            print(dataset_name + ' T-SNE result')
            print('---------------')
            DR_method = 'LDA ' + dataset_name + ' cluster=' + CLUSTER
            if c > 2:
                model = LinearDiscriminantAnalysis(n_components=2)
            else:
                model = LinearDiscriminantAnalysis(n_components=1)
            lda_data = model.fit_transform(dataset_data, dataset_target)
            L_lda = Clustering(lda_data, dataset_target, DR_method, CLUSTER)
            ri_lda.append(L_lda[0])
            ch_lda.append(L_lda[1])
            fm_lda.append(L_lda[2])
            v_lda.append(L_lda[3])
            dbs_lda.append(L_lda[4])
            ss_lda.append(L_lda[5])


            ############## PLS
            print(dataset_name + ' PLS result')
            print('---------------')
            DR_method = 'PLS ' + dataset_name + ' cluster=' + CLUSTER
            model = PLSRegression(n_components=2)
            pls_data = model.fit_transform(dataset_data, y=dataset_target)
            L_pls = Clustering(pls_data, dataset_target, DR_method, CLUSTER)
            ri_pls.append(L_pls[0])
            ch_pls.append(L_pls[1])
            fm_pls.append(L_pls[2])
            v_pls.append(L_pls[3])
            dbs_pls.append(L_pls[4])
            ss_pls.append(L_pls[5])

            
    results[dataset_name] = { "RAW": [ri_raw, ch_raw, fm_raw, v_raw, dbs_raw, ss_raw], 
                              "ISOMAP": [ri_iso, ch_iso, fm_iso, v_iso, dbs_iso, ss_iso],
                              "KISOMAP": [ri_sskiso, ch_sskiso, fm_sskiso, v_sskiso, dbs_sskiso, ss_sskiso],
                              "SSKISOMAP": [ri_sskiso, ch_sskiso, fm_sskiso, v_sskiso, dbs_sskiso, ss_sskiso],
                              "UMAP": [ri_umap, ch_umap, fm_umap, v_umap, dbs_umap, ss_umap],
                              "TSNE": [ri_tsne, ch_tsne, fm_tsne, v_tsne, dbs_tsne, ss_tsne],
                              "LLE": [ri_lle, ch_lle, fm_lle, v_lle, dbs_lle, ss_lle],
                              "SE": [ri_se, ch_se, fm_se, v_se, dbs_se, ss_se],
                              "KPCA": [ri_kpca, ch_kpca, fm_kpca, v_kpca, dbs_kpca, ss_kpca],
                              "LDA": [ri_lda, ch_lda, fm_lda, v_lda, dbs_lda, ss_lda],
                              "PLS": [ri_pls, ch_pls, fm_pls, v_pls, dbs_pls, ss_pls]
                            }
        
    # normalize data results
    ri_data = ri_raw + ri_iso + ri_kiso + ri_sskiso + ri_umap + ri_tsne + ri_lle + ri_se + ri_kpca + ri_lda + ri_pls
    min1, max1 = np.min(ri_data), np.max(ri_data)
    ri_data_normalized = (ri_data - min1) / (max1 - min1)
    ri_raw_norm = ri_data_normalized[:len(ri_raw)].tolist()
    ri_iso_norm = ri_data_normalized[len(ri_raw):len(ri_raw)+len(ri_iso)].tolist()
    ri_kiso_norm = ri_data_normalized[len(ri_raw)+len(ri_iso):len(ri_raw)+len(ri_iso)+len(ri_kiso)].tolist()
    ri_sskiso_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)].tolist()
    ri_umap_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)].tolist()
    ri_tsne_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)].tolist()
    ri_lle_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)].tolist()
    ri_se_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se)].tolist()
    ri_kpca_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se)+len(ri_kpca)].tolist()
    ri_lda_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se)+len(ri_kpca):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se)+len(ri_kpca)+len(ri_lda)].tolist()
    ri_pls_norm = ri_data_normalized[len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se)+len(ri_kpca)+len(ri_lda):len(ri_raw)+len(ri_iso)+len(ri_kiso)+len(ri_sskiso)+len(ri_umap)+len(ri_tsne)+len(ri_lle)+len(ri_se)+len(ri_kpca)+len(ri_lda)+len(ri_pls)].tolist()
    
    ch_data = ch_raw + ch_iso + ch_kiso + ch_sskiso + ch_umap + ch_tsne + ch_lle + ch_se + ch_kpca + ch_lda + ch_pls
    min1, max1 = np.min(ch_data), np.max(ch_data)
    ch_data_normalized = (ch_data - min1) / (max1 - min1)
    ch_raw_norm = ch_data_normalized[:len(ch_raw)].tolist()
    ch_iso_norm = ch_data_normalized[len(ch_raw):len(ch_raw)+len(ch_iso)].tolist()
    ch_kiso_norm = ch_data_normalized[len(ch_raw)+len(ch_iso):len(ch_raw)+len(ch_iso)+len(ch_kiso)].tolist()
    ch_sskiso_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)].tolist()
    ch_umap_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)].tolist()
    ch_tsne_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)].tolist()
    ch_lle_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)].tolist()
    ch_se_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se)].tolist()
    ch_kpca_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se)+len(ch_kpca)].tolist()
    ch_lda_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se)+len(ch_kpca):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se)+len(ch_kpca)+len(ch_lda)].tolist()
    ch_pls_norm = ch_data_normalized[len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se)+len(ch_kpca)+len(ch_lda):len(ch_raw)+len(ch_iso)+len(ch_kiso)+len(ch_sskiso)+len(ch_umap)+len(ch_tsne)+len(ch_lle)+len(ch_se)+len(ch_kpca)+len(ch_lda)+len(ch_pls)].tolist()


    fm_data = fm_raw + fm_iso + fm_kiso + fm_sskiso + fm_umap + fm_tsne + fm_lle + fm_se + fm_kpca + fm_lda + fm_pls
    min1, max1 = np.min(fm_data), np.max(fm_data)
    fm_data_normalized = (fm_data - min1) / (max1 - min1)
    fm_raw_norm = fm_data_normalized[:len(fm_raw)].tolist()
    fm_iso_norm = fm_data_normalized[len(fm_raw):len(fm_raw)+len(fm_iso)].tolist()
    fm_kiso_norm = fm_data_normalized[len(fm_raw)+len(fm_iso):len(fm_raw)+len(fm_iso)+len(fm_kiso)].tolist()
    fm_sskiso_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)].tolist()
    fm_umap_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)].tolist()
    fm_tsne_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)].tolist()
    fm_lle_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)].tolist()
    fm_se_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se)].tolist()
    fm_kpca_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se)+len(fm_kpca)].tolist()
    fm_lda_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se)+len(fm_kpca):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se)+len(fm_kpca)+len(fm_lda)].tolist()
    fm_pls_norm = fm_data_normalized[len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se)+len(fm_kpca)+len(fm_lda):len(fm_raw)+len(fm_iso)+len(fm_kiso)+len(fm_sskiso)+len(fm_umap)+len(fm_tsne)+len(fm_lle)+len(fm_se)+len(fm_kpca)+len(fm_lda)+len(fm_pls)].tolist()    
    
    
    v_data = v_raw + v_iso + v_kiso + v_sskiso + v_umap + v_tsne + v_lle + v_se + v_kpca + v_lda + v_pls
    min1, max1 = np.min(v_data), np.max(v_data)
    v_data_normalized = (v_data - min1) / (max1 - min1)
    v_raw_norm = v_data_normalized[:len(v_raw)].tolist()
    v_iso_norm = v_data_normalized[len(v_raw):len(v_raw)+len(v_iso)].tolist()
    v_kiso_norm = v_data_normalized[len(v_raw)+len(v_iso):len(v_raw)+len(v_iso)+len(v_kiso)].tolist()
    v_sskiso_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)].tolist()
    v_umap_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)].tolist()
    v_tsne_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)].tolist()
    v_lle_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)].tolist()
    v_se_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se)].tolist()
    v_kpca_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se)+len(v_kpca)].tolist()
    v_lda_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se)+len(v_kpca):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se)+len(v_kpca)+len(v_lda)].tolist()
    v_pls_norm = v_data_normalized[len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se)+len(v_kpca)+len(v_lda):len(v_raw)+len(v_iso)+len(v_kiso)+len(v_sskiso)+len(v_umap)+len(v_tsne)+len(v_lle)+len(v_se)+len(v_kpca)+len(v_lda)+len(v_pls)].tolist()    
    
    
    dbs_data = dbs_raw + dbs_iso + dbs_kiso + dbs_sskiso + dbs_umap + dbs_tsne + dbs_lle + dbs_se + dbs_kpca + dbs_lda + dbs_pls
    min1, max1 = np.min(dbs_data), np.max(dbs_data)
    dbs_data_normalized = (dbs_data - min1) / (max1 - min1)
    dbs_raw_norm = dbs_data_normalized[:len(dbs_raw)].tolist()
    dbs_iso_norm = dbs_data_normalized[len(dbs_raw):len(dbs_raw)+len(dbs_iso)].tolist()
    dbs_kiso_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)].tolist()
    dbs_sskiso_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)].tolist()
    dbs_umap_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)].tolist()
    dbs_tsne_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)].tolist()
    dbs_lle_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)].tolist()
    dbs_se_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se)].tolist()
    dbs_kpca_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se)+len(dbs_kpca)].tolist()
    dbs_lda_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se)+len(dbs_kpca):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se)+len(dbs_kpca)+len(dbs_lda)].tolist()
    dbs_pls_norm = dbs_data_normalized[len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se)+len(dbs_kpca)+len(dbs_lda):len(dbs_raw)+len(dbs_iso)+len(dbs_kiso)+len(dbs_sskiso)+len(dbs_umap)+len(dbs_tsne)+len(dbs_lle)+len(dbs_se)+len(dbs_kpca)+len(dbs_lda)+len(dbs_pls)].tolist()    
           

    ss_data = ss_raw + ss_iso + ss_kiso + ss_sskiso + ss_umap + ss_tsne + ss_lle + ss_se + ss_kpca + ss_lda + ss_pls
    min1, max1 = np.min(ss_data), np.max(ss_data)
    ss_data_normalized = (ss_data - min1) / (max1 - min1)
    ss_raw_norm = ss_data_normalized[:len(ss_raw)].tolist()
    ss_iso_norm = ss_data_normalized[len(ss_raw):len(ss_raw)+len(ss_iso)].tolist()
    ss_kiso_norm = ss_data_normalized[len(ss_raw)+len(ss_iso):len(ss_raw)+len(ss_iso)+len(ss_kiso)].tolist()
    ss_sskiso_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)].tolist()
    ss_umap_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)].tolist()
    ss_tsne_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)].tolist()
    ss_lle_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)].tolist()
    ss_se_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se)].tolist()
    ss_kpca_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se)+len(ss_kpca)].tolist()
    ss_lda_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se)+len(ss_kpca):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se)+len(ss_kpca)+len(ss_lda)].tolist()
    ss_pls_norm = ss_data_normalized[len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se)+len(ss_kpca)+len(ss_lda):len(ss_raw)+len(ss_iso)+len(ss_kiso)+len(ss_sskiso)+len(ss_umap)+len(ss_tsne)+len(ss_lle)+len(ss_se)+len(ss_kpca)+len(ss_lda)+len(ss_pls)].tolist()    

    
    results[dataset_name + '_norm'] = { "RAW": [ri_raw_norm, ch_raw_norm, fm_raw_norm, v_raw_norm, dbs_raw_norm, ss_raw_norm],
                                        "ISOMAP": [ri_iso_norm, ch_iso_norm, fm_iso_norm, v_iso_norm, dbs_iso_norm, ss_iso_norm],
                                        "KISOMAP": [ri_kiso_norm, ch_kiso_norm, fm_kiso_norm, v_kiso_norm, dbs_kiso_norm, ss_kiso_norm],
                                        "SSKISOMAP": [ri_sskiso_norm, ch_sskiso_norm, fm_sskiso_norm, v_sskiso_norm, dbs_sskiso_norm, ss_sskiso_norm],
                                        "UMAP": [ri_umap_norm, ch_umap_norm, fm_umap_norm, v_umap_norm, dbs_umap_norm, ss_umap_norm],
                                        "TSNE": [ri_tsne_norm, ch_tsne_norm, fm_tsne_norm, v_tsne_norm, dbs_tsne_norm, ss_tsne_norm],
                                        "LLE": [ri_lle_norm, ch_lle_norm, fm_lle_norm, v_lle_norm, dbs_lle_norm, ss_lle_norm],
                                        "SE": [ri_se_norm, ch_se_norm, fm_se_norm, v_se_norm, dbs_se_norm, ss_se_norm],
                                        "KPCA": [ri_kpca_norm, ch_kpca_norm, fm_kpca_norm, v_kpca_norm, dbs_kpca_norm, ss_kpca_norm],
                                        "LDA": [ri_lda_norm, ch_lda_norm, fm_lda_norm, v_lda_norm, dbs_lda_norm, ss_lda_norm],
                                        "PLS": [ri_pls_norm, ch_pls_norm, fm_pls_norm, v_pls_norm, dbs_pls_norm, ss_pls_norm],
                                    }

    print('Dataset ', dataset_name,' complete')
    print()
    
    # Check previous results
    try:
        with open(file_results, 'r') as f:
            previous_results = json.load(f)
    except FileNotFoundError:
        previous_results = {}
            
    results = {key: {**results.get(key, {}), **previous_results.get(key, {})} for key in results.keys() | previous_results.keys()}

    # Save results
    try:
        with open(file_results, 'w') as f:
            json.dump(results, f)
    except IOError as e:
        print(f"An error occurred while writing to the file: {file_results} - {e}")
    
    
    if plot_results:
        print('*********************************************')
        print('******* SUMMARY OF THE RESULTS **************')
        print('*********************************************')
        print()


        ############## RAW DATA
        print('RAW DATA result')
        print('-----------------')
        L_ = Clustering(raw_data.T, dataset_target, 'RAW DATA', CLUSTER)
        labels_ = L_[6]
        #PlotaDados(raw_data.T, labels_, 'RAW DATA')
        
        
        ############## Regular ISOMAP 
        print('ISOMAP result')
        print('---------------')
        model = Isomap(n_neighbors=nn, n_components=2)
        isomap_data = model.fit_transform(dataset_data)
        isomap_data = isomap_data.T
        L_iso = Clustering(isomap_data, dataset_target, 'ISOMAP', CLUSTER)
        labels_iso = L_iso[6]
        PlotaDados(isomap_data.T, labels_iso, dataset_name + ' ISOMAP')
        
        
        # Find best result in terms of Rand index
        print('Best K-ISOMAP result in terms of Rand index')
        print('----------------------------------------------')
        ri_star = max(enumerate(ri_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, ri_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[6]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + ' K-ISOMAP RI')


        # Find best result in terms of Rand index
        print('Best K-ISOMAP result in terms of Calinski-Harabasz')
        print('-----------------------------------------------------')
        ch_star = max(enumerate(ch_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, ch_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[6]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + ' K-ISOMAP CH')


        # Find best result in terms of Fowlkes Mallows
        print('Best K-ISOMAP result in terms of Fowlkes Mallows')
        print('----------------------------------------------')
        fm_star = max(enumerate(fm_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, fm_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[6]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + dataset_name + ' K-ISOMAP FM')


        # Find best result in terms of V measure
        print('Best K-ISOMAP result in terms of V measure')
        print('-----------------------------------------------------')
        v_star = max(enumerate(v_kiso), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dataset_data, nn, 2, v_star) 
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[6]
        PlotaDados(dados_kiso, labels_kiso, dataset_name + ' K-ISOMAP VS')


        ############## SSK-ISOMAP
        print('SSK-ISOMAP result')
        print('---------------')
        model = SSKIsomap(dataset_data, nn, 2, dataset_target)
        sskiso_data = model.fit_transform(dataset_data)
        sskiso_data = sskiso_data.T
        L_sskiso = Clustering(sskiso_data, dataset_target, 'sskiso', CLUSTER)
        labels_sskiso = L_sskiso[6]
        PlotaDados(sskiso_data.T, labels_sskiso, dataset_name + ' SSK-ISOMAP')
        

        ############## UMAP
        print('UMAP result')
        print('---------------')
        model = UMAP(n_components=2)
        umap_data = model.fit_transform(dataset_data)
        umap_data = umap_data.T
        L_umap = Clustering(umap_data, dataset_target, 'UMAP', CLUSTER)
        labels_umap = L_umap[6]
        PlotaDados(umap_data.T, labels_umap, dataset_name + ' UMAP')
        
        
        ############## T-SNE
        print('T-SNE result')
        print('---------------')
        model = TSNE(n_components=2)
        tsne_data = model.fit_transform(dataset_data)
        tsne_data = tsne_data.T
        L_tsne = Clustering(tsne_data, dataset_target, 'T-SNE', CLUSTER)
        labels_tsne = L_tsne[6]
        PlotaDados(tsne_data.T, labels_tsne, dataset_name + ' T-SNE')
        
        
        ############## LLE LocallyLinearEmbedding
        print('LLE result')
        print('---------------')
        model = LocallyLinearEmbedding(n_components=2)
        lle_data = model.fit_transform(dataset_data)
        lle_data = lle_data.T
        L_lle = Clustering(lle_data, dataset_target, 'LLE', CLUSTER)
        labels_lle = L_lle[6]
        PlotaDados(lle_data.T, labels_lle, dataset_name + ' LLE')


        ############## SpectralEmbedding
        print('SpectralEmbedding')
        print('---------------')
        model = SpectralEmbedding(n_components=2)
        se_data = model.fit_transform(dataset_data)
        se_data = se_data.T
        L_se = Clustering(se_data, dataset_target, 'SE', CLUSTER)
        labels_se = L_se[6]
        PlotaDados(se_data.T, labels_se, dataset_name + ' SE')
        
        
        ############## KernelPCA
        print('KPCA')
        print('---------------')
        model = KernelPCA(n_components=2)
        kpca_data = model.fit_transform(dataset_data)
        kpca_data = kpca_data.T
        L_kpca = Clustering(kpca_data, dataset_target, 'KPCA', CLUSTER)
        labels_se = L_kpca[6]
        PlotaDados(kpca_data.T, labels_se, dataset_name + ' KPCA')
        
        
        ############## LDA
        print('LDA')
        print('---------------')
        if c > 2:
            model = LinearDiscriminantAnalysis(n_components=2)
        else:
            model = LinearDiscriminantAnalysis(n_components=1)
        lda_data = model.fit_transform(dataset_data)
        lda_data = lda_data.T
        L_lda = Clustering(lda_data, dataset_target, 'LDA', CLUSTER)
        labels_se = L_lda[6]
        PlotaDados(lda_data.T, labels_se, dataset_name + ' LDA')


        ############## PLS
        print('PLS')
        print('---------------')
        model = PLSRegression(n_components=2)
        pls_data = model.fit_transform(dataset_data)
        pls_data = pls_data.T
        L_pls = Clustering(pls_data, dataset_target, 'PLS', CLUSTER)
        labels_se = L_pls[6]
        PlotaDados(pls_data.T, labels_se, dataset_name + ' PLS')    