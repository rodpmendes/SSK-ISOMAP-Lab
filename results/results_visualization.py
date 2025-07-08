import json
import matplotlib.pyplot as plt
import numpy as np

files_results = ['dataset_results.json']

file = files_results[0]
 
with open(file, 'r') as f:
    results = json.load(f)
       

#################################################
# Plot results

# Noise parameters
# Standard deviation (spread or “width”) of the distribution. Must be non-negative
magnitude = 1 # normalized data base 

# Define magnitude
magnitude = np.linspace(0, magnitude, 1)


datasets = [name for name in results.keys() if not name.endswith('_norm')]
datasets_norm = [name for name in results.keys() if name.endswith('_norm')]
dr_methods = list(results[datasets[0]].keys())
metrics = ['Rand Index', 'Calinski-Harabasz Score', 'Fowlkes-Mallow Index', 'V measure', 'Davies Bouldin Score', 'Silhouette Score']


datasets = datasets_norm

cols = 6
rows = 3
pages = 1 if len(datasets)//5 == 0 else len(datasets)//5
idx_db = 0


for p in range(1, pages+1):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
    
    for i, dataset in enumerate(datasets[idx_db:p*rows]):
        if i < rows:
            for j, metric in enumerate(metrics):
                ax = axs[i, j]  
                for method in dr_methods:
                    ax.plot(magnitude, results[dataset][method][j], label=method)
                    
                if j == 0:
                    ax.set_ylabel(dataset.replace('_norm', ''))  # Set the y label here
                    plt.setp(ax.get_yticklabels(), visible=True)
                else:
                    plt.setp(ax.get_yticklabels(), visible=True) 
                if i == 0:
                    ax.set_title(metric)  
                if i == 0 and j==2:
                    ax.legend()
    
    idx_db += rows
    plt.savefig('first_dataset_results_page_' + str(p) +'.jpeg', dpi=300,format='jpeg')
    plt.show()
    
    a = 'stop'