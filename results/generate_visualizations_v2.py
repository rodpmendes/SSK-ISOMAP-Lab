"""
High-resolution visualization for experimental results from dataset_results_one.json.

Generates one figure per metric (ri, ch, fm, v, dbs, ss), for both raw and normalized data.
Colors:
- SSKISOMAP: Orange (on top)
- KISOMAP: Blue (above others)
- Others: Light Gray (bottom layer)
"""

import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt


def load_results(file_path: str) -> Dict:
    """Load experimental results from a JSON file.

    Args:
        file_path (str): Path to the results JSON.

    Returns:
        Dict: Nested dictionary with experiment results.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_metric(results: Dict, metric_name: str, normalized: bool) -> Dict[str, Dict[str, float]]:
    """Extract metric values for each method and dataset.

    Args:
        results (Dict): Experimental results dictionary.
        metric_name (str): Metric key (e.g., 'ri', 'ch', etc.).
        normalized (bool): Whether to extract from normalized datasets.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary of dataset → method → metric value.
    """
    extracted = {}
    for dataset_key in results:
        is_norm = '_norm' in dataset_key
        if is_norm != normalized:
            continue

        dataset_name = dataset_key.replace('_norm', '')
        extracted[dataset_name] = {}

        for method, metrics_dict in results[dataset_key].items():
            if metric_name in metrics_dict:
                extracted[dataset_name][method] = metrics_dict[metric_name][0]
    return extracted


def plot_single_metric(metric_name: str, results: Dict, normalized: bool, output_dir: str, base_filename: str) -> None:
    """Plot a single metric across datasets and methods.

    Args:
        metric_name (str): Name of the metric (e.g., 'ri').
        results (Dict): Experimental results.
        normalized (bool): Whether to plot normalized or raw data.
        output_dir (str): Directory to save plots.
        base_filename (str): Base filename for saving.
    """
    data = extract_metric(results, metric_name, normalized)
    datasets = sorted(data.keys())
    methods = ['RAW', 'ISOMAP', 'KISOMAP', 'SSKISOMAP', 'UMAP', 'TSNE', 'LLE', 'SE', 'KPCA', 'LDA', 'PLS']

    plt.figure(figsize=(12, 6))

    for method in methods:
        y_vals = []
        for dataset in datasets:
            value = data[dataset].get(method, None)
            y_vals.append(value if value is not None else float('nan'))

        if method == 'SSKISOMAP':
            color = 'orange'
            z = 3
            lw = 2.5
        elif method == 'KISOMAP':
            color = 'blue'
            z = 2
            lw = 2.5
        else:
            color = 'lightgray'
            z = 1
            lw = 1

        plt.plot(
            datasets,
            y_vals,
            marker='o',
            label=method if method in ['KISOMAP', 'SSKISOMAP'] else '_nolegend_',
            color=color,
            zorder=z,
            linewidth=lw
        )

    plt.title(f"{metric_name.upper()} - {'Normalized' if normalized else 'Raw'}", fontsize=14)
    plt.xlabel("Datasets", fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.legend(["KISOMAP", "SSKISOMAP"], loc='best')
    plt.tight_layout()

    filename = f"{base_filename}_{'norm' if normalized else 'raw'}_{metric_name}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to generate high-resolution visualizations."""
    results_file = os.path.join('results', 'dataset_results_one.json')
    output_dir = os.path.join('results', 'img')
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    metrics = ['ri', 'ch', 'fm', 'v', 'dbs', 'ss']
    base_filename = os.path.splitext(os.path.basename(results_file))[0]

    for metric_name in metrics:
        plot_single_metric(metric_name, results, normalized=False, output_dir=output_dir, base_filename=base_filename)
        plot_single_metric(metric_name, results, normalized=True, output_dir=output_dir, base_filename=base_filename)


if __name__ == "__main__":
    main()
