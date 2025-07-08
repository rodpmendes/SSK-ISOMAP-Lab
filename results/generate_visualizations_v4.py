"""
High-resolution visualization for all metrics combined in one figure.
Handles multiple SSKISOMAP variations (by centrality and proportion).

Colors:
- SSKISOMAP variants: Orange
- KISOMAP: Blue
- Others: Light Gray
"""

import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np


def load_results(file_path: str) -> Dict:
    """Load experimental results from a JSON file.

    Args:
        file_path (str): Path to the results JSON.

    Returns:
        Dict: Nested dictionary with experiment results.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_all_metrics(results: Dict, normalized: bool, metrics: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Extract all metric values for each dataset and method.

    Args:
        results (Dict): Experimental results.
        normalized (bool): Whether to extract from normalized datasets.
        metrics (List[str]): List of metric names.

    Returns:
        Dict[str, Dict[str, List[float]]]: dataset → method → list of metrics.
    """
    extracted = {}
    for dataset_key in results:
        is_norm = '_norm' in dataset_key
        if is_norm != normalized:
            continue

        dataset_name = dataset_key.replace('_norm', '')
        extracted[dataset_name] = {}

        for method, metric_values in results[dataset_key].items():
            extracted[dataset_name][method] = []
            for metric in metrics:
                if metric in metric_values and len(metric_values[metric]) > 0:
                    extracted[dataset_name][method].append(metric_values[metric][0])
                else:
                    extracted[dataset_name][method].append(np.nan)
    return extracted


def plot_all_metrics(results: Dict, normalized: bool, output_dir: str, base_filename: str) -> None:
    """Plot all metrics (6 subplots) in one figure.

    Args:
        results (Dict): Experimental results.
        normalized (bool): Whether to plot normalized or raw data.
        output_dir (str): Directory to save plots.
        base_filename (str): Base filename for saving.
    """
    metrics = ['ri', 'ch', 'fm', 'v', 'dbs', 'ss']
    data = extract_all_metrics(results, normalized, metrics)
    datasets = sorted(data.keys())

    # Dynamically get method list based on first dataset found
    example_dataset = next(iter(data.values()))
    methods = list(example_dataset.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for method in methods:
            y_vals = []
            for dataset in datasets:
                method_metrics = data[dataset].get(method, [np.nan] * len(metrics))
                y_vals.append(method_metrics[idx])

            # Coloring rules
            if method.startswith('SSKISOMAP'):
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

            line, = ax.plot(
                datasets,
                y_vals,
                marker='o',
                label=method if method in ['KISOMAP'] or method.startswith('SSKISOMAP') else '_nolegend_',
                color=color,
                zorder=z,
                linewidth=lw
            )

            # Legend only for KISOMAP and SSKISOMAP variants (once each)
            if (method == 'KISOMAP' or method.startswith('SSKISOMAP')) and method not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(method)

        ax.set_title(f"{metric.upper()} - {'Normalized' if normalized else 'Raw'}", fontsize=12)
        ax.set_xlabel("Datasets", fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Custom legend
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"{base_filename}_{'norm' if normalized else 'raw'}_all_metrics.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to generate combined high-resolution visualizations."""
    results_file = os.path.join('results', 'dataset_results_ssk_variation.json')
    output_dir = os.path.join('results', 'img')
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    base_filename = os.path.splitext(os.path.basename(results_file))[0]

    plot_all_metrics(results, normalized=False, output_dir=output_dir, base_filename=base_filename)
    plot_all_metrics(results, normalized=True, output_dir=output_dir, base_filename=base_filename)


if __name__ == "__main__":
    main()
