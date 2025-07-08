import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# -------- CONFIG --------
#CENTRALITY_TYPE = 'betweenness_centrality'  # Opções: 'betweenness_centrality', 'degree_centrality', 'closeness_centrality', 'eigenvector_centrality'
#PROPORTIONS = ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100']

CENTRALITY_TYPE = 'betweenness_centrality'
PROPORTIONS = ['p10','p20', 'p30', 'p100']
# ------------------------


def load_results(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_metric(results: Dict, metric_name: str, normalized: bool) -> Dict[str, Dict[str, float]]:
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


def select_methods(all_methods: List[str]) -> (List[str], List[str], List[str]):
    reference_methods = [m for m in all_methods if m not in ['KISOMAP'] and not m.startswith('SSKISOMAP')]
    selected_ssk = []
    for method in all_methods:
        if CENTRALITY_TYPE in method:
            if any(prop in method for prop in PROPORTIONS):
                selected_ssk.append(method)
    kisomap = ['KISOMAP'] if 'KISOMAP' in all_methods else []
    return reference_methods, kisomap, selected_ssk


def plot_all_metrics(results: Dict, normalized: bool, output_dir: str, base_filename: str) -> None:
    metrics = ['ri', 'ch', 'fm', 'v', 'dbs', 'ss']
    num_metrics = len(metrics)

    # Detect all methods
    sample_data = extract_metric(results, metrics[0], normalized)
    all_methods = set()
    for dataset in sample_data:
        all_methods.update(sample_data[dataset].keys())
    all_methods = sorted(all_methods)

    reference_methods, kisomap_methods, selected_ssk_methods = select_methods(all_methods)
    total_methods = reference_methods + kisomap_methods + selected_ssk_methods
    color_map = cm.get_cmap('tab10', len(selected_ssk_methods))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics):
        data = extract_metric(results, metric_name, normalized)
        datasets = sorted(data.keys())

        ax = axes[idx]

        # Plot métodos de referência (cinza)
        for method in reference_methods:
            y_vals = [data[ds].get(method, np.nan) for ds in datasets]
            ax.plot(
                datasets,
                y_vals,
                marker='o',
                color='lightgray',
                linewidth=1,
                label=method
            )

        # Plot KISOMAP (azul)
        for method in kisomap_methods:
            y_vals = [data[ds].get(method, np.nan) for ds in datasets]
            ax.plot(
                datasets,
                y_vals,
                marker='o',
                color='blue',
                linewidth=2.5,
                label=method
            )

        # Plot SSKISOMAPs selecionados (coloridos)
        for ssk_idx, method in enumerate(selected_ssk_methods):
            y_vals = [data[ds].get(method, np.nan) for ds in datasets]
            ax.plot(
                datasets,
                y_vals,
                marker='o',
                color=color_map(ssk_idx),
                linewidth=2.5,
                label=method
            )

        ax.set_title(metric_name.upper(), fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Legenda única fora da área dos subplots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # espaço para legenda

    filename = f"{base_filename}_{CENTRALITY_TYPE}_{'norm' if normalized else 'raw'}_all_metrics.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    results_file = os.path.join('results', 'dataset_results_ssk_variation_controled_scaling.json')
    output_dir = os.path.join('results', 'img/v8_plot_controled_scaling')
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    base_filename = os.path.splitext(os.path.basename(results_file))[0]

    plot_all_metrics(results, normalized=False, output_dir=output_dir, base_filename=base_filename)
    plot_all_metrics(results, normalized=True, output_dir=output_dir, base_filename=base_filename)


if __name__ == "__main__":
    main()
