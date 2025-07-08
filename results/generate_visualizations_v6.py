import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


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


def plot_single_metric(metric_name: str, results: Dict, normalized: bool, output_dir: str, base_filename: str) -> None:
    data = extract_metric(results, metric_name, normalized)
    datasets = sorted(data.keys())

    # Detect all method names dynamically from data
    all_methods = set()
    for dataset in data:
        all_methods.update(data[dataset].keys())
    all_methods = sorted(all_methods)

    # Split methods: SSKISOMAP variations, KISOMAP, others
    ssk_methods = [m for m in all_methods if m.startswith('SSKISOMAP')]
    color_map = cm.get_cmap('tab10', len(ssk_methods))

    plt.figure(figsize=(12, 6))

    # Plot non-SSK methods first (light gray)
    for method in all_methods:
        if method.startswith('SSKISOMAP') or method == 'KISOMAP':
            continue
        y_vals = [data[ds].get(method, np.nan) for ds in datasets]
        plt.plot(
            datasets,
            y_vals,
            marker='o',
            color='lightgray',
            linewidth=1,
            label='_nolegend_',
            zorder=1
        )

    # Plot KISOMAP (blue)
    if 'KISOMAP' in all_methods:
        y_vals = [data[ds].get('KISOMAP', np.nan) for ds in datasets]
        plt.plot(
            datasets,
            y_vals,
            marker='o',
            color='blue',
            linewidth=2.5,
            label='KISOMAP',
            zorder=2
        )

    # Plot all SSKISOMAP variations with distinct colors
    for idx, method in enumerate(ssk_methods):
        y_vals = [data[ds].get(method, np.nan) for ds in datasets]
        plt.plot(
            datasets,
            y_vals,
            marker='o',
            color=color_map(idx),
            linewidth=2.5,
            label=method,
            zorder=3 + idx
        )

    plt.title(f"{metric_name.upper()} - {'Normalized' if normalized else 'Raw'}", fontsize=14)
    plt.xlabel("Datasets", fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    # Legenda fora da área do gráfico (lado direito)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    filename = f"{base_filename}_{'norm' if normalized else 'raw'}_{metric_name}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    results_file = os.path.join('results', 'dataset_results_ssk_variation.json')
    output_dir = os.path.join('results', 'img/plot_v6')
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    metrics = ['ri', 'ch', 'fm', 'v', 'dbs', 'ss']
    base_filename = os.path.splitext(os.path.basename(results_file))[0]

    for metric_name in metrics:
        plot_single_metric(metric_name, results, normalized=False, output_dir=output_dir, base_filename=base_filename)
        plot_single_metric(metric_name, results, normalized=True, output_dir=output_dir, base_filename=base_filename)


if __name__ == "__main__":
    main()
