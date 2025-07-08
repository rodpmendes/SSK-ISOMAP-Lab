import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_visualizations_v6(results_file_path: str, output_dir: str) -> None:
    """
    Plots metrics for each method across 4 centrality types.
    X-axis has only 4 positions (one per centrality), Y-values are taken directly from JSON without any averaging.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file_path, 'r') as f:
        results = json.load(f)

    metrics = ['ri', 'ch', 'fm', 'v', 'dbs', 'ss']
    centralities = [
        'SSKISOMAP_betweenness_centrality',
        'SSKISOMAP_degree_centrality',
        'SSKISOMAP_closeness_centrality',
        'SSKISOMAP_eigenvector_centrality'
    ]

    # Method order inside each centrality block
    method_order = [
        'RAW', 'ISOMAP',
        'SSKISOMAP_{}_p10', 'SSKISOMAP_{}_p20', 'SSKISOMAP_{}_p30', 'SSKISOMAP_{}_p40',
        'SSKISOMAP_{}_p50', 'SSKISOMAP_{}_p60', 'SSKISOMAP_{}_p70', 'SSKISOMAP_{}_p80',
        'SSKISOMAP_{}_p90', 'SSKISOMAP_{}_p100',
        'KISOMAP', 'UMAP', 'TSNE', 'LLE', 'SE', 'KPCA', 'LDA', 'PLS'
    ]

    # Build a list of all unique methods (expanding the placeholders for each centrality)
    all_methods = []
    for method in method_order:
        if '{}' in method:
            for centrality in centralities:
                all_methods.append(method.format(centrality.split('_')[-1]))
        else:
            all_methods.append(method)
    all_methods = list(set(all_methods))  # Remove duplicates if any

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        x_positions = list(range(len(centralities)))
        x_labels = centralities

        # For each method, collect Y-values across the 4 centralities
        for method in method_order:
            y_values = []

            for idx, centrality in enumerate(centralities):
                # Map the method for this centrality
                if '{}' in method:
                    current_method = method.format(centrality.split('_')[-1])
                else:
                    current_method = method

                # Search the value across all datasets and take the first found (no averaging)
                value_found = False
                for dataset in results:
                    if not dataset.endswith('_norm'):
                        if current_method in results[dataset]:
                            val_list = results[dataset][current_method].get(metric, [np.nan])
                            y_values.append(val_list[0] if len(val_list) > 0 else np.nan)
                            value_found = True
                            break
                if not value_found:
                    y_values.append(np.nan)

            color = 'blue' if method == 'KISOMAP' else ('orange' if 'SSKISOMAP' in method else 'lightgray')
            zorder = 5 if method == 'KISOMAP' else (2 if 'SSKISOMAP' in method else 1)
            linewidth = 2  # Mesma espessura pra todas, incluindo KISOMAP

            plt.plot(
                x_positions,
                y_values,
                marker='o',
                label=method,
                color=color,
                linewidth=linewidth,
                zorder=zorder
            )

        plt.title(f"{metric.upper()} Comparison Across Centralities", fontsize=14)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.xlabel("Centrality Type", fontsize=12)
        plt.xticks(x_positions, x_labels, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.legend(
            fontsize=8,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=True
        )

        output_path = os.path.join(output_dir, f"v6_{metric}_centrality_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {output_path}")

if __name__ == "__main__":
    generate_visualizations_v6(
        results_file_path='results/dataset_results_ssk_variation.json',
        output_dir='results/img/v6_plots'
    )
