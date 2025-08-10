import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_and_save_graph(B: np.ndarray, output_dir: str = "./graph_utils/graph_img", filename: str = "graph.png") -> None:
    """Visualizes and saves a weighted graph based on the adjacency matrix B.

    This function builds a weighted graph from a NumPy array, applies a spring layout,
    and saves the resulting graph visualization as a high-resolution PNG image.

    Args:
        B (np.ndarray): Weighted adjacency matrix (2D square array).
        output_dir (str): Directory where the image will be saved. Defaults to "grafo_imagens".
        filename (str): Name of the output PNG file. Defaults to "graph.png".

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create an undirected weighted graph from the matrix
    G = nx.from_numpy_array(B)

    # Compute node positions using the spring layout for better visualization
    pos = nx.spring_layout(G, seed=42)

    # Extract all edge weights
    all_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(all_weights) if all_weights else 1

    # Normalize edge weights for drawing
    norm_weights = [w / max_weight * 3 for w in all_weights]

    # Start drawing
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, width=norm_weights, edge_color="gray", alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Save the figure with 300 DPI
    output_path = os.path.join(output_dir, filename)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Graph saved at: {output_path}")


def visualize_and_save_graph_colors(B: np.ndarray,
                             self_labels: np.ndarray,
                             output_dir: str = "./graph_utils/graph_img",
                             filename: str = "graph.png",
                             edge_alpha: float = 0.5) -> None:
    """Visualizes and saves a weighted graph with nodes colored by cluster labels,
    and edges colored with corresponding node colors with transparency.

    Args:
        B (np.ndarray): Weighted adjacency matrix (2D square array).
        self_labels (np.ndarray): Array of cluster labels corresponding to each node.
        output_dir (str): Directory to save the output image. Defaults to "grafo_imagens".
        filename (str): Filename for the saved image. Defaults to "graph.png".
        edge_alpha (float): Transparency for edges (0.0 to 1.0). Defaults to 0.3.

    Returns:
        None
    """
    import os
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    os.makedirs(output_dir, exist_ok=True)
    G = nx.from_numpy_array(B)
    pos = nx.spring_layout(G, seed=42)

    all_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(all_weights) if all_weights else 1
    edge_widths = [w / max_weight * 3 for w in all_weights]

    unique_labels = np.unique(self_labels)
    cmap = plt.get_cmap('tab10')
    label_to_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}
    node_colors = [label_to_color[label] for label in self_labels]

    # Prepara cores RGBA para nós para usar nas arestas com transparência
    node_colors_rgba = [mcolors.to_rgba(color, alpha=1.0) for color in node_colors]

    edge_colors = []
    for u, v in G.edges():
        # Cor média dos dois nós para a aresta
        color_u = node_colors_rgba[u]
        color_v = node_colors_rgba[v]
        avg_color = tuple((np.array(color_u) + np.array(color_v)) / 2)
        # Aplica transparência alpha desejada
        edge_colors.append((avg_color[0], avg_color[1], avg_color[2], edge_alpha))

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Graph saved at: {output_path}")