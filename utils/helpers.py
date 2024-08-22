import torch
import numpy as np
import networkx as nx
import logging
from scipy.stats import kurtosis
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_kurtosis(
    models: List[Dict[str, np.ndarray]], layer_keys: List[str]
) -> np.ndarray:
    """
    Computes the mean kurtosis of selected layers for each model.

    Args:
        models (List[Dict[str, np.ndarray]]): A list of state dictionaries.
        layer_keys (List[str]): The keys of the layers to consider for kurtosis computation.

    Returns:
        np.ndarray: An array of kurtosis values for each model.
    """
    logging.info("Computing kurtosis for selected layers...")
    kurtosis_values = []
    for model in models:
        kurts = []
        for layer_key in layer_keys:
            weights = model[layer_key].numpy()
            kurts.append(kurtosis(weights.flatten()))
        kurtosis_values.append(np.mean(kurts))

    kurtosis_array = np.array(kurtosis_values)
    logging.info(f"Kurtosis values: {kurtosis_array}")
    return kurtosis_array


def compute_layerwise_distances(
    models: List[Dict[str, np.ndarray]], layer_keys: List[str]
) -> np.ndarray:
    """
    Computes the pairwise Euclidean distances between models for selected layers, normalizes these distances,
    and then averages them.

    Args:
        models (List[Dict[str, np.ndarray]]): A list of state dictionaries.
        layer_keys (List[str]): The keys of the layers to consider for distance computation.

    Returns:
        np.ndarray: A distance matrix containing the normalized and averaged pairwise distances.
    """
    logging.info("Computing layerwise distances...")
    num_models = len(models)
    num_layers = len(layer_keys)

    # Initialize the full distance tensor
    c_full = np.zeros((num_models, num_models, num_layers))

    # Compute pairwise distances for each layer
    for idx in range(num_models):
        for jdx in range(idx + 1, num_models):
            for kdx, lkx in enumerate(layer_keys):
                # Get weight pairs
                w1 = models[idx][lkx].flatten()
                w2 = models[jdx][lkx].flatten()
                # Compute Euclidean distance
                dtmp = np.linalg.norm(w1 - w2, ord=2)
                # Populate c_full with the distances
                c_full[idx, jdx, kdx] = dtmp
                c_full[jdx, idx, kdx] = dtmp

    # Normalize the distances
    cmins = c_full.min(axis=(0, 1))
    cmaxs = c_full.max(axis=(0, 1))
    c_normalized = (c_full - cmins) / (cmaxs - cmins)

    # Compute the mean across the selected layers
    c_mean = np.mean(c_normalized, axis=2)

    logging.info(f"Layerwise distance matrix:\n{c_mean}")
    return c_mean


def _find_min_weighted_directed_tree(D: np.ndarray) -> np.ndarray:
    """
    Finds the minimum weighted directed tree in a graph and returns its adjacency matrix.

    Args:
        D (np.ndarray): The distance matrix representing the graph.

    Returns:
        np.ndarray: The adjacency matrix of the minimum weighted directed tree.
    """
    G = nx.DiGraph()
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i, j] < float("inf"):
                G.add_edge(i, j, weight=D[i, j])

    min_arborescence = nx.algorithms.tree.branchings.Edmonds(G).find_optimum(
        attr="weight",
        default=float("inf"),
        kind="min",
        style="arborescence",
        preserve_attrs=False,
    )

    adjacency_matrix = nx.to_numpy_array(min_arborescence, weight=None)
    return adjacency_matrix
