import numpy as np
import logging
from typing import List, Dict
from scipy.cluster.hierarchy import linkage, fcluster

from utils.helpers import (
    compute_layerwise_distances,
    _find_min_weighted_directed_tree,
    compute_kurtosis,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def hierarchical_clustering(
    dist_matrix: np.ndarray, num_clusters: int = 2
) -> np.ndarray:
    """
    Performs hierarchical clustering on the distance matrix.

    Args:
        dist_matrix (np.ndarray): The pairwise distance matrix.
        num_clusters (int): The number of clusters to form.

    Returns:
        np.ndarray: An array of cluster labels for each element.
    """
    logging.info("Performing hierarchical clustering...")
    Z = linkage(dist_matrix, method="ward")
    clusters = fcluster(Z, num_clusters, criterion="maxclust")

    logging.info(f"Cluster assignments: {clusters}")
    return clusters


def build_tree_per_cluster(
    d_c: np.ndarray, k_c: np.ndarray, l: float = 0.4
) -> np.ndarray:
    """
    Builds a tree for each cluster based on distances and kurtosis.

    Args:
        d_c (np.ndarray): Distance matrix for the cluster.
        k_c (np.ndarray): Kurtosis values for the models in the cluster.
        l (float): Lambda parameter to weight the kurtosis-based term.

    Returns:
        np.ndarray: The adjacency matrix for the cluster.
    """
    logging.info("Building tree for cluster...")
    k_c = np.array(k_c)  # Ensure kurtosis values are in NumPy format
    T = (k_c[:, np.newaxis] > k_c[np.newaxis, :]).astype(
        int
    )  # Comparison matrix for kurtosis

    inf_values = np.full((T.shape[0],), float("inf"))
    diag_inf = np.diag(inf_values)

    d_mean = np.mean(d_c)
    M = d_c + l * d_mean * T + diag_inf

    logging.info(f"Tree construction matrix M:\n{M}")
    A_c = _find_min_weighted_directed_tree(M)
    return A_c


def build_global_tree(
    models: List[Dict[str, np.ndarray]],
    clusters: np.ndarray,
    layer_keys: List[str],
    kurtosis_values: np.ndarray,
    l: float = 0.4,
) -> np.ndarray:
    """
    Constructs a global tree by building minimal trees for each cluster and combining them.
    Handles clusters with only one node by skipping the tree-building step.

    Args:
        models (List[Dict[str, np.ndarray]]): A list of state dictionaries.
        clusters (np.ndarray): Cluster assignments for each model.
        layer_keys (List[str]): The keys of the layers to consider for distance computation.
        kurtosis_values (np.ndarray): The kurtosis values computed for each model.
        l (float): Lambda parameter to weight the kurtosis-based term.

    Returns:
        np.ndarray: The global adjacency matrix.
    """
    layerwise_distances = compute_layerwise_distances(models, layer_keys)
    clunq = np.unique(clusters)

    M = np.zeros((len(models), len(models)))

    for cdx in clunq:
        c_idx = np.where(clusters == cdx)[0]

        # Handle clusters with only one node
        if len(c_idx) == 1:
            logging.info(f"Cluster {cdx} has only one node. Skipping tree building.")
            continue

        d_c = layerwise_distances[c_idx][:, c_idx].squeeze()
        k_c = kurtosis_values[c_idx]

        M_c = build_tree_per_cluster(d_c, k_c, l)
        M[np.ix_(c_idx, c_idx)] = M_c

    logging.info(f"Global adjacency matrix:\n{M}")
    return M
