from models.model_operations import (
    load_model,
    get_state_dict,
    add_noise_to_state_dict,
    are_dicts_close,
)
from utils.clustering import (
    compute_layerwise_distances,
    hierarchical_clustering,
    build_global_tree,
    compute_kurtosis,
)
import torch
import numpy as np

# Load model and obtain state dictionary
model = load_model()
root_a = get_state_dict(model)

# Add noise to create variations
root_b = add_noise_to_state_dict(root_a, noise_level=0.5)
key = "encoder.layers.encoder_layer_8.ln_1.weight"
print(
    f"Are root_a and root_b close for key {key}? ", are_dicts_close(root_a, root_b, key)
)

# Further state dictionaries with noise
a_1 = add_noise_to_state_dict(root_a, noise_level=0.1)
a_11 = add_noise_to_state_dict(a_1, noise_level=0.1)
a_12 = add_noise_to_state_dict(a_1, noise_level=0.1)
a_13 = add_noise_to_state_dict(a_1, noise_level=0.1)
b_1 = add_noise_to_state_dict(root_b, noise_level=0.1)
b_11 = add_noise_to_state_dict(b_1, noise_level=0.1)
b_12 = add_noise_to_state_dict(b_1, noise_level=0.1)
b_13 = add_noise_to_state_dict(b_1, noise_level=0.1)

# List of models
models = [root_a, a_1, a_11, a_12, a_13, root_b, b_1, b_11, b_12, b_13]

# Ground truth adjacency matrix
A_true = np.zeros((len(models), len(models)))
A_true[0, 1] = 1  # root_a -> a_1
A_true[1, 2:5] = 1  # a_1 -> a_1x
A_true[5, 6] = 1  # root_b -> b_1
A_true[6, 7:10] = 1  # b_1 -> b_1x
print("Ground Truth Adjacency Matrix:\n", A_true)

# Layer keys to consider
layer_keys = [
    "encoder.layers.encoder_layer_0.self_attention.in_proj_weight",
    "encoder.layers.encoder_layer_0.mlp.0.weight",
    "encoder.layers.encoder_layer_11.mlp.3.weight",
]

# Compute kurtosis values for models
kurtosis_values = compute_kurtosis(models, layer_keys)
print("Kurtosis Values:\n", kurtosis_values)

# Perform hierarchical clustering to get cluster labels
layerwise_distances = compute_layerwise_distances(models, layer_keys)
num_clusters = 3
clusters = hierarchical_clustering(layerwise_distances, num_clusters)

# Build the global tree and obtain the final adjacency matrix
M = build_global_tree(models, clusters, layer_keys, kurtosis_values, l=0.4)
print("Predicted Adjacency Matrix M:\n", M)

# Compare with the ground truth
gt_flat = torch.tensor(A_true.flatten())  # Convert NumPy array to PyTorch tensor
pred_flat = torch.tensor(M.flatten())  # Convert NumPy array to PyTorch tensor

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP = torch.sum((gt_flat == 1) & (pred_flat == 1)).item()
FP = torch.sum((gt_flat == 0) & (pred_flat == 1)).item()
FN = torch.sum((gt_flat == 1) & (pred_flat == 0)).item()
TN = torch.sum((gt_flat == 0) & (pred_flat == 0)).item()

print(f"TP: {TP} - FP: {FP} - FN: {FN} - TN: {TN}")

# Calculate Accuracy, Precision, and Recall
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
