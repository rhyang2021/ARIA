import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Load data
print("Loading data...")
results = []
with open('action_list_with_id.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

# Extract data
state_embeddings, action_embeddings, states, actions, state_ids, game_action_ids, game_ids = [], [], [], [], [], [], []
for result in results:
    action_embeddings.append(result["action_embedding"])
    actions.append(result["action"])
    game_action_ids.append(result["game_action_id"])

# Print data length
print(f"Number of action embedding vectors: {len(action_embeddings)}")

# Create DataFrame
result_dict = {"action": actions, "action_embedding": action_embeddings, "game_action_id": game_action_ids}
df = pd.DataFrame(result_dict)

# Convert to numpy arrays
action_vectors = np.array(action_embeddings)
actions = np.array(actions)
action_ids = np.array(game_action_ids)

# Create linkage matrix for hierarchical clustering
print("Computing hierarchical clustering linkage matrix...")
linkage_matrix = linkage(action_vectors, method='ward')

# Set output directory
base_output_dir = "./bargaining"
os.makedirs(base_output_dir, exist_ok=True)

def save_clusters_flat(cluster_data, output_dir):
    """
    Save each cluster's data flat to directory, one JSONL file per cluster.
    """
    for cluster_id, items in cluster_data.items():
        file_path = os.path.join(output_dir, f"Cluster_{cluster_id}.jsonl")
        
        with open(file_path, "w") as f:
            for item in items:
                if isinstance(item["action"], str):
                    try:
                        item["action"] = json.loads(item["action"])
                    except json.JSONDecodeError:
                        pass
                f.write(json.dumps(item) + "\n")
    print(f"Cluster data saved to: {output_dir}")

# Perform clustering for each value from k=2 to k=100
for k in range(2, 101):
    print(f"Performing clustering for k={k}...")
    
    # Create output directory for current k value
    output_dir = os.path.join(base_output_dir, f"k{k}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
    
    # Storage for cluster data
    cluster_data = {i: [] for i in range(1, k + 1)}
    
    # Aggregate data within clusters
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_data[cluster_id].append({
            "action_id": action_ids[idx],
            "action": actions[idx],
        })
    
    # Save cluster data
    save_clusters_flat(cluster_data, output_dir)
    
    # Removed dendrogram generation part here

print("All clustering tasks completed!")

# Create a summary table showing cluster size distribution for each k value
cluster_sizes_summary = []

for k in range(2, 101):
    cluster_sizes = []
    output_dir = os.path.join(base_output_dir, f"k{k}")
    
    for i in range(1, k + 1):
        file_path = os.path.join(output_dir, f"Cluster_{i}.jsonl")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                count = sum(1 for _ in f)
                cluster_sizes.append(count)
    
    cluster_sizes_summary.append({
        'k': k,
        'cluster_sizes': cluster_sizes,
        'min_size': min(cluster_sizes),
        'max_size': max(cluster_sizes),
        'avg_size': sum(cluster_sizes) / len(cluster_sizes)
    })

# Save summary information
summary_path = os.path.join(base_output_dir, "clustering_summary.json")
with open(summary_path, 'w') as f:
    json.dump(cluster_sizes_summary, f, indent=2)

print(f"Clustering summary information saved to: {summary_path}")