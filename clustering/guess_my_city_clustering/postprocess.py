import json
import numpy as np
import pandas as pd
import os
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
from collections import defaultdict

# Part 2: Merge all clustering results into a unified JSONL file
def merge_clusters():
    print("Starting to merge clustering results...")
    base_dir = "guess_my_city"
    base_path = Path(base_dir)
    output_path = base_path / "merged_clusters.jsonl"
    
    # Create dictionary to store all data
    all_actions = defaultdict(dict)
    
    # Iterate through all directories from k2 to k100
    for k in range(2, 101):
        cluster_dir = base_path / f"k{k}"
        if not cluster_dir.exists():
            print(f"Warning: Directory {cluster_dir} does not exist, skipping")
            continue
        
        print(f"Processing clustering for k={k}...")
        
        # Iterate through all Cluster_N.jsonl files in current k directory
        for cluster_file in cluster_dir.glob("Cluster_*.jsonl"):
            cluster_number = cluster_file.stem.split("_")[1]  # Get cluster number
            
            # Read each line in the cluster file
            with open(cluster_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        action_id = data.get('action_id')
                        
                        if action_id:
                            # If this is the first time encountering this action_id, save its action
                            if 'action' not in all_actions[action_id]:
                                all_actions[action_id]['action'] = data.get('action', '')
                            
                            # Add cluster label for current k value
                            all_actions[action_id][f'k{k}_label'] = cluster_number
                    except json.JSONDecodeError:
                        print(f"Warning: Unable to parse line: {line.strip()} in file {cluster_file}")
    
    # Write all merged data to output file
    print(f"Writing merged data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for action_id, data in all_actions.items():
            # Replace 'b' at the beginning of action_id with 'n'
            if action_id.startswith('b'):
                new_action_id = 'n' + action_id[1:]
            else:
                new_action_id = action_id
            
            # Create a new dictionary ensuring action_id is at the front
            ordered_data = {"action_id": new_action_id}
            # Add other fields
            ordered_data.update(data)
            # Remove potentially duplicate action_id
            if "action_id" in ordered_data:
                del ordered_data["action_id"]
            # Add action_id to the front again
            final_data = {"action_id": new_action_id}
            final_data.update(ordered_data)
            
            f.write(json.dumps(final_data, ensure_ascii=False) + '\n')
    
    print(f"Complete! Merged data saved to {output_path}")
    print(f"Total processed {len(all_actions)} different actions")

# Part 3: Update original file to add action labels
def add_action_labels():
    print("Starting to add action labels...")
    # File paths
    merge_file = "guess_my_city/merged_clusters.jsonl"  # Merged data
    origin_file = "llama3-8b_guess_my_city_embedding_msgs_game.jsonl"  # Original file
    updated_file = "llama3-8b_guess_my_city_embedding_msgs_game_with_action_labels.jsonl"  # Updated file

    # Load merge_clusters.jsonl data into memory
    # Create a dictionary for each k value
    merge_data = {}
    for k in range(2, 101):
        merge_data[k] = {}

    with open(merge_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            action_id = entry["action_id"]
            
            # Get labels for each k value
            for k in range(2, 101):
                k_label_key = f"k{k}_label"
                if k_label_key in entry:
                    # If no game_id provided, use only action_id as key
                    merge_data[k][action_id] = entry[k_label_key]

    # Open original file, process line by line and update
    with open(origin_file, 'r') as infile, open(updated_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            # Get game_id
            game_id = data["game_id"]
            
            # Create action_labels for each k value
            action_labels_by_k = {}
            for k in range(2, 101):
                action_labels_by_k[k] = [
                    merge_data[k].get(action_id, None)  # Look up using only action_id
                    for action_id in data["action_ids"]
                ]

            # Reconstruct JSON data
            new_data = {}

            # First insert all key-value pairs from original data
            for key, value in data.items():
                if key == "action_embeddings":
                    # Insert all action_k{num}_label at this position
                    for k in range(2, 101):
                        new_data[f"action_k{k}_label"] = action_labels_by_k[k]
                
                new_data[key] = value  # Insert original key-value pair

            # Write updated JSON data to new file
            outfile.write(json.dumps(new_data) + '\n')

    print(f"Action labels added, saved as {updated_file}")

# Part 4: Update file with action labels to add observation labels
def add_observation_labels():
    print("Starting to add observation labels...")
    # File paths
    merge_file = "guess_my_city/merged_clusters.jsonl"  # Merged data
    origin_file = "llama3-8b_guess_my_city_embedding_msgs_game_with_action_labels.jsonl"  # File with action labels
    updated_file = "llama3-8b_guess_my_city_with_labels_k2_to_k100.jsonl"  # Final updated file

    # Load merge_clusters.jsonl data into memory
    # Create a dictionary for each k value
    merge_data = {}
    for k in range(2, 101):
        merge_data[k] = {}

    with open(merge_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            observation_id = entry["action_id"]
            
            # Get labels for each k value
            for k in range(2, 101):
                k_label_key = f"k{k}_label"
                if k_label_key in entry:
                    # If no game_id provided, use only action_id as key
                    merge_data[k][observation_id] = entry[k_label_key]

    # Open original file, process line by line and update
    with open(origin_file, 'r') as infile, open(updated_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            # Get game_id
            game_id = data["game_id"]
            
            # Create observation_labels for each k value
            observation_labels_by_k = {}
            for k in range(2, 101):
                observation_labels_by_k[k] = [
                    merge_data[k].get(observation_id, None)  # Look up using only observation_id
                    for observation_id in data["observation_ids"]
                ]

            # Reconstruct JSON data
            new_data = {}

            # First insert all key-value pairs from original data
            for key, value in data.items():
                if key == "action_embeddings":
                    # Insert all observation_k{num}_label at this position
                    for k in range(2, 101):
                        new_data[f"observation_k{k}_label"] = observation_labels_by_k[k]
                
                new_data[key] = value  # Insert original key-value pair

            # Write updated JSON data to new file
            outfile.write(json.dumps(new_data) + '\n')

    print(f"Observation labels added, saved as {updated_file}")

# Main function
def main():
    print("Starting guess my city data hierarchical clustering and label processing workflow...")
    
    # Execute Part 2: Merge clustering results
    merge_clusters()
    
    # Execute Part 3: Add action labels
    add_action_labels()
    
    # Execute Part 4: Add observation labels
    add_observation_labels()
    
    print("All processing completed!")

if __name__ == "__main__":
    main()