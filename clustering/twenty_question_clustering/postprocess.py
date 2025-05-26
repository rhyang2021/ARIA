#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import defaultdict

# Part 1: Merge all clustering results into a unified JSONL file
def merge_clusters(base_dir="twenty_question"):
    """
    Merge all clustering results into a unified JSONL file
    
    Args:
        base_dir: Path to the clustering results directory
    """
    print(f"Starting to merge clustering results, base directory: {base_dir}")
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
    return output_path

# Part 2: Update original file with action labels
def add_action_labels(
    merge_file="twenty_question/merged_clusters.jsonl", 
    origin_file="llama3-8b_twenty_questions_embedding_msgs_game.jsonl", 
    updated_file="llama3-8b_twenty_questions_with_labels_k2_to_k100.jsonl"
):
    """
    Add clustering labels to original data
    
    Args:
        merge_file: Path to merged clustering data file
        origin_file: Path to original data file
        updated_file: Path to updated output file
    """
    print(f"Starting to add action labels...")
    
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
    
    print(f"Clustering label data loaded")
    
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
    return updated_file

def main():
    print("Starting twenty questions game data clustering post-processing workflow...")
    
    # Fixed directory and file paths
    base_dir = "twenty_question"
    origin_file = "llama3-8b_twenty_questions_embedding_msgs_game.jsonl"
    updated_file = "llama3-8b_twenty_questions_with_labels_k2_to_k100.jsonl"
    
    # Step 1: Merge clustering results
    merge_file = merge_clusters(base_dir)
    
    # Step 2: Add action labels
    final_file = add_action_labels(merge_file, origin_file, updated_file)
    
    print(f"Processing complete! Final results saved in: {final_file}")

if __name__ == "__main__":
    main()