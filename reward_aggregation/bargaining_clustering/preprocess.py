import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Part 1: Add game IDs to data
def add_game_ids():
    print("Starting to add game IDs...")
    game_id = 0
    with open(f"../../dataset/llama3-8b_bargaining_embedding_msgs.jsonl", 'r') as infile, open(f"../dataset/llama3-8b_bargaining_embedding_msgs_game.jsonl", "w") as outfile:
        for line in infile:
            data = json.loads(line)
            data0 = {}
            data0["game_id"] = game_id
            data0 = {**data0, **data}
            game_id += 1
            outfile.write(json.dumps(data0) + "\n")
    print(f"Game ID addition completed, processed {game_id} games")

# Part 2: Extract Alice and Bob's actions, create action list with IDs
def create_action_list():
    print("Starting to create action list...")
    with open(f'../dataset/llama3-8b_bargaining_embedding_msgs_game.jsonl', 'r') as infile, open(f'../dataset/action_list_with_id.jsonl', 'w') as outfile:
        for line in infile:
            # Parse each line of JSON
            data = json.loads(line.strip())
            
            # Get game ID and Alice/Bob's data
            game_id = data["game_id"]
            alice_actions = data["alice_actions"]
            bob_actions = data["bob_actions"]
            alice_ids = data["alice_ids"]
            bob_ids = data["bob_ids"]
            alice_embeddings = data["alice_action_embeddings"]
            bob_embeddings = data["bob_actions_embeddings"]
            
            # Process Alice's actions
            for action, action_id, action_embedding in zip(alice_actions, alice_ids, alice_embeddings):
                new_entry = {
                    "game_id": game_id,
                    "action_id": action_id,
                    "game_action_id": f"b_{game_id}_{action_id}",
                    "action": action,
                    "action_embedding": action_embedding
                }
                # Write Alice's action
                outfile.write(json.dumps(new_entry) + '\n')
            
            # Process Bob's actions
            for action, action_id, action_embedding in zip(bob_actions, bob_ids, bob_embeddings):
                new_entry = {
                    "game_id": game_id,
                    "action_id": action_id,
                    "game_action_id": f"b_{game_id}_{action_id}",
                    "action": action,
                    "action_embedding": action_embedding
                }
                # Write Bob's action
                outfile.write(json.dumps(new_entry) + '\n')
    print("Action list creation completed")

# Part 3: Format IDs and create new file
def format_ids():
    print("Starting to format IDs...")
    with open(f'../dataset/llama3-8b_bargaining_embedding_msgs_game.jsonl', 'r') as infile, open(f'../dataset/bargaining_with_formatted_ids.jsonl', 'w') as outfile:
        for line in infile:
            # Parse each line of JSON
            data = json.loads(line.strip())
            
            # Get game ID
            game_id = data["game_id"]
            
            # Reformat alice_ids
            formatted_alice_ids = [f"b_{game_id}_{alice_id}" for alice_id in data["alice_ids"]]
            
            # Reformat bob_ids
            formatted_bob_ids = [f"b_{game_id}_{bob_id}" for bob_id in data["bob_ids"]]
            
            # Update data
            data["alice_ids"] = formatted_alice_ids
            data["bob_ids"] = formatted_bob_ids
            
            # Write to new file
            outfile.write(json.dumps(data) + '\n')
    print("ID formatting completed, new file saved as bargaining_with_formatted_ids.jsonl")


# Main function
def main():
    print("Starting to process bargaining data for hierarchical clustering...")
    
    # Execute Part 1: Add game IDs to data
    add_game_ids()
    
    # Execute Part 2: Create action list with IDs
    create_action_list()
    
    # Execute Part 3: Format IDs
    format_ids()
    

    print("All processing completed!")

if __name__ == "__main__":
    main()