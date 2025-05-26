#!/usr/bin/env python3
"""
Data Processing Script for Game Environments
Generates training data from clustered results by selecting optimal k values
"""

import json
import argparse
from pathlib import Path


def process_observation(obs):
    """Convert text observation to standard label"""
    if 'Yes' in obs or 'yes' in obs:
        return 'a'
    elif 'No' in obs or 'no' in obs:
        return 'b'
    else:
        return 'c'


def process_twenty_questions(input_file, output_file, k=36):
    """
    Process twenty questions data with selected k value
    
    Args:
        input_file: Path to input JSONL file with k2_to_k100 labels
        output_file: Path to output JSONL file
        k: Selected k value for clustering
    """
    print(f"Processing twenty questions data with k={k}")
    
    # Load data
    with open(input_file) as f:
        results = [json.loads(line) for line in f.readlines()]
    
    print(f"Loaded {len(results)} games")
    
    new_data = []
    for item in results:
        # Add selected k labels
        item.update({"action_label": item[f"action_k{k}_label"]})
        
        # Process observations manually for twenty questions
        observations = item['obs']
        observation_labels = [process_observation(obs) for obs in observations]   
        item.update({"observation_label": observation_labels})
        
        # Remove embedding data to save space
        item.pop("action_embeddings", None)
        item.pop("obs_embeddings", None)
        
        new_data.append(item)
    
    # Save processed data
    with open(output_file, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(new_data)} processed games to {output_file}")


def process_bargaining(input_file, output_file, alice_k=36, bob_k=36):
    """
    Process bargaining data with selected k values for Alice and Bob
    
    Args:
        input_file: Path to input JSONL file with k2_to_k100 labels
        output_file: Path to output JSONL file
        alice_k: Selected k value for Alice's clustering
        bob_k: Selected k value for Bob's clustering
    """
    print(f"Processing bargaining data with alice_k={alice_k}, bob_k={bob_k}")
    
    # Load data
    with open(input_file) as f:
        results = [json.loads(line) for line in f.readlines()]
    
    print(f"Loaded {len(results)} games")
    
    new_data = []
    for item in results:
        # Add selected k labels for both Alice and Bob
        item.update({
            "alice_label": item[f"alice_k{alice_k}_label"],
            "bob_label": item[f"bob_k{bob_k}_label"]
        })
        
        # Remove embedding data to save space
        item.pop("bob_actions_embeddings", None)
        item.pop("alice_action_embeddings", None)
        
        new_data.append(item)
    
    # Save processed data
    with open(output_file, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(new_data)} processed games to {output_file}")


def process_negotiation(input_file, output_file, alice_k=36, bob_k=36):
    """
    Process negotiation data with selected k values for Alice and Bob
    Same processing logic as bargaining
    
    Args:
        input_file: Path to input JSONL file with k2_to_k100 labels
        output_file: Path to output JSONL file
        alice_k: Selected k value for Alice's clustering
        bob_k: Selected k value for Bob's clustering
    """
    print(f"Processing negotiation data with alice_k={alice_k}, bob_k={bob_k}")
    
    # Load data
    with open(input_file) as f:
        results = [json.loads(line) for line in f.readlines()]
    
    print(f"Loaded {len(results)} games")
    
    new_data = []
    for item in results:
        # Add selected k labels for both Alice and Bob
        item.update({
            "alice_label": item[f"alice_k{alice_k}_label"],
            "bob_label": item[f"bob_k{bob_k}_label"]
        })
        
        # Remove embedding data to save space
        item.pop("bob_actions_embeddings", None)
        item.pop("alice_action_embeddings", None)
        
        new_data.append(item)
    
    # Save processed data
    with open(output_file, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(new_data)} processed games to {output_file}")


def process_guess_my_city(input_file, output_file, action_k=36, observation_k=36):
    """
    Process guess my city data with selected k values for actions and observations
    
    Args:
        input_file: Path to input JSONL file with k2_to_k100 labels
        output_file: Path to output JSONL file
        action_k: Selected k value for action clustering
        observation_k: Selected k value for observation clustering
    """
    print(f"Processing guess my city data with action_k={action_k}, observation_k={observation_k}")
    
    # Load data
    with open(input_file) as f:
        results = [json.loads(line) for line in f.readlines()]
    
    print(f"Loaded {len(results)} games")
    
    new_data = []
    for item in results:
        # Add selected k labels for both actions and observations
        item.update({
            "action_label": item[f"action_k{action_k}_label"],
            "observation_label": item[f"observation_k{observation_k}_label"]
        })
        
        # Remove embedding data to save space
        item.pop("action_embeddings", None)
        item.pop("obs_embeddings", None)
        
        new_data.append(item)
    
    # Save processed data
    with open(output_file, "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(new_data)} processed games to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process game data with optimal k values")
    parser.add_argument("environment", choices=["twenty_questions", "bargaining", "negotiation", "guess_my_city"], 
                       help="Game environment to process")
    parser.add_argument("input_file", help="Input JSONL file with k2_to_k100 labels")
    parser.add_argument("output_file", help="Output JSONL file for training data")
    parser.add_argument("--k", type=int, default=36, help="K value for single-agent environments")
    parser.add_argument("--alice-k", type=int, default=36, help="K value for Alice in multi-agent environments")
    parser.add_argument("--bob-k", type=int, default=36, help="K value for Bob in multi-agent environments")
    parser.add_argument("--action-k", type=int, default=36, help="K value for actions")
    parser.add_argument("--observation-k", type=int, default=36, help="K value for observations")
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    # Process based on environment
    if args.environment == "twenty_questions":
        process_twenty_questions(args.input_file, args.output_file, args.k)
    elif args.environment == "bargaining":
        process_bargaining(args.input_file, args.output_file, args.alice_k, args.bob_k)
    elif args.environment == "negotiation":
        process_negotiation(args.input_file, args.output_file, args.alice_k, args.bob_k)
    elif args.environment == "guess_my_city":
        process_guess_my_city(args.input_file, args.output_file, args.action_k, args.observation_k)
    
    print("Processing completed successfully!")


# Example usage functions for direct calling
def process_twenty_questions_example():
    """Example usage for twenty questions"""
    env = "twenty_questions"
    k = 36
    
    input_file = f"llama3-8b_iter2_{env}_online_with_labels_k2_to_k100.jsonl"
    output_file = f"llama3-8b_online_iter2_{env}_embedding_msgs_with_labels_q.jsonl"
    
    process_twenty_questions(input_file, output_file, k)


def process_bargaining_example():
    """Example usage for bargaining"""
    env = "bargaining"
    alice_k = 36
    bob_k = 36
    
    input_file = f"llama3-8b_{env}_with_labels_k2_to_k100.jsonl"
    output_file = f"llama3-8b_{env}_with_selected_labels.jsonl"
    
    process_bargaining(input_file, output_file, alice_k, bob_k)


def process_negotiation_example():
    """Example usage for negotiation"""
    env = "negotiation"
    alice_k = 36
    bob_k = 36
    
    input_file = f"llama3-8b_{env}_with_labels_k2_to_k100.jsonl"
    output_file = f"llama3-8b_{env}_with_selected_labels.jsonl"
    
    process_negotiation(input_file, output_file, alice_k, bob_k)


def process_guess_my_city_example():
    """Example usage for guess my city"""
    env = "guess_my_city"
    action_k = 36
    observation_k = 36
    
    input_file = f"llama3-8b_{env}_with_labels_k2_to_k100.jsonl"
    output_file = f"llama3-8b_{env}_with_selected_labels.jsonl"
    
    process_guess_my_city(input_file, output_file, action_k, observation_k)


if __name__ == "__main__":
    main()