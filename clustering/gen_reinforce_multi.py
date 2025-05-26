import json
import copy
import sys
import pdb
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from prompt_base import INSTRUCT_BARGAIN_SYSTEM, INSTRUCT_NEGO_BUYER_SYSTEM, INSTRUCT_NEGO_SELLER_SYSTEM
sys.path.append("/path/to/your/NLGames/utils")
from reward_utils import get_single_values, get_multi_values


# Constants definition
DATA_DIR = "/path/to/your/NLGames/data/train"
GAMES = ['bargaining', 'negotiation']
n_rollout = 0
reward = 'q'
OUTPUT_PATH = f"{DATA_DIR}/actor_reinforce_llama3-8b_gamma_0.001_multi_{reward}_filter_{n_rollout}.json"

def process_alice_interactions(result: Dict[str, Any], alice_rewards: Dict[str, float]) -> List[Dict[str, Any]]:
    outputs = []
    conversations = f"{result['player_1_system_prompt']}\n\n"
    alice_actions = result['alice_actions']
    bob_actions = result['bob_actions']
    alice_labels = result['alice_label']
    bob_labels = result['bob_label']
    
    # Alice's state representation consistent with get_multi_values
    sequence = []
    for i in range(min(len(alice_labels), len(bob_labels))):
        # First add current Alice's action label
        sequence.append(str(alice_labels[i]))
        
        # If it already contains at least one action, we can build state-action pairs
        if i > 0:
            # Build current conversation (excluding current Alice action)
            cur_conv = copy.deepcopy(conversations)
            
            # Get current state sequence string (excluding latest Alice action)
            state_key = "-".join(sequence[:-1])
            
            # Build state-action pair
            action_key = f"{state_key}-{sequence[-1]}"
            
            # Check if this state-action pair has a reward value
            if action_key in alice_rewards:
                cur_reward = alice_rewards[action_key]
                outputs.append({"observation": cur_conv.strip(), "action": f"{alice_actions[i]}", "reward": cur_reward})
        
        # Add Bob's action to sequence and conversation
        if i < len(bob_labels):
            sequence.append(str(bob_labels[i]))
            conversations += f"Alice: {alice_actions[i]}\nBob: {bob_actions[i]}\n\n"
    
    return outputs


def process_bob_interactions(result: Dict[str, Any], bob_rewards: Dict[str, float]) -> List[Dict[str, Any]]:
    outputs = []
    conversations = f"{result['player_2_system_prompt']}\n\n"
    alice_actions = result['alice_actions']
    bob_actions = result['bob_actions']
    alice_labels = result['alice_label']
    bob_labels = result['bob_label']
    
    # Bob's state representation consistent with get_multi_values
    sequence = []
    for i in range(min(len(alice_labels), len(bob_labels))):
        # Build sequence
        sequence.append(str(alice_labels[i]))
        
        # Add Alice's action to conversation
        cur_conv = copy.deepcopy(conversations)
        cur_conv += f"Alice: {alice_actions[i]}\n"
        
        # Add Bob's action to sequence
        sequence.append(str(bob_labels[i]))
        
        # Get state sequence (excluding latest Bob action)
        state_key = "-".join(sequence[:-1])
        
        # Build state-action pair
        action_key = f"{state_key}-{sequence[-1]}"
        
        # Check if this state-action pair has a reward value
        if action_key in bob_rewards:
            cur_reward = bob_rewards[action_key]
            outputs.append({"observation": cur_conv.strip(), "action": f"{bob_actions[i]}", "reward": cur_reward})
        
        # Update conversation history
        conversations += f"Alice: {alice_actions[i]}\nBob: {bob_actions[i]}\n\n"
    
    return outputs


def main():
    """Main function"""
    import random
    random.seed(42)
    
    all_outputs = []
    
    for game in GAMES:
        print(f"Processing game: {game}")
        # Load data
        input_file = f"{DATA_DIR}/llama3-8b_gamma_0.001_{game}_embedding_msgs_with_labels_{reward}.jsonl"
        with open(input_file) as f:
            results = [json.loads(line) for line in f]
        
        # Calculate reward mapping
        alice_rewards = get_multi_values(results, agent='alice', value_type=reward, game=game, n_rollout=n_rollout, visualize=False)
        bob_rewards = get_multi_values(results, agent='bob', value_type=reward, game=game, n_rollout=n_rollout, visualize=False)
        
        # Process each conversation
        for result in tqdm(results):
            # Process Alice's interactions
            alice_outputs = process_alice_interactions(result, alice_rewards)
            all_outputs.extend(alice_outputs)
            
            # Process Bob's interactions
            bob_outputs = process_bob_interactions(result, bob_rewards)
            all_outputs.extend(bob_outputs)
        
        print(f"Total outputs after processing {game}: {len(all_outputs)}")
    
    if reward == 'adv':
        # Reward normalization
        all_rewards = [output['reward'] for output in all_outputs]
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        
        # Standardization
        normalized_rewards = [(r - mean_reward) / std_reward for r in all_rewards]
        
        # Find max and min values
        min_reward = min(normalized_rewards)
        max_reward = max(normalized_rewards)
        
        # Apply Min-Max normalization
        for i, output in enumerate(all_outputs):
            normalized_reward = (normalized_rewards[i] - min_reward) / (max_reward - min_reward)
            output['reward'] = normalized_reward

    # Save output
    random.shuffle(all_outputs)  # Randomize output order
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_outputs, f, indent=4)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()