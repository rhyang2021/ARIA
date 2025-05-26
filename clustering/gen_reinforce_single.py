import json
import copy
import sys
import pdb
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from prompt_base import INSTRUCTION_GUESS_MY_CITY, INSTRUCTION_TWENTY_QUESTIONS
sys.path.append("/path/to/your/NLGames/utils")
from reward_utils import get_single_values, get_multi_values

# Constants definition
DATA_DIR = "/path/to/your/NLGames/data/train"
n_rollout = 0
reward = 'q'
GAMES = ['twenty_questions', 'guess_my_city']
OUTPUT_PATH = f"{DATA_DIR}/actor_reinforce_llama3-8b_online_iter2_single_{reward}_filter_{n_rollout}.json"

def process_interactions(result: Dict[str, Any], rewards: Dict[str, float], game) -> List[Dict[str, Any]]:
    """
    Process single agent interactions, mapping state-action pairs to labels and obtaining rewards
    
    Parameters:
    - result: Single conversation result dictionary
    - rewards: Mapping dictionary from state-action pairs to rewards
    - game: Game type
    
    Returns:
    - outputs: List of processed interactions for training
    """
    outputs = []
    instruction = INSTRUCTION_TWENTY_QUESTIONS if game == 'twenty_questions' else INSTRUCTION_GUESS_MY_CITY
    conversations = ""
    actions = result['actions']
    observations = result['obs']
    
    # Get action and observation labels
    action_labels = result['action_label']
    obs_labels = result['observation_label']
    
    min_length = min(len(action_labels), len(obs_labels))
    
    # Build same state representation as get_single_values
    sequence = []
    
    for i in range(min_length - 1):  # -1 because we need next action to form (s,a) pair
        # Special handling for initial state
        if i == 0:
            sequence.append(str(action_labels[0]))
            
            # For first round, no previous state, add directly
            cur_conv = ""  # Empty history
            
            # No need to build new sample, as first question has no previous state
        
        # Add current observation
        sequence.append(str(obs_labels[i]))
        
        # Add next action to build state-action pair
        sequence.append(str(action_labels[i+1]))
        
        # Get current state-action key (consistent with get_single_values)
        action_key = "-".join(sequence)
        
        # Check if this state-action pair has a reward value
        if action_key in rewards:
            # Build current conversation history
            cur_conv = copy.deepcopy(conversations)
            
            # Add to output
            cur_reward = rewards[action_key]
            outputs.append({
                "observation": instruction.format(history="Here is the game history:\n"+cur_conv if cur_conv else "").strip(), 
                "action": f"Question: {actions[i+1]}",  # i+1 is the next action 
                "reward": cur_reward
            })
        
        # Update conversation history (current question and answer)
        conversations += f"Question: {actions[i]}\nAnswer: {observations[i]}\n\n"
        
    # Add conversation history for the last round
    if min_length > 0:
        conversations += f"Question: {actions[min_length-1]}\nAnswer: {observations[min_length-1]}\n\n"
    
    return outputs


def main():
    """Main function"""
    import random
    random.seed(42)
    
    all_outputs = []
    
    for game in GAMES:
        print(f"Processing game: {game}")
        # Load data
        input_file = f"{DATA_DIR}/llama3-8b_online_iter2_{game}_embedding_msgs_with_labels_{reward}.jsonl"
        with open(input_file) as f:
            results = [json.loads(line) for line in f]
        
        # Calculate reward mapping - using get_single_values function
        rewards = get_single_values(results, value_type=reward, game=game, n_rollout=n_rollout, visualize=False)

        # Process each conversation
        for result in tqdm(results):
            # Process agent interactions
            outputs = process_interactions(result, rewards, game)
            all_outputs.extend(outputs)
            
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
    
    # Randomize output order
    random.shuffle(all_outputs)
    
    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_outputs, f, indent=4)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()