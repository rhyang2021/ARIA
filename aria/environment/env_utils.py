import os
import json
import time
import random
from typing import List, Dict, Any, Tuple
import sys
import torch
import numpy as np
import wandb
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()
import pdb
from accelerate import Accelerator
from twenty_questions import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv
from guess_my_city import GuessMyCityEnv, BatchedGuessMyCityEnv
from llm_base import llm_azure, llm_openai, llm_gpt, vllm


def collect_environment_data(env_name: str, agent, word_idx: int = None) -> Tuple[List[Dict], Dict]:
    """
    Collect trajectory data from a specific environment with an agent.
    
    Args:
        env_name: Name of the environment to use ('twenty_questions' or 'guess_my_city')
        agent: Agent to interact with the environment
        word_idx: Optional index to specify which word to use in the environment
        
    Returns:
        Tuple containing: 
          - list of trajectory dictionaries with state/action/reward info
          - metadata dictionary with summary statistics
    """
    if env_name == "twenty_questions":
        env = TwentyQuestionsEnv()
        word_list_key = "curr_word[0]"
    elif env_name == "guess_my_city":
        env = GuessMyCityEnv()
        word_list_key = "curr_word"
    else:
        raise NotImplementedError(f"Environment '{env_name}' not implemented")
    
    init = env.reset(idx=word_idx) if word_idx is not None else env.reset()
    next_obs = init
    steps, done = 0, False
    trajectory = []
    

    while not done:
        steps += 1
        observation = next_obs[0]['content']
        action = agent.get_action([observation])[0]
        next_obs, answer, reward, done = env._step(question=action)
        
        curr_word = env.curr_word[0] if env_name == "twenty_questions" else env.curr_word
        
        trajectory_item = {
            "instruction": init[0]['content'],
            "curr_world": curr_word,
            "question": action,
            "answer": answer,
            "observation": observation,
            "action": action,
            "next_observation": next_obs[0]['content'],
            "reward": reward,
            "done": done
        }
        trajectory.append(trajectory_item)
        
        if steps > env.max_conversation_length:
            break

    # Calculate different reward metrics
    total_reward = sum(item['reward'] for item in trajectory)
    final_reward = trajectory[-1]['reward'] if trajectory else 0
    discount_factor = 0.95
    discount_return = sum(pow(discount_factor, len(trajectory) - 1 - idx) * final_reward 
                            for idx in range(len(trajectory)))
    
    # Add aggregated reward metrics to each trajectory step
    for item in trajectory:
        item.update({"trajectory_reward": total_reward, "mc_return": discount_return})
    
    metadata = {
        "total_reward": total_reward,
        "final_reward": final_reward,
        "length": len(trajectory),
        "curr_word": curr_word,
        "success": done and final_reward == 0
    }
    
    return trajectory, metadata


def interact_with_environment(env_name: str, agent, output_dir: str, repeat: int) -> List[Dict]:
    """
    Run multiple interactions with an environment and save trajectories.
    
    Args:
        env_name: Name of the environment to use
        agent: Agent to interact with the environment
        output_dir: Directory to save trajectory data
        repeat: Number of trajectories to generate
        
    Returns:
        List of all trajectory dictionaries collected
    """
    env_output_dir = f"{output_dir}/{env_name}"
    os.makedirs(env_output_dir, exist_ok=True)
    all_trajectories = []
    
    for variation in tqdm(range(repeat), desc=f"Generating {env_name} trajectories"):
        variation_path = f"{env_output_dir}/variation-{variation}.jsonl"
        
        trajectory, _ = collect_environment_data(env_name, agent)
        
        if trajectory:
            with open(variation_path, 'w') as f:
                for item in trajectory:
                    f.write(json.dumps(item) + "\n")
            all_trajectories.extend(trajectory)
    
    return all_trajectories

def get_word_list(env_name: str):
    """
    Get the list of possible words/targets for a specific environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        List of words or targets for the specified environment
    """
    if env_name == "twenty_questions":
        from twenty_questions import DEFAULT_OBJECT_LIST
        word_list = DEFAULT_OBJECT_LIST
        # Split multiple possible spellings into a list
        return [list(map(lambda x: x.lower(), word.split(";"))) for word in word_list]
    elif env_name == "guess_my_city":
        from guess_my_city import CITY_LIST
        return CITY_LIST
    raise ValueError(f"Unknown environment: {env_name}")



def batch_interact_environment(agent, iteration: int, sample_size: int = None, n_rollout: int = None, agent_type: str = "") -> List[Dict]:
    """
    Run agent interactions with multiple environments and collect trajectory data.
    
    Args:
        agent: Agent providing actions
        iteration: Current iteration number, used for output directory naming
        sample_size: Number of samples to collect per environment
        n_rollout: Not used, kept for compatibility
        agent_type: Agent type identifier for output directory
        
    Returns:
        List of all trajectory data from all environments
    """
    if agent_type:
        output_dir = f"outputs/{agent_type}/trajectories/iter_{iteration}/"
    else:
        output_dir = f"outputs/trajectories/iter_{iteration}/"
    all_outputs = []
    
    for env_name in ['twenty_questions', 'guess_my_city']:
        outputs = interact_with_environment(env_name, agent, output_dir, sample_size or 10)
        all_outputs.extend(outputs)
    
    return all_outputs