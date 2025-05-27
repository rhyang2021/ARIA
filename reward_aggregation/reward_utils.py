import json
from typing import Dict, Any, List
import copy
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def visualize_q_values_count(q_values, top_n=50):
    """
    Visualize the length of values list for each key in Q-values dictionary
    
    Parameters:
    - q_values: Q-values dictionary in format {key: [values...]}
    - top_n: Show top N state-action pairs with most samples
    
    Returns:
    - plt: matplotlib plot object
    """
    # Calculate the length of values list for each key
    counts = {k: len(v) for k, v in q_values.items()}
    
    # Get top_n keys with most samples
    top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not top_items:
        print("No data to visualize")
        return None
    
    keys, count_values = zip(*top_items)
    
    # Create DataFrame for seaborn plotting
    df = pd.DataFrame({
        'State-Action Pair': keys,
        'Sample Count': count_values
    })
    
    # Set figure size
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Sample Count', y='State-Action Pair', data=df)
    
    # Add title and labels
    plt.title(f'Top {top_n} State-Action Pairs with Most Samples', fontsize=16)
    plt.xlabel('Sample Count', fontsize=14)
    plt.ylabel('State-Action Pair', fontsize=14)
    
    # Display count values on bars
    for i, count in enumerate(count_values):
        ax.text(count + 0.5, i, str(count), va='center')
    
    plt.tight_layout()
    
    # Print basic statistics
    all_counts = list(counts.values())
    print(f"Total number of state-action pairs: {len(counts)}")
    print(f"Total number of samples: {sum(all_counts)}")
    print(f"Average samples per state-action pair: {sum(all_counts)/len(all_counts) if all_counts else 0:.2f}")
    print(f"Maximum sample count: {max(all_counts) if all_counts else 0}")
    print(f"Minimum sample count: {min(all_counts) if all_counts else 0}")
    
    return plt

# Usage example
# Assuming you have q_values dictionary after pdb breakpoint
# visualize_q_values_count(q_values).savefig('q_values_counts.png')

def filter_q_values_by_count(q_values, threshold=3):
    """
    Filter out Q-values with sample count less than threshold
    
    Parameters:
    - q_values: Q-values dictionary in format {key: [values...]}
    - threshold: Sample count threshold, keys with fewer samples will be filtered out
    
    Returns:
    - filtered_q_values: Filtered Q-values dictionary
    """
    # Filter out keys with values list length less than threshold
    filtered_q_values = {k: v for k, v in q_values.items() if len(v) >= threshold}
    
    # Print filtering statistics
    original_count = len(q_values)
    filtered_count = len(filtered_q_values)
    removed_count = original_count - filtered_count
    
    print(f"Original state-action pair count: {original_count}")
    print(f"Filtered state-action pair count: {filtered_count}")
    print(f"Removed state-action pair count: {removed_count} ({removed_count/original_count*100:.2f}% removed)")
    
    return filtered_q_values

# Usage example
# Filter before calculating average Q-values
# pdb.set_trace()
# filtered_q_values = filter_q_values_by_count(q_values, threshold=5)
# 
# # Then use filtered q_values to calculate averages
# for key, values in filtered_q_values.items():
#     filtered_q_values[key] = sum(values) / len(values) if values else 0

def get_single_values(results, value_type='adv', game="twenty_questions", n_rollout=0, visualize=True):
    """
    Calculate Q-values or advantage values for single agent
    
    Parameters:
    - results: Results list
    - value_type: Value type ('q' or 'adv' for advantage values)
    - game: Game type
    - n_rollout: Filter threshold, only keep state-action pairs with sample count above threshold
    - visualize: Whether to generate visualization plots
    
    Returns:
    - Value dictionary: State-action pair to value mapping
    """
    gamma = 0.95  # Discount factor
    
    # Calculate state value function V(s)
    v_values = {}
    for result in results:
        action_labels = result['action_label']
        obs_labels = result['observation_label']
        reward = result['reward']+1
        
        min_length = min(len(action_labels), len(obs_labels))
        sequence = []
        
        # Build state sequence and calculate state values
        for i in range(min_length):
            # Append action and observation labels
            sequence.append(str(action_labels[i]))
            sequence.append(str(obs_labels[i]))
            
            # Build state key
            state_key = "-".join(sequence)
            
            # Collect rewards for this state
            if state_key not in v_values:
                v_values[state_key] = []
            v_values[state_key].append(reward)
    
    # Calculate average reward for each state as state value
    for key, values in v_values.items():
        v_values[key] = sum(values) / len(values) if values else 0

    # Calculate state-action value function Q(s,a)
    q_values = {}
    
    # Calculate Q values
    for result in results:
        action_labels = result['action_label']
        obs_labels = result['observation_label']
        reward = result['reward']+1
        min_length = min(len(action_labels), len(obs_labels))
        
        # Build state-action sequence and calculate Q values
        sequence = []
        for i in range(min_length - 1):  # -1 because need next action
            # Special handling for first round
            if i == 0:
                sequence.append(str(action_labels[i]))
            
            # Add observation and next action
            sequence.append(str(obs_labels[i]))
            sequence.append(str(action_labels[i+1]))
            
            # Build state-action key
            sa_key = "-".join(sequence)
            
            # Collect rewards for this state-action pair
            if sa_key not in q_values:
                q_values[sa_key] = []
            
            # Process reward based on value type
            if value_type == 'q':
                # Apply discount factor for Q values (discount for remaining steps)
                discount = pow(gamma, max(min_length - len(sequence) // 2, 0))
                q_values[sa_key].append(reward * discount)
            else:
                # Advantage values use original reward
                q_values[sa_key].append(reward)
    
    # Visualize Q-value distribution (if needed)
    if visualize:
        visualize_q_values_count(q_values).savefig(f'{game}_q_values_counts.png')
    
    # Filter Q values based on sample count
    q_values = filter_q_values_by_count(q_values, threshold=n_rollout)
    
    # Calculate average value for each state-action pair
    for key, values in q_values.items():
        q_values[key] = sum(values) / len(values) if values else 0
    
    # If only Q values needed, return directly
    if value_type == 'q':
        return q_values
    
    # Calculate advantage values A(s,a) = Q(s,a) - V(s)
    advantage = {}
    for sa_key, q in q_values.items():
        # Extract state key from state-action key (remove last action)
        state_key = '-'.join(sa_key.split("-")[:-1])
        
        # Calculate advantage value
        advantage[sa_key] = q - v_values.get(state_key, 0)
    
    return advantage



def get_multi_values(results, agent='alice', value_type='adv', game="bargaining", n_rollout=0, visualize=True):
    """
    Calculate Q-values or advantage values for agents
    
    Parameters:
    - results: Results list
    - agent: Agent name ('alice' or 'bob')
    - value_type: Value type ('q' or 'advantage')
    
    Returns:
    - Value dictionary
    """
    # Select reward based on agent
    if agent == 'alice':
        reward_key = 'alice_reward'
        gamma = 0.95  # Alice's discount factor
    else:  # bob
        reward_key = 'bob_reward'
        gamma = 0.95  # Bob's discount factor
    
    # Calculate state value function V(s)
    v_values = {}
    
    if agent == 'alice':
        # Alice's state representation
        for result in results:
            alice_k_label = result[f'alice_label']
            bob_k_label = result[f'bob_label']
            reward = result[reward_key]
            
            min_length = min(len(alice_k_label), len(bob_k_label))
            sequence = []
            
            for i in range(min_length):
                sequence.append(str(alice_k_label[i]))
                sequence.append(str(bob_k_label[i]))
                key = "-".join(sequence)
                
                if key not in v_values:
                    v_values[key] = []
                v_values[key].append(reward)
    else:
        # Bob's state representation
        for result in results:
            alice_k_label = result[f'alice_label']
            bob_k_label = result[f'bob_label']
            reward = result[reward_key]
            
            min_length = min(len(alice_k_label), len(bob_k_label))
            
            # Build all possible state sequences: (a1), (a1,b1,a2), ...
            for i in range(min_length):
                sequence = []
                for j in range(i+1):
                    if j == 0:
                        sequence.append(str(alice_k_label[j]))
                    else:
                        sequence.append(str(bob_k_label[j-1]))
                        if j < i+1:  # Avoid adding extra alice action at the end
                            sequence.append(str(alice_k_label[j]))
                
                key = "-".join(sequence)
                if key not in v_values:
                    v_values[key] = []
                v_values[key].append(reward)
    
    # Calculate average reward for each state
    for key, values in v_values.items():
        v_values[key] = sum(values) / len(values) if values else 0

    # Calculate action-state value function Q(s,a)
    q_values = {}
    
    if agent == 'alice':
        # Alice's Q value calculation
        for result in results:
            alice_k_label = result[f'alice_label']
            bob_k_label = result[f'bob_label']
            reward = result[reward_key]
            min_length = min(len(alice_k_label), len(bob_k_label))
            
            sequence = []
            for i in range(min_length - 1):
                if i == 0:
                    sequence.append(str(alice_k_label[i]))
                sequence.append(str(bob_k_label[i]))
                sequence.append(str(alice_k_label[i+1]))
                
                key = "-".join(sequence)
                if key not in q_values:
                    q_values[key] = []
                
                # For Q values, may apply discount factor; for advantage values, don't apply
                if value_type == 'q':
                    q_values[key].append(reward * pow(gamma, max(min_length-len(sequence), 0)))
                else:
                    q_values[key].append(reward)
    else:
        # Bob's Q value calculation
        for result in results:
            alice_k_label = result[f'alice_label']
            bob_k_label = result[f'bob_label']
            reward = result[reward_key]
            
            min_length = min(len(alice_k_label), len(bob_k_label))
            
            # Build all possible state-action sequences: (a1,b1), (a1,b1,a2,b2), ...
            for i in range(min_length):
                sequence = []
                for j in range(i+1):
                    sequence.append(str(alice_k_label[j]))
                    sequence.append(str(bob_k_label[j]))
                
                key = "-".join(sequence)
                if key not in q_values:
                    q_values[key] = []
                q_values[key].append(reward)
    
    if visualize:
        visualize_q_values_count(q_values).savefig(f'{game}_{agent}_q_values_counts.png')
    # Calculate average Q values
    q_values = filter_q_values_by_count(q_values, threshold=n_rollout)
    for key, values in q_values.items():
        q_values[key] = sum(values) / len(values) if values else 0
    
    # If only Q values needed, return directly
    if value_type == 'q':
        return q_values
    
    # Calculate advantage values A(s,a) = Q(s,a) - V(s)
    advantage = {}
    
    if agent == 'alice':
        for k, q in q_values.items():
            v_key = '-'.join(k.split("-")[:-1])
            advantage[k] = q - v_values.get(v_key, 0)
    else:
        for k, q in q_values.items():
            # For state-action sequence (a1,b1,a2,b2), corresponding state sequence is (a1,b1,a2)
            tokens = k.split("-")
            if len(tokens) >= 2:
                v_key = "-".join(tokens[:-1])  # Remove last bob action
                advantage[k] = q - v_values.get(v_key, 0)
    
    return advantage