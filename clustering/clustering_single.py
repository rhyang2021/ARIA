import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

def save_dict_to_json(data_dict, filename):
    """Save dictionary as a JSON file"""
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)

def process_observation(obs):
    """Convert text observation to standard label"""
    if 'Yes' in obs or 'yes' in obs:
        return 'a'
    elif 'No' in obs or 'no' in obs:
        return 'b'
    else:
        return 'c'

def get_values(results, index, value_type='advantage'):
    """
    Calculate the Q-values or advantage values for a specific k index
    
    Args:
        results: List of result dictionaries
        index: The k index to calculate values for
        value_type: Value type ('q' or 'advantage')
        
    Returns:
        Dictionary of values
    """
    gamma = 0.95  # Discount factor
    
    # Initialize state value function V(s)
    v_values = {}
    
    # Process each result
    for result_idx, result in enumerate(results):
        # Get action labels - check if they exist or create them
        if f'action_k{index}_label' not in result:
            continue
        alice_k_label = result[f'action_k{index}_label']
        
        # Get observation labels - check if they exist or create them from obs
        if f'observation_k{index}_label' not in result:
            if 'obs' in result:
                observations = result['obs']
                bob_k_label = [process_observation(obs) for obs in observations]
                # Store processed labels for future use
                result[f'observation_k{index}_label'] = bob_k_label
            else:
                continue
        else:
            bob_k_label = result[f'observation_k{index}_label']
        
        # Ensure labels are lists
        alice_k_label = [alice_k_label] if not isinstance(alice_k_label, list) else alice_k_label
        bob_k_label = [bob_k_label] if not isinstance(bob_k_label, list) else bob_k_label
        
        # Get reward
        reward = result.get('reward', 0)
        
        # Get minimum length between action and observation labels
        min_length = min(len(alice_k_label), len(bob_k_label))
        
        # Build state sequences and calculate values
        sequence = []
        for i in range(min_length):
            # Add Alice's question and Bob's answer
            sequence.append(str(alice_k_label[i]))
            sequence.append(str(bob_k_label[i]))
            
            # Convert sequence to state key
            state_key = "-".join(sequence)
            
            # Add reward for this state
            if state_key not in v_values:
                v_values[state_key] = []
            v_values[state_key].append(reward)
    
    # Calculate average reward for each state
    for key, values in v_values.items():
        v_values[key] = sum(values) / len(values) if values else 0
    
    # Calculate state-action value function Q(s,a)
    q_values = {}
    
    # Process each result again for Q-values
    for result_idx, result in enumerate(results):
        # Get action and observation labels
        if f'action_k{index}_label' not in result or f'observation_k{index}_label' not in result:
            continue
            
        alice_k_label = result[f'action_k{index}_label']
        bob_k_label = result[f'observation_k{index}_label']
        
        # Ensure labels are lists
        alice_k_label = [alice_k_label] if not isinstance(alice_k_label, list) else alice_k_label
        bob_k_label = [bob_k_label] if not isinstance(bob_k_label, list) else bob_k_label
        
        # Get reward
        reward = result.get('reward', 0)
        
        # Get minimum length
        min_length = min(len(alice_k_label), len(bob_k_label))
        
        # Skip if not enough data for transitions
        if min_length <= 1:
            continue
            
        # Build state-action sequences and calculate Q-values
        for i in range(min_length - 1):  # -1 for next action
            # Special handling for first round
            if i == 0:
                sequence = [str(alice_k_label[i])]
            else:
                sequence = sequence[:-1]  # Remove last element (next action from previous iteration)
            
            # Add Bob's answer and Alice's next question
            sequence.append(str(bob_k_label[i]))
            sequence.append(str(alice_k_label[i+1]))
            
            # Convert sequence to state-action key
            sa_key = "-".join(sequence)
            
            # Add reward for this state-action pair
            if sa_key not in q_values:
                q_values[sa_key] = []
            
            # Apply discount based on value type
            if value_type == 'q':
                # Apply discount factor
                discounted_reward = reward * pow(gamma, min_length - len(sequence) // 2)
                q_values[sa_key].append(discounted_reward)
            else:
                # For advantage values, use original reward
                q_values[sa_key].append(reward)
    
    # Calculate average for each state-action pair
    for key, values in q_values.items():
        q_values[key] = sum(values) / len(values) if values else 0
    
    # If only Q-values are needed, return them
    if value_type == 'q':
        return q_values
    
    # Calculate advantage values A(s,a) = Q(s,a) - V(s)
    advantage = {}
    
    for sa_key, q in q_values.items():
        # Extract state from state-action key
        parts = sa_key.split("-")
        if len(parts) >= 2:
            state_key = '-'.join(parts[:-1])
            
            # Calculate advantage value
            if state_key in v_values:
                advantage[sa_key] = q - v_values[state_key]
    
    return advantage

def create_result_dict(results, values_dict):
    """
    Create result dictionary, mapping action_key to values at different k indices
    
    Args:
        results: List of result dictionaries
        values_dict: Dictionary mapping k index to value dictionaries
        
    Returns:
        Dictionary mapping action_key to k-indexed values
    """
    result_dict = {}
    
    # Skip if no values
    if not any(values_dict.values()):
        print("Warning: No values found in values_dict")
        return result_dict
    
    # Process each result
    for result in results:
        # For each k index
        for index in values_dict.keys():
            # Skip if no values for this index
            if not values_dict[index]:
                continue
                
            # Get value dictionary for this index
            cur_vals = values_dict[index]
            
            # Skip if required keys don't exist
            if f'action_k{index}_label' not in result or f'observation_k{index}_label' not in result:
                continue
            
            # Get Alice's actions and labels
            alice_actions = result.get('actions', [])
            alice_k_label = result[f'action_k{index}_label']
            
            # Get Bob's observations and labels
            observations = result.get('obs', [])
            bob_k_label = result[f'observation_k{index}_label']
            bob_actions = observations
            
            # Ensure labels are lists
            alice_k_label = [alice_k_label] if not isinstance(alice_k_label, list) else alice_k_label
            bob_k_label = [bob_k_label] if not isinstance(bob_k_label, list) else bob_k_label
            
            # Skip if not enough data
            if len(bob_k_label) <= 1 or len(alice_k_label) <= 1:
                continue
            
            # Build action sequences and label sequences
            for i in range(len(bob_k_label) - 1):
                actions = []  # For action_key
                sequence = []  # For label_key
                
                # Special handling for first round
                if i == 0:
                    if i < len(alice_actions):
                        actions.append(str(alice_actions[i]))
                    if i < len(alice_k_label):
                        sequence.append(str(alice_k_label[i]))
                
                # Add Bob's answer and Alice's next question
                if i < len(bob_k_label):
                    sequence.append(str(bob_k_label[i]))
                    if i < len(bob_actions):
                        actions.append(str(bob_actions[i]))
                
                if i+1 < len(alice_k_label):
                    sequence.append(str(alice_k_label[i+1]))
                    if i+1 < len(alice_actions):
                        actions.append(str(alice_actions[i+1]))
                
                # Build keys
                label_key = "-".join(sequence)
                action_key = "-".join(actions)
                
                # If label sequence exists in value dictionary, add to result dictionary
                if label_key in cur_vals:
                    if action_key not in result_dict:
                        result_dict[action_key] = {}
                    result_dict[action_key][index] = cur_vals[label_key]
    
    return result_dict

def calculate_change_rates(result_dict):
    """
    Calculate change rates from index to index+1 for each action_key
    
    Args:
        result_dict: Dictionary mapping action_key to k-indexed values
        
    Returns:
        Dictionary mapping index to average change rate
    """
    change_rates = {}
    
    # Skip if no results
    if not result_dict:
        return change_rates
    
    # Process each action_key
    for action_key, index_values in result_dict.items():
        # Get sorted indices
        indices = sorted(index_values.keys())
        
        # For each pair of adjacent indices, calculate change rate
        for i in range(len(indices) - 1):
            current_idx = indices[i]
            next_idx = indices[i + 1]
            
            # Only consider consecutive indices (index and index+1)
            if next_idx == current_idx + 1:
                current_value = index_values[current_idx]
                next_value = index_values[next_idx]
                
                # Calculate absolute change
                change = abs(next_value - current_value)
                
                # Add change to corresponding index's list
                if current_idx not in change_rates:
                    change_rates[current_idx] = []
                change_rates[current_idx].append(change)
    
    # Calculate average change rate for each index
    avg_change_rates = {}
    for idx, changes in change_rates.items():
        avg_change_rates[idx] = np.mean(changes) if changes else 0
    
    return avg_change_rates

def plot_change_rates(avg_change_rates, filename, title, color='#82A9D9', threshold=0.01, stable_count=10):
    """
    Plot average change rates from index to index+1, and mark stable indices below threshold
    
    Args:
        avg_change_rates: Dictionary mapping index to average change rate
        filename: Output filename
        title: Chart title
        color: Line color
        threshold: Change rate threshold
        stable_count: Consecutive points below threshold to be considered stable
    """
    # Skip if no change rates
    if not avg_change_rates:
        print("Warning: No change rates to plot")
        return
    
    # Define font size and colors
    fs = 20  # 基础字体大小
    colors = ["#EE8535", "#BECB51", "#C1D5EC", "#82A9D9"]  # 颜色定义
    
    # Get sorted indices and corresponding change rates
    indices = sorted(avg_change_rates.keys())
    avg_changes = [avg_change_rates[idx] for idx in indices]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    
    # Plot main line with larger width
    plt.plot(indices, avg_changes, marker='o', linestyle='-', color=color, linewidth=3)
    
    # Set title and labels with larger fonts
    plt.title(title, fontsize=fs)
    plt.xlabel('Index (k)', fontsize=fs)
    plt.ylabel('Average Change Rate', fontsize=fs)
    
    # Remove top and right borders, make bottom and left borders thicker
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Add horizontal grid lines only
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add threshold horizontal line
    plt.axhline(y=threshold, color='purple', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Threshold = {threshold}')
    
    # Set x-axis ticks, every 5 indices with larger font
    xticks = []
    if indices:
        xticks.append(indices[0])  # Add first index
        
        # Add every 5th index
        for idx in indices:
            if idx % 5 == 0:
                xticks.append(idx)
        
        # Ensure last index is included
        if indices[-1] not in xticks:
            xticks.append(indices[-1])
        
        # Remove duplicates and sort
        xticks = sorted(set(xticks))
    
    # Set tick font sizes
    plt.xticks(xticks, fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    
    # Add labels for specific points with larger font
    for i, value in enumerate(avg_changes):
        idx = indices[i]
        if idx in xticks:
            plt.text(idx, value, f'{value:.2f}', ha='center', 
                    va='bottom' if value > 0 else 'top', fontsize=fs-7)
    
    # Find and mark max and min values with better colors
    if avg_changes:
        max_idx = indices[np.argmax(avg_changes)]
        min_idx = indices[np.argmin(avg_changes)]
        max_val = max(avg_changes)
        min_val = min(avg_changes)
        
        plt.scatter(max_idx, max_val, color=colors[0], s=150, marker='*', 
                    zorder=5, label=f'Max: {max_val:.2f} at k={max_idx}')
        plt.scatter(min_idx, min_val, color=colors[2], s=150, marker='*', 
                    zorder=5, label=f'Min: {min_val:.2f} at k={min_idx}')
    
    # Find stable index below threshold (consecutive points all below threshold)
    stable_below_threshold_idx = None
    stable_below_threshold_val = None
    
    for i in range(len(avg_changes) - stable_count + 1):
        if all(avg_changes[i+j] < threshold for j in range(stable_count)):
            stable_below_threshold_idx = indices[i]
            stable_below_threshold_val = avg_changes[i]
            break
    
    # Mark stable index if found - with larger marker
    if stable_below_threshold_idx is not None:
        plt.scatter(stable_below_threshold_idx, stable_below_threshold_val, 
                    color='purple', s=180, marker='D', 
                    edgecolor='black', linewidth=2, zorder=10, 
                    label=f'Stable below {threshold}: k={stable_below_threshold_idx} ({stable_below_threshold_val:.2f})')
        
        # Add vertical line at this index
        plt.axvline(x=stable_below_threshold_idx, color='purple', linestyle=':', alpha=0.6)
        
        print(f"Stable below threshold {threshold}: Starting at k={stable_below_threshold_idx}, rate={stable_below_threshold_val:.4f}")
    else:
        print(f"No stable sequence found below threshold {threshold} (needs {stable_count} consecutive points)")
    
    # Add legend with larger font
    plt.legend(fontsize=fs-7, loc='best')
    
    # Apply tight layout
    plt.tight_layout()
    
    # Save figure in both PNG and PDF formats
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Also save as PDF
    pdf_filename = filename.replace('.png', '.pdf')
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data as JSON
    data_to_save = {
        "values": {str(k): v for k, v in avg_change_rates.items()},
        "threshold": threshold,
        "stable_count": stable_count
    }
    
    if avg_changes:
        data_to_save["max"] = {"index": int(max_idx), "value": float(max_val)}
        data_to_save["min"] = {"index": int(min_idx), "value": float(min_val)}
    
    if stable_below_threshold_idx is not None:
        data_to_save["stable_below_threshold"] = {
            "index": int(stable_below_threshold_idx),
            "value": float(stable_below_threshold_val)
        }
    
    # Save data to JSON file
    save_dict_to_json(data_to_save, filename.replace('.png', '.json'))

def analyze_change_rates(results, env_name, output_dir, start_idx=2, end_idx=100, 
                         value_type='advantage', threshold=0.01, stable_count=10):
    """
    Analyze change rates across different k values
    
    Args:
        results: List of result dictionaries
        env_name: Environment name
        output_dir: Output directory
        start_idx: Start index, default 2
        end_idx: End index, default 100
        value_type: Value type ('q' or 'advantage')
        threshold: Change rate threshold
        stable_count: Consecutive points below threshold to be considered stable
        
    Returns:
        Dictionary mapping index to average change rate
    """
    print(f"Analyzing change rates for k values {start_idx}-{end_idx}...")
    
    # Calculate value dictionary for each k value
    values_dict = {}
    for k in tqdm(range(start_idx, end_idx + 1), desc=f"Calculating {value_type} values for each k"):
        # Check if k exists in the first few records
        key_exists = all(f'action_k{k}_label' in result for result in results[:5])
        
        if not key_exists:
            # Try to process observations if action labels exist but observation labels don't
            for result in results[:5]:
                if 'obs' in result and len(result.get('obs', [])) > 0:
                    # Process observations to create labels
                    observations = result['obs']
                    bob_k_label = [process_observation(obs) for obs in observations]
                    result[f'observation_k{k}_label'] = bob_k_label
        
        # Calculate values for current k
        values_dict[k] = get_values(results, k, value_type)
    
    # Skip if no values were calculated
    if not values_dict or not any(values_dict.values()):
        print("Error: No valid k values found")
        return {}
    
    print(f"Calculated values for {len(values_dict)} k indices")
    
    # Create result dictionary (action_id -> k values -> values)
    result_dict = create_result_dict(results, values_dict)
    
    # Output debug info
    print(f"Result dictionary contains {len(result_dict)} different action keys")
    
    # Calculate change rates
    avg_change_rates = calculate_change_rates(result_dict)
    
    if not avg_change_rates:
        print("Warning: No change rates calculated")
        return {}
    
    # Plot change rates
    output_file = os.path.join(output_dir, f"{env_name}_{value_type}_change_rates.png")
    plot_change_rates(
        avg_change_rates, 
        output_file,
        f"Average {value_type.capitalize()} Change Rate from k to k+1 ({env_name})",
        color='blue',
        threshold=threshold,
        stable_count=stable_count
    )
    
    # Identify k value with minimum change rate (most stable index)
    if avg_change_rates:
        best_k = min(avg_change_rates.items(), key=lambda x: x[1])[0]
        print(f"Most stable k value: {best_k}, change rate: {avg_change_rates[best_k]:.4f}")
    
    return avg_change_rates

def main():
    # Create command line argument parser
    parser = argparse.ArgumentParser(description="Analyze change rates across different k values")
    parser.add_argument("--env", default="twenty_questions", 
                        help="Environment name")
    parser.add_argument("--input_dir", type=str, default="",
                        help="Path to JSONL data file")
    parser.add_argument("--output_dir", type=str, default="", 
                        help="Output directory path")
    parser.add_argument("--index_range", type=str, default="2-50",
                        help="Index range to analyze, format 'start-end', default '2-50'")
    parser.add_argument("--value_type", choices=["q", "advantage", "adv"], default="q",
                        help="Value type to analyze (q or advantage)")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Change rate threshold for marking stable k values, default 0.005")
    parser.add_argument("--stable_count", type=int, default=20,
                        help="Consecutive points below threshold to be considered stable, default 20")
    
    args = parser.parse_args()
    
    # Parse index range
    try:
        start_idx, end_idx = map(int, args.index_range.split('-'))
        if start_idx < 2 or end_idx < start_idx:
            raise ValueError
    except:
        print("Invalid index range format, should be 'start-end', e.g. '2-100'")
        print("Start index cannot be less than 2")
        return
    
    # Normalize value_type
    if args.value_type == "adv":
        args.value_type = "advantage"
    
    # Create output directory
    output_path = os.path.join(args.output_dir, args.env)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    data_path = f"{args.input_dir}/llama3-8b_{args.env}_with_labels_k2_to_k100.jsonl"
    print(f"Loading data: {data_path}")
    try:
        # Try loading JSONL format
        if data_path.endswith('.jsonl'):
            with open(data_path) as f:
                results = [json.loads(line) for line in f]
        # Try loading JSON format
        elif data_path.endswith('.json'):
            with open(data_path) as f:
                data = json.load(f)
                # Check data structure, adapt to different formats
                if isinstance(data, list):
                    results = data
                elif isinstance(data, dict) and 'results' in data:
                    results = data['results']
                else:
                    print("Cannot recognize JSON data structure")
                    return
        else:
            print("Unsupported file format, please use .jsonl or .json")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded {len(results)} data records")
    
    # Add observation and action labels if they don't exist
    for result in results:
        if 'obs' in result and not any(f'observation_k{k}_label' in result for k in range(start_idx, end_idx + 1)):
            observations = result['obs']
            for k in range(start_idx, end_idx + 1):
                # Create observation labels
                bob_k_label = [process_observation(obs) for obs in observations]
                result[f'observation_k{k}_label'] = bob_k_label
    
    # Print analysis parameters
    print(f"Analysis index range: {start_idx}-{end_idx}, value type: {args.value_type}")
    print(f"Change rate threshold: {args.threshold}, stable count: {args.stable_count}")
    
    # Start analysis
    avg_change_rates = analyze_change_rates(
        results, 
        args.env, 
        output_path, 
        start_idx, 
        end_idx, 
        args.value_type,
        args.threshold,
        args.stable_count
    )
    
    print("Analysis complete, results saved to:", output_path)

if __name__ == "__main__":
    main()