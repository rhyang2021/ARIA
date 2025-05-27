import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os


def save_dict_to_json(data_dict, filename):
    """Save dictionary to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)


def get_values(results, index, agent='alice', value_type='advantage'):
    """
    Calculate Q-values or advantage values for the agent
    
    Parameters:
    - results: List of results
    - index: k index to calculate
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
        gamma = 0.95  # Bob doesn't use discount factor
    
    # Calculate state value function V(s)
    v_values = {}
    
    if agent == 'alice':
        # Alice's state representation
        for result in results:
            alice_k_label = result[f'alice_k{index}_label']
            bob_k_label = result[f'bob_k{index}_label']
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
            alice_k_label = result[f'alice_k{index}_label']
            bob_k_label = result[f'bob_k{index}_label']
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
        # Alice's Q-value calculation
        for result in results:
            alice_k_label = result[f'alice_k{index}_label']
            bob_k_label = result[f'bob_k{index}_label']
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
                
                # For Q-values, may apply discount factor; for advantage values, don't apply
                if value_type == 'q':
                    q_values[key].append(reward * pow(gamma, min_length-len(sequence)))
                else:
                    q_values[key].append(reward)
    else:
        # Bob's Q-value calculation
        for result in results:
            alice_k_label = result[f'alice_k{index}_label']
            bob_k_label = result[f'bob_k{index}_label']
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
    
    # Calculate average Q-values
    for key, values in q_values.items():
        q_values[key] = sum(values) / len(values) if values else 0
    
    # If only Q-values are needed, return directly
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


def create_result_dict(results, values_dict, agent='alice'):
    """
    Create result dictionary, mapping action_key to values under each index
    
    Parameters:
    - results: List of results
    - values_dict: Value dictionary for each index
    - agent: Agent name ('alice' or 'bob')
    
    Returns:
    - Result dictionary
    """
    result_dict = {}
    
    if agent == 'alice':
        # Alice's result dictionary creation
        for result in results:
            for index in values_dict.keys():
                cur_vals = values_dict[index]
                alice_actions = result['alice_ids']
                bob_actions = result['bob_ids']
                alice_k_label = result[f'alice_k{index}_label']
                bob_k_label = result[f'bob_k{index}_label']
                
                actions = []
                sequence = []
                
                for i in range(min(len(bob_k_label), len(alice_k_label)) - 1):
                    if i == 0:
                        actions.append(str(alice_actions[i]))
                        sequence.append(str(alice_k_label[i]))
                    sequence.append(str(bob_k_label[i]))
                    sequence.append(str(alice_k_label[i+1]))
                    actions.append(str(bob_actions[i]))
                    actions.append(str(alice_actions[i+1]))
                    
                    label_key = "-".join(sequence)
                    action_key = "-".join(actions)
                    
                    if label_key in cur_vals:
                        if action_key not in result_dict:
                            result_dict[action_key] = {}
                        result_dict[action_key][index] = cur_vals[label_key]
    else:
        # Bob's result dictionary creation
        for result in results:
            for index in values_dict.keys():
                cur_vals = values_dict[index]
                alice_actions = result['alice_ids']
                bob_actions = result['bob_ids']
                alice_k_label = result[f'alice_k{index}_label']
                bob_k_label = result[f'bob_k{index}_label']
                
                min_length = min(len(alice_k_label), len(bob_k_label))
                
                # Build all possible action sequences
                for i in range(min_length):
                    # Build label_key (state-action pair sequence)
                    label_sequence = []
                    for j in range(i+1):
                        label_sequence.append(str(alice_k_label[j]))
                        label_sequence.append(str(bob_k_label[j]))
                    
                    label_key = "-".join(label_sequence)
                    
                    # Build action_key (actual dialogue ID sequence)
                    action_sequence = []
                    for j in range(i+1):
                        action_sequence.append(str(alice_actions[j]))
                        action_sequence.append(str(bob_actions[j]))
                    
                    action_key = "-".join(action_sequence)
                    
                    if label_key in cur_vals:
                        if action_key not in result_dict:
                            result_dict[action_key] = {}
                        result_dict[action_key][index] = cur_vals[label_key]
    
    return result_dict


def calculate_change_rates(result_dict):
    """Calculate change rate for each action_key from index to index+1"""
    change_rates = {}
    
    for _, index_values in result_dict.items():
        indices = sorted(index_values.keys())
        for i in range(len(indices) - 1):
            current_index = indices[i]
            next_index = indices[i + 1]
            
            if next_index == current_index + 1:
                current_value = index_values[current_index]
                next_value = index_values[next_index]
                change = next_value - current_value
                
                if current_index not in change_rates:
                    change_rates[current_index] = []
                change_rates[current_index].append(abs(change))
    
    # Calculate average change rate for each index
    avg_change_rates = {}
    for idx, changes in change_rates.items():
        avg_change_rates[idx] = np.mean(changes) if changes else 0
    
    return avg_change_rates


def plot_change_rates(avg_change_rates, filename, title, color='#82A9D9', threshold=0.01, stable_count=20):
    """
    Plot average change rate from index to index+1, and mark k values that are stable below threshold
    
    Parameters:
    - avg_change_rates: Change rate dictionary
    - filename: Output filename
    - title: Chart title
    - color: Line color, default blue (#82A9D9)
    - threshold: Change rate threshold
    - stable_count: How many consecutive points below threshold to consider stable
    """
    indices = sorted(avg_change_rates.keys())
    avg_changes = [avg_change_rates[idx] for idx in indices]
    
    # Define base font size
    fs = 18  # Base font size
    
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    
    # Main line
    plt.plot(indices, avg_changes, marker='o', linestyle='-', color=color, linewidth=2.5)
    plt.title(title, fontsize=fs+4)
    plt.xlabel('Index', fontsize=fs)
    plt.ylabel('Average Change Rate', fontsize=fs)
    
    # Add threshold horizontal line
    plt.axhline(y=threshold, color='purple', linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Threshold = {threshold}')
    
    # Set x-axis ticks, show every 5th
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
    
    plt.xticks(xticks, fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    
    # Remove top and right borders, and thicken left and bottom borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Add grid lines - only horizontal dashed lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add labels only for specific points - using larger font
    for i, value in enumerate(avg_changes):
        idx = indices[i]
        if idx in xticks:
            plt.text(idx, value, f'{value:.2f}', ha='center', 
                    va='bottom' if value > 0 else 'top',
                    fontsize=fs-4)
    
    # Find max and min values, and add special markers
    if avg_changes:
        max_idx = indices[np.argmax(avg_changes)]
        min_idx = indices[np.argmin(avg_changes)]
        max_val = max(avg_changes)
        min_val = min(avg_changes)
        
        plt.scatter(max_idx, max_val, color='#EE8535', s=120, marker='*', 
                    zorder=5, label=f'Max: {max_val:.2f} at k={max_idx}')
        plt.scatter(min_idx, min_val, color='#C1D5EC', s=120, marker='*', 
                    zorder=5, label=f'Min: {min_val:.2f} at k={min_idx}')
    
    # Find k value that is stable below threshold (stable_count consecutive points below threshold)
    stable_below_threshold_idx = None
    stable_below_threshold_val = None
    
    for i in range(len(avg_changes) - stable_count + 1):
        if all(avg_changes[i+j] < threshold for j in range(stable_count)):
            stable_below_threshold_idx = indices[i]
            stable_below_threshold_val = avg_changes[i]
            break
    
    # If stable below threshold k is found, mark it
    if stable_below_threshold_idx is not None:
        plt.scatter(stable_below_threshold_idx, stable_below_threshold_val, color='purple', s=140, marker='D', 
                    edgecolor='black', linewidth=1.5, zorder=10, 
                    label=f'Stable below {threshold}: k={stable_below_threshold_idx} ({stable_below_threshold_val:.2f})')
        
        # Add vertical line to mark this k value
        plt.axvline(x=stable_below_threshold_idx, color='purple', linestyle=':', alpha=0.5)
        
        print(f"Starting k value stable below threshold {threshold}: {stable_below_threshold_idx}, change rate: {stable_below_threshold_val:.4f}")
    
    # Set legend
    plt.legend(fontsize=fs-4, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Also save as PDF format
    pdf_filename = filename.replace('.png', '.pdf')
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
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
    
    save_dict_to_json(data_to_save, filename.replace('.png', '.json'))

def analyze_agent(results, agent, env_name, output_dir, start_idx=2, end_idx=30, value_type='advantage', threshold=0.01, stable_count=17):
    """
    Analyze agent's change rates
    
    Parameters:
    - results: List of results
    - agent: Agent name ('alice' or 'bob')
    - env_name: Environment name
    - output_dir: Output directory
    - start_idx: Starting index, default 2
    - end_idx: Ending index, default 30
    - value_type: Value type to analyze ('q' or 'advantage')
    - threshold: Change rate threshold
    - stable_count: How many consecutive points below threshold to consider stable
    
    Returns:
    - avg_change_rates: Average change rate dictionary
    """
    print(f"Analyzing {agent.capitalize()}'s change rates for {env_name} environment...")
    
    # Calculate values for each index
    values_dict = {}
    for i in tqdm(range(start_idx, end_idx + 1), desc=f"Calculating {agent.capitalize()}'s {value_type} values"):
        values_dict[i] = get_values(results, i, agent, value_type)
    
    # Create result dictionary
    result_dict = create_result_dict(results, values_dict, agent)
    
    # Calculate and plot change rates
    avg_change_rates = calculate_change_rates(result_dict)
    
    # Set color: Alice uses blue, Bob uses red
    color = 'blue' if agent == 'alice' else 'red'
    
    plot_change_rates(
        avg_change_rates, 
        os.path.join(output_dir, f"{env_name}_{agent}_{value_type}_change_rates.png"),
        f"{agent.capitalize()}'s {value_type.capitalize()} Change Rate from Index to Index+1 ({env_name})",
        color=color,
        threshold=threshold,
        stable_count=stable_count
    )
    
    # Identify k with minimum change rate (most stable index)
    if avg_change_rates:
        best_k = min(avg_change_rates.items(), key=lambda x: x[1])[0]
        print(f"{agent.capitalize()}'s most stable k value: {best_k}, change rate: {avg_change_rates[best_k]:.4f}")
    
    return avg_change_rates

def compare_alice_bob_change_rates(alice_changes, bob_changes, env_name, output_dir, value_type='advantage', threshold=0.01, stable_count=17):
    """
    Compare change rates of Alice and Bob, and plot their average change rates, marking k values stable below threshold
    
    Parameters:
    - alice_changes: Alice's change rate dictionary
    - bob_changes: Bob's change rate dictionary
    - env_name: Environment name
    - output_dir: Output directory
    - value_type: Value type ('q' or 'advantage')
    - threshold: Change rate threshold
    - stable_count: How many consecutive points below threshold to consider stable, default 17
    """
    if alice_changes and bob_changes:
        # Find common indices
        common_indices = sorted(set(alice_changes.keys()).intersection(set(bob_changes.keys())))
        
        if not common_indices:
            print("Alice and Bob have no common indices, cannot compare")
            return
        
        alice_change_vals = [alice_changes[idx] for idx in common_indices]
        bob_change_vals = [bob_changes[idx] for idx in common_indices]
        
        # Calculate average change rates
        avg_change_vals = [(a + b) / 2 for a, b in zip(alice_change_vals, bob_change_vals)]
        
        # Set font size and colors
        fs = 20  # Base font size
        colors = ["#EE8535","#BECB51","#C1D5EC","#82A9D9"]  # Color definitions
        
        # Create comparison chart: Alice, Bob, and average in one chart
        plt.figure(figsize=(12, 8))
        plt.style.use('default')
        
        plt.plot(common_indices, alice_change_vals, marker='o', linestyle='-', color=colors[0], 
                linewidth=3, label='Alice')
        plt.plot(common_indices, bob_change_vals, marker='s', linestyle='-', color=colors[1], 
                linewidth=3, label='Bob')
        plt.plot(common_indices, avg_change_vals, marker='^', linestyle='-', color=colors[3], 
                linewidth=3, label='Average (Alice+Bob)/2')
        
        # Add threshold horizontal line
        plt.axhline(y=threshold, color='purple', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Threshold = {threshold}')
        
        plt.title(f'Alice, Bob, and Average {value_type.capitalize()} Change Rates ({env_name})', fontsize=fs)
        plt.xlabel('Index', fontsize=fs)
        plt.ylabel(f'Average {value_type.capitalize()} Change Rate', fontsize=fs)
        
        # Remove top and right borders, and thicken left and bottom borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        # Add grid lines - only horizontal dashed lines
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Set x-axis ticks, show every 5th
        xticks = []
        if common_indices:
            xticks.append(common_indices[0])  # Add first index
            
            # Add every 5th index
            for idx in common_indices:
                if idx % 5 == 0:
                    xticks.append(idx)
            
            # Ensure last index is included
            if common_indices[-1] not in xticks:
                xticks.append(common_indices[-1])
            
            # Remove duplicates and sort
            xticks = sorted(set(xticks))
        
        plt.xticks(xticks, fontsize=fs-2)
        plt.yticks(fontsize=fs-2)
        
        # Find min and max of average change rates
        min_avg_idx = common_indices[np.argmin(avg_change_vals)]
        max_avg_idx = common_indices[np.argmax(avg_change_vals)]
        min_avg_val = min(avg_change_vals)
        max_avg_val = max(avg_change_vals)
        
        # Mark max and min of average change rates
        plt.scatter(min_avg_idx, min_avg_val, color=colors[3], s=150, marker='*', 
                    edgecolor='black', linewidth=1.5, zorder=10, 
                    label=f'Min Avg: {min_avg_val:.2f} at k={min_avg_idx}')
        plt.scatter(max_avg_idx, max_avg_val, color=colors[0], s=150, marker='*', 
                    edgecolor='black', linewidth=1.5, zorder=10, 
                    label=f'Max Avg: {max_avg_val:.2f} at k={max_avg_idx}')
        
        # Find k value stable below threshold (stable_count consecutive points below threshold)
        stable_below_threshold_idx = None
        stable_below_threshold_val = None
        
        for i in range(len(avg_change_vals) - stable_count + 1):
            if all(avg_change_vals[i+j] < threshold for j in range(stable_count)):
                stable_below_threshold_idx = common_indices[i]
                stable_below_threshold_val = avg_change_vals[i]
                break
        
        # If stable below threshold k is found, mark it
        if stable_below_threshold_idx is not None:
            plt.scatter(stable_below_threshold_idx, stable_below_threshold_val, color='purple', s=180, marker='D', 
                        edgecolor='black', linewidth=2, zorder=10, 
                        label=f'Stable below {threshold}: k={stable_below_threshold_idx} ({stable_below_threshold_val:.2f})')
            
            # Add vertical line to mark this k value
            plt.axvline(x=stable_below_threshold_idx, color='purple', linestyle=':', alpha=0.6)
            
            print(f"Starting k value stable below threshold {threshold}: {stable_below_threshold_idx}, average change rate: {stable_below_threshold_val:.4f}")
            print(f"(Consecutive {stable_count} points all below threshold)")
        else:
            print(f"No k value found stable below threshold {threshold} (requires consecutive {stable_count} points all below threshold)")
        
        # Add value labels (only for average values)
        for i, val in enumerate(avg_change_vals):
            # Only add labels for specific points: max, min, stable threshold point, and x-tick points
            idx = common_indices[i]
            if idx == min_avg_idx or idx == max_avg_idx or idx == stable_below_threshold_idx or idx in xticks:
                plt.text(idx, val, f'{val:.2f}', ha='center', va='bottom', color=colors[3], fontsize=fs-7)
        
        plt.legend(fontsize=fs-7, loc='upper center', ncol=2, handlelength=0.8, 
                  bbox_to_anchor=(0.64, 1.0))
        plt.tight_layout()
        
        # Save image
        plt.savefig(os.path.join(output_dir, f"{env_name}_alice_bob_avg_{value_type}_change_rates.png"), dpi=300, bbox_inches='tight')
        
        # Also save as PDF format
        pdf_filename = os.path.join(output_dir, f"{env_name}_alice_bob_avg_{value_type}_change_rates.pdf")
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Common best k value (minimum change rate): {min_avg_idx}, average change rate: {min_avg_val:.4f}")
        
        # Save comparison data
        comparison_data = {
            "alice_change_rates": {str(k): v for k, v in alice_changes.items()},
            "bob_change_rates": {str(k): v for k, v in bob_changes.items()},
            "average_change_rates": {str(idx): val for idx, val in zip(common_indices, avg_change_vals)},
            "best_k": {
                "index": int(min_avg_idx),
                "value": float(min_avg_val)
            }
        }
        
        # Add stable threshold information
        if stable_below_threshold_idx is not None:
            comparison_data["stable_below_threshold"] = {
                "threshold": threshold,
                "stable_count": stable_count,
                "index": int(stable_below_threshold_idx),
                "value": float(stable_below_threshold_val)
            }
        
        save_dict_to_json(comparison_data, os.path.join(output_dir, f"{env_name}_alice_bob_avg_{value_type}_change_rates.json"))

def main():
    # Create command line argument parser
    parser = argparse.ArgumentParser(description="Analyze change rates of Alice and Bob in different environments")
    parser.add_argument("--env", default="negotiation", choices=["bargaining", "negotiation"],
                        help="Specify environment name (bargaining or negotiation)")
    parser.add_argument("--data_path", default="/path/to/your/dataset", type=str,
                        help="Path to JSONL data file")
    parser.add_argument("--output_dir",type=str, default="/path/to/your/dataset", 
                        help="Output directory path")
    parser.add_argument("--index_range", type=str, default="2-50",
                        help="Specify index range to analyze, format 'start_index-end_index', default '2-100'")
    parser.add_argument("--value_type", choices=["q", "advantage"], default="q",
                        help="Specify value type to analyze (q or advantage)")
    
    args = parser.parse_args()
    
    # Parse index range
    try:
        start_idx, end_idx = map(int, args.index_range.split('-'))
        if start_idx < 2 or end_idx > 100 or start_idx > end_idx:
            raise ValueError
    except:
        print("Index range format incorrect, should be 'start_index-end_index', e.g. '2-100'")
        print("Start index cannot be less than 2, end index cannot be greater than 100")
        return
    
    # Create output directory
    output_path = f"{args.output_dir}/{args.env}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    print(f"Loading data: {args.data_path}")
    data_path = f"{args.data_path}/llama3-8b_{args.env}_with_labels_k2_to_k100.jsonl"
    with open(data_path) as f:
        results = [json.loads(line) for line in f]
    
    print(f"Analyzing index range: {start_idx}-{end_idx}, value type: {args.value_type}")
    
    # Always analyze both Alice and Bob
    print("Analyzing Alice's change rates...")
    alice_changes = analyze_agent(results, "alice", args.env, output_path, 
                                  start_idx, end_idx, args.value_type)
    print("Alice analysis completed")
    
    print("Analyzing Bob's change rates...")
    bob_changes = analyze_agent(results, "bob", args.env, output_path, 
                                start_idx, end_idx, args.value_type)
    print("Bob analysis completed")
    
    # Generate combined chart
    print("Generating combined chart of Alice, Bob, and average change rates...")
    compare_alice_bob_change_rates(alice_changes, bob_changes, args.env, output_path, args.value_type)
    print("Combined chart generation completed")
    
    print("Analysis completed, results saved to:", output_path)


if __name__ == "__main__":
    main()