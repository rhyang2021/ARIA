import json

game_id = 0  # Please ensure this variable is defined

with open("../../dataset/llama3-8b_twenty_questions_embedding_msgs.jsonl", 'r') as infile, open("llama3-8b_twenty_questions_embedding_msgs_game.jsonl", "w") as outfile:
    for line_num, line in enumerate(infile, 1):
        try:
            data = json.loads(line)
            data0 = {}
            data0["game_id"] = game_id
            data0["action_ids"] = [f"t_full_{game_id}_{i}" for i in range(len(data["actions"]))]
            data0 = {**data0, **data}
            game_id += 1
            outfile.write(json.dumps(data0) + "\n")
        except json.JSONDecodeError as e:
            print(f"Error at line {line_num}: {e}")
            print(f"First 100 characters of problematic line: {line[:100]}...")
            # Can choose to skip this line or stop processing
            continue  # Skip this line and continue processing


# Open source file for reading and target file for writing
with open('llama3-8b_twenty_questions_embedding_msgs_game.jsonl', 'r') as infile, open('action_list_with_id.jsonl', 'w') as outfile:
    for line in infile:
        # Parse each line of JSON
        data = json.loads(line.strip())
        
        # Get game ID and Alice/Bob's actions
        # Get game ID and Alice/Bob's data
        game_id = data["game_id"]
        actions = data["actions"]
        ids = data["action_ids"]
        embeddings = data["action_embeddings"]
        
        
        # Process Bob's actions
        for action, action_id, action_embedding in zip(actions, ids, embeddings):
            new_entry = {
                "game_id": game_id,
                "action_id": action_id,
                "game_action_id": action_id,
                "action": action,
                "action_embedding" : action_embedding
            }
            # Write Bob's action
            outfile.write(json.dumps(new_entry) + '\n')