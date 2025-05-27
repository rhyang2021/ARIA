import json

game_id = 0
with open("../../dataset/llama3-8b_negotiation_embedding_msgs.jsonl", 'r') as infile, open("llama3-8b_negotiation_embedding_msgs_game.jsonl", "w") as outfile:
    for line in infile:
        data = json.loads(line)
        data0 = {}
        data0["game_id"] = game_id
        data0 = {**data0, **data}
        game_id += 1
        outfile.write(json.dumps(data0) + "\n")

# Open source file for reading and target file for writing
with open('llama3-8b_negotiation_embedding_msgs_game.jsonl', 'r') as infile, open('action_list_with_id.jsonl', 'w') as outfile:
    for line in infile:
        # Parse each line of JSON
        data = json.loads(line.strip())
        
        # Get game ID and Alice/Bob's actions
        # Get game ID and Alice/Bob's data
        game_id = data["game_id"]
        alice_actions = data["alice_actions"]
        bob_actions = data["bob_actions"]
        alice_ids = data["alice_ids"]
        bob_ids = data["bob_ids"]
        alice_embeddings = data["alice_action_embeddings"]
        bob_actions = data["bob_actions"]
        bob_embeddings = data["bob_actions_embeddings"]
        
       
        # Process Alice's actions
        for action, action_id, action_embedding in zip(alice_actions, alice_ids, alice_embeddings):
            new_entry = {
                "game_id": game_id,
                "action_id": action_id,
                "game_action_id": f"_{game_id}_{action_id}",
                "action": action,
                "action_embedding" : action_embedding
            }
            # Write Alice's action
            outfile.write(json.dumps(new_entry) + '\n')  
        # Process Bob's actions
        for action, action_id, action_embedding in zip(bob_actions, bob_ids, bob_embeddings):
            new_entry = {
                "game_id": game_id,
                "action_id": action_id,
                "game_action_id": f"_{game_id}_{action_id}",
                "action": action,
                "action_embedding" : action_embedding
            }
            # Write Bob's action
            outfile.write(json.dumps(new_entry) + '\n')        
        
        
with open('llama3-8b_negotiation_embedding_msgs_game.jsonl', 'r') as infile, open('negotiation_with_formatted_ids.jsonl', 'w') as outfile:
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

print("Conversion completed, new file saved as negotiation_with_formatted_ids.jsonl")