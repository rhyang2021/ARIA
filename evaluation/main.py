import torch
import transformers
from tqdm import tqdm
from twenty_questions import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv
from guess_my_city import GuessMyCityEnv, BatchedGuessMyCityEnv
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import time
import hydra
import json
import pdb
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from llm_base import llm_openai, vllm
transformers.logging.set_verbosity_error()


def main(args):
    # load environment
    if args.env_name == "twenty_questions":
        env = TwentyQuestionsEnv()
        eval_env = env
    elif args.env_name == "guess_my_city":
        env = GuessMyCityEnv()
        eval_env = env
    else:
        raise NotImplementedError("Environment not implemented.")
    
    output_dir = f"{args.output_dir}/{args.env_name}/{int(time.time())}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    numbers = list(range(0, 90)) 
    for variation in tqdm(range(args.repeat), desc="Generating Trajectories"):
        variation_to_save = f"{output_dir}/variation-{variation}"
        trajectories = []
        done = False
        # obs = reset_to(env, 69)
        # init = env.reset()
        init = env.reset(random.choice(numbers))
        next_obs = init
        steps = 0
        while not done:
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            action = vllm(prompt = next_obs, model=args.model_id, port=args.model_port, temperature=0)
            _return = env._step(question=action)
            next_obs, answer, r, done = _return
            new_item = {"instruction": init[0]['content'],
                        "curr_world": env.curr_word[0] if args.env_name == "twenty_questions" else env.curr_word,
                        "question": action,
                        "answer": answer,
                        "next_observation": next_obs[0]['content'],
                        "reward": r,
                        "done": done}
            trajectories.append(new_item)
            with open(f'{variation_to_save}.jsonl', 'a') as f: 
                f.write(json.dumps(new_item) + "\n") 

            if steps > env.max_conversation_length:
                break
    

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import random
    random.seed(42)

    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--env_name", type=str, default="twenty_questions", help="Dataset for trajectories")
    parser.add_argument("--model_id", type=str, default="llama3-8B")
    parser.add_argument("--model_port", type=int, default=8035)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="../results")
    args = parser.parse_args()
    
    main(args)
