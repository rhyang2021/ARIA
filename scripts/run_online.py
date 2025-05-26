import torch
import transformers
from tqdm import tqdm
from aria.models import OnlineReinforceAgent
from aria.algorithms import onpolicy_train
from aria.utils import colorful_print
import torch.nn as nn
import numpy as np 
import wandb
import os
import sys
import argparse
import yaml
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='ArCHer training script')
    parser.add_argument('--config-name', type=str, default='onlinereinforce_llm',
                        help='config name')
    parser.add_argument('--config-path', type=str, default='config/',
                        help='config path')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='distributed')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                        help='DeepSpeed config')

    args, unknown = parser.parse_known_args()
    
    # Load Config
    config_file = os.path.join(args.config_path, f"{args.config_name}.yaml")
    config = load_config(config_file)
    local_rank = args.local_rank
    if local_rank != -1:
        os.environ['LOCAL_RANK'] = str(local_rank)
    
    print(f">>> Running with LOCAL_RANK: {local_rank}")
    colorful_print(f">>> Configuration file: {args.config_name}.yaml <<<", fg='blue')
    colorful_print(yaml.dump(config), fg='red')
    
    try:
        from huggingface_hub import login
        if 'huggingface_token' in config and config['huggingface_token']:
            login(token=config['huggingface_token'])
    except:
        print(">>> Huggingface token not found.")
    
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=15))
    accelerator = Accelerator(kwargs_handlers=[timeout])
    device = accelerator.device
    
    # load decision model
    agent = OnlineReinforceAgent(device=device, accelerator=accelerator, 
                        temperature=config['temperature'], 
                        do_sample=config['do_sample'], 
                        policy_lm=config['policy_lm'], 
                        cache_dir=config.get('cache_dir', None), 
                        max_new_tokens=config['max_new_tokens'],
                        use_lora=config.get('use_lora', False),
                        eos_str=config.get('eos_str', None))
    tokenizer = agent.tokenizer
    if config.get('checkpoint_path') is not None:
        state_dict = torch.load(config['checkpoint_path'], map_location=device)['model_state_dict']
        agent.model.load_state_dict(state_dict)
    
    if config.get('use_wandb', False) and accelerator.is_main_process:
        wandb.login(key=config.get('wandb_key'))
        wandb.init(project=config['project_name'], 
                  name=config['run_name'], 
                  config=config)

    onpolicy_train(agent=agent,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    **config)


if __name__ == "__main__":
    main()
