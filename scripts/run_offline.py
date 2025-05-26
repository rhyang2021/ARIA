import torch
import transformers
from tqdm import tqdm 
from aria.models import ReinforceAgent
from aria.algorithms import offpolicy_train
from aria.utils import colorful_print
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

CONFIG_NAME = "reinforce_llm"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    # load decision model
    print(">>> Using Offline Reinforce agent with LLM")
    agent = ReinforceAgent(device=device, accelerator=accelerator, 
                            policy_lm=config.policy_lm, 
                            cache_dir=config.cache_dir, 
                            max_new_tokens=config.max_new_tokens,
                            use_lora=config.use_lora)

    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
        agent.model.load_state_dict(state_dict)

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    offpolicy_train(agent = agent,
                tokenizer = tokenizer,
                accelerator = accelerator,
                **config)


if __name__ == "__main__":
    main()