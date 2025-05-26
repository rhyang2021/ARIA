from aria.data import DummyDataset,  ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from aria.algorithms.trainer import ReinforceTrainer
import wandb
import threading
import os
import torch
import time
import json
def offpolicy_train(
                agent,
                tokenizer,
                accelerator,
                warmup_iter: int = 20,
                batch_size: int = 2,
                grad_accum_steps: int = 1,
                lm_lr: float = 1e-5,
                use_wandb: bool = False,
                data_load_path: str = '',
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 25,
                agent_type: str = "offline_reinforce",
                **kwargs):
    trainer = ReinforceTrainer(agent=agent,
                                accelerator=accelerator,
                                tokenizer=tokenizer,
                                lm_lr = lm_lr,
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
   
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            print("Loading from checkpoint")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)
    agent.prepare()
   
    # main training loop
    with open(f"{data_load_path}", "r") as f:
        replay_buffer = json.load(f)

    info = {}
    accelerator.wait_for_everyone()
    print("Training")
    info.update(trainer.update(replay_buffer, batch_size))

    if save_path is not None and accelerator.is_main_process:
        print("Saving")
        trainer.save(os.path.join(save_path, 'trainer.pt'))
        torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))