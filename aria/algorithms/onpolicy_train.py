from aria.environment import batch_interact_environment
from aria.data import DummyDataset,  ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from aria.algorithms.trainer import OnlineReinforceTrainer
import wandb
import threading
import os
import torch
import time
import pdb
import random
import json
random.seed(42)

def get_log_probs_with_unwrapped_model(model, tokenizer, obs, act, max_length=256, device=None):
    template = getattr(model, 'template', None)
    if template is not None:
        obs = template.replace("{obs}", obs)
    
    obs_ids = tokenizer(obs, return_tensors='pt', padding='max_length', 
                    max_length=128, truncation=True).to(device)
    action_ids = tokenizer(act, return_tensors='pt', padding='max_length', 
                        max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)
        attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        values = None
        if isinstance(outputs, tuple):
            values, outputs = outputs
        
        softmax = torch.nn.Softmax(dim=-1)
        prediction_probs = softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1],action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        
        per_token_log_probs = torch.log(selected_prediction_probs) * action_ids["attention_mask"]
        
        per_token_log_probs = per_token_log_probs.masked_fill(action_ids["attention_mask"] == 0, -100.0)
        
        if values is not None:
            padded_values = values[:, obs_ids["attention_mask"].size(1)-1:-1]
            if padded_values.size(1) < max_length:
                padding = torch.zeros((padded_values.size(0), max_length - padded_values.size(1)),
                            device=padded_values.device)
                padded_values = torch.cat([padded_values, padding], dim=1)
            return (padded_values, per_token_log_probs, action_ids["attention_mask"])
        else:
            if per_token_log_probs.size(1) < max_length:
                padding = torch.full((per_token_log_probs.size(0), max_length - per_token_log_probs.size(1)), 
                            -100.0, device=per_token_log_probs.device)
                per_token_log_probs = torch.cat([per_token_log_probs, padding], dim=1)
            return per_token_log_probs

def onpolicy_train(
                agent,
                tokenizer,
                accelerator,
                warmup_iter: int = 20,
                rollout_size: int = 50,
                batch_size: int = 2,
                capacity: int = 500000,
                iterations: int = 10,
                grad_accum_steps: int = 1,
                do_sample: bool = False,
                temperature: float = 2.0,
                lm_lr: float = 1e-5,
                use_wandb: bool = False,
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 5,
                eval_freq: int = 5,
                agent_type: str = "online_reinforce",
                env_type: str = "single",
                **kwargs):

    trainer = OnlineReinforceTrainer(agent=agent,
                            accelerator=accelerator,
                            tokenizer=tokenizer,
                            lm_lr = lm_lr,
                            actor_epochs = actor_epochs,
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm)
    
    all_trajectories = []
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            print("Loading from checkpoint")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)
    agent.prepare()
    # main training loop
    print(">>>start iterations")
    for i in range(iterations):
        replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
        
        print(">>>Interacting with Environment")
        if accelerator.is_main_process:
            torch.save(True, os.path.join(save_path, 'sampling_in_progress.flag'))
        accelerator.wait_for_everyone()
        
        print(">>>Load Replay Buffer")
        if accelerator.is_main_process:
            info = {}
            data = batch_interact_environment(agent=agent, 
                                            iteration=i, 
                                            sample_size=rollout_size,
                                            agent_type=agent_type.lower()) 
            
            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),
                    "rollout.reward.max": np.max([d["reward"] for d in data]),
                    "rollout.reward.min": np.min([d["reward"] for d in data]),
                    "rollout.traj_reward.mean": np.mean([d["trajectory_reward"] for d in data])})
            
            wandb.log(info)
            for t in data:
                replay_buffer.insert(**t)

            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))

            if os.path.exists(os.path.join(save_path, 'sampling_in_progress.flag')):
                os.remove(os.path.join(save_path, 'sampling_in_progress.flag'))
            torch.save(True, os.path.join(save_path, 'sampling_complete.flag'))
        else:
            info = {}
            while not os.path.exists(os.path.join(save_path, 'sampling_complete.flag')):
                time.sleep(10) 
        
        accelerator.wait_for_everyone()
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        if accelerator.is_main_process and os.path.exists(os.path.join(save_path, 'sampling_complete.flag')):
            os.remove(os.path.join(save_path, 'sampling_complete.flag'))
        
        print(">>>Training")
        if 'filtered' in agent_type.lower():
            filtered_buffer= ReplayBuffer(batch_size=batch_size, capacity=capacity)
            episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
            cutoff = np.quantile(episode_rewards, 1 - 0.1)
            print("Episode Reward Cutoff: ", cutoff)
            filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
            data = sum(filtered_trajectories, [])
            for d in data:
                filtered_buffer.insert(**d)
            info.update(trainer.update(filtered_buffer, no_update_actor = (i < warmup_iter)))
        else:
            info.update(trainer.update(replay_buffer, no_update_actor = (i < warmup_iter)))
        
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        
        if (i+1) % 5 == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, f'trainer-{i}.pt'))
            torch.save(replay_buffer, os.path.join(save_path, f'replay_buffer-{i}.pt'))
        accelerator.wait_for_everyone()
    