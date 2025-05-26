import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from aria.data import DummyDataset
import copy
import pdb
import threading
from typing import Tuple
import random
import copy
import time
import wandb
import os
import requests
from requests.exceptions import Timeout

os.environ["NCCL_TIMEOUT"] = "18000000"
def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class OnlineReinforceTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    lm_lr: float = 1e-5,\
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = float(lm_lr))
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.step = 0
        self.actor_step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)

    def rm_request(self, obs: str="", action: str=""):
        url = 'http://hostname:port/generate'
        # request
        data = {
            'index': 1,
            'obs': obs,
            'action': action,
            }
        response = requests.post(url, json=data)
        print(response.json())
        
        return response.json()['response']

    def actor_loss(self, observation, pi_action, advantage, **kwargs):
        action = pi_action
        log_prob = self.agent.get_log_prob(observation, action)
        advantage = torch.Tensor(advantage).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        #in the case where a baseline is used
        if isinstance(log_prob, Tuple):
            values, log_prob, mask = log_prob
            values = values.squeeze(-1)
            advantage = advantage.reshape(-1, 1).broadcast_to(values.size())
            value_loss = torch.mean(((advantage - values)*mask)**2)
            with torch.no_grad():
                residual_advantage = advantage - values
            pg_loss = -torch.mean(torch.sum(residual_advantage*log_prob*mask, dim = 1))

        else:
            advantages = advantage.flatten()
            values = torch.zeros_like(advantages)
            residual_advantage = torch.zeros_like(advantages)
            pg_loss = -torch.mean(log_prob.flatten()*advantages)
            value_loss = torch.zeros_like(pg_loss, requires_grad=True) * 0

        advantages = advantage.flatten()
        self.accelerator.backward(pg_loss+value_loss)
        advantages = advantages.detach().cpu()
        return {"pg.loss": pg_loss.detach().cpu().item(),
                "advantages.mean": advantages.mean(),
                "advantages.max": torch.max(advantages),
                "advantages.min": torch.min(advantages),
                "advantages.std": torch.std(advantages)}

    def update(self, replay_buffer, no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        info.update(dict_mean(info_list))
        
        if not no_update_actor:
            torch.cuda.empty_cache()
            print(">>> Updating actor")
            
            # set batchsize for llm actor
            action_bsize =  replay_buffer.batch_size
            info_list = []            
            all_data = [replay_buffer.sample(1) for _ in range(len(replay_buffer))]
            for d in all_data:
                for k, v in d.items():
                    d[k] = v[0]
                    
            for epoch in range(self.actor_epochs):
                dataloader = DataLoader(
                    DummyDataset(all_data), 
                    batch_size=action_bsize,
                    shuffle=True
                )
                dataloader = self.accelerator.prepare(dataloader)

                accum_step = 0
                epoch_info_list = []
                
                self.lm_optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(dataloader):
                    with torch.no_grad():
                        print(f"cur obs: {len(batch['observation'])}")
                        pi_action = self.agent.get_action(batch["observation"])
                        # pi_action = batch["action"]
                        advantages = []
                        for obs, action in zip(batch["observation"], pi_action):
                            advantages.append(self.rm_request(obs, action))

                    # min-max normalize
                    # advantages_tensor = torch.Tensor(advantages)
                    # min_val, max_val = torch.min(advantages_tensor), torch.max(advantages_tensor)
                    # if max_val == min_val:
                        # advantages_normalized = torch.one_like(advantages_tensor)*0.5
                    # else:
                        # advantages_normalized = (advantages_tensor - min_val) / (max_val - min_val)
                    
                    # advantages = advantages_normalized.tolist()
                    
                    self.accelerator.wait_for_everyone()
                    batch_info = self.actor_loss(**batch, pi_action=pi_action, advantage=advantages)
                    epoch_info_list.append(batch_info)
                    
                    if self.accelerator.is_main_process:
                        print(f"Step {self.step}, Epoch {epoch} Completed, "
                            f"Info: {dict_mean(epoch_info_list)}")
                    
                    accum_step += 1
            
                    if accum_step == self.grad_accum_steps or batch_idx == len(dataloader) - 1:
                        self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                        self.lm_optimizer.step()
                        self.actor_step += 1
                        batch_update_info = dict_mean(epoch_info_list[-accum_step:])
                        info_list.append(batch_update_info)
                    
                        if self.accelerator.is_main_process:
                            actor_metrics = {"actor/" + k: v for k, v in batch_update_info.items()}
                            actor_metrics["actor_updates"] = self.actor_step
                            wandb.log(actor_metrics)
                        
                        accum_step = 0
                        self.lm_optimizer.zero_grad()
            info.update(dict_mean(info_list))
            
        return info

    def save(self, path):
        torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent
