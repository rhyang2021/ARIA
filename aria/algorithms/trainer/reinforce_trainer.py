import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from aria.data import DummyDataset
import copy
import threading
from typing import Tuple
import random
import copy
import pdb
import time
import wandb

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class ReinforceTrainer():
    def __init__(self, agent,\
                 accelerator,\
                tokenizer,\
                lm_lr: float = 1e-5,\
                grad_accum_steps: int = 8,\
                max_grad_norm: float=0.01,
                actor_epochs: int = 3):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.sft_coefficient = 0.5
        self.step = 0
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)


    def actor_loss(self, observation, action, reward, **kwargs):
        advantage = reward
        # action = [element for tup in action for element in tup]
        # observation = [element for tup in observation for element in tup]
        log_prob = self.agent.get_log_prob(observation, action)
        advantage = torch.Tensor(advantage).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype)
        sft_mask = (advantage == -1)
        rlhf_mask = ~sft_mask

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
            # Handle SFT loss (reward = -1)
            sft_advantages = torch.ones_like(advantage)
            sft_loss = -torch.mean(log_prob.flatten()[sft_mask] * sft_advantages.flatten()[sft_mask]) if torch.any(sft_mask) else torch.tensor(0.0, device=advantage.device)
            
            # Handle RLHF loss (other rewards)
            advantages = advantage.flatten()
            values = torch.zeros_like(advantages)
            residual_advantage = torch.zeros_like(advantages)
            rlhf_loss = -torch.mean(log_prob.flatten()[rlhf_mask] * advantages[rlhf_mask]) if torch.any(rlhf_mask) else torch.tensor(0.0, device=advantage.device)
            
            # Combine losses
            pg_loss = self.sft_coefficient * sft_loss + rlhf_loss
            value_loss = torch.zeros_like(pg_loss)

        advantages = advantage.flatten()
        self.accelerator.backward(pg_loss+value_loss)
        advantages = advantages.detach().cpu()
        return {"pg.loss": pg_loss.detach().cpu().item(),
                "sft.loss": sft_loss.detach().cpu().item(),
                "rlhf.loss": rlhf_loss.detach().cpu().item(),
                "values.loss": value_loss.detach().cpu().item(),
                "values.mean": values.mean(),
                "values.max": torch.max(values),
                "values.min": torch.min(values),
                "values.std": torch.std(values),
                "advantages.mean": advantages.mean(),
                "advantages.max": torch.max(advantages),
                "advantages.min": torch.min(advantages),
                "advantages.std": torch.std(advantages),
                "residual_advantages.mean": residual_advantage.mean(),
                "residual_advantages.max": torch.max(residual_advantage),
                "residual_advantages.min": torch.min(residual_advantage),
                "residual_advantages.std": torch.std(residual_advantage),}

    def update(self, replay_buffer, batch_size):
        self.step += 1
        info = {}
        info_list = []
        
        # update actor
        print(">>>updating actor")
        # batchsize for the actor
        action_bsize = 2 if 'mistral' in self.agent.policy_lm else batch_size
        
        for epoch in range(self.actor_epochs):
            dataloader = DataLoader(
                DummyDataset(replay_buffer), 
                batch_size=action_bsize,
                shuffle=True
            )
            dataloader = self.accelerator.prepare(dataloader)
            
            accum_step = 0
            epoch_info_list = []
            
            self.lm_optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(dataloader):
                batch_info = self.actor_loss(**batch)
                epoch_info_list.append(batch_info)
                
                if self.accelerator.is_main_process:
                    print(f"Step {self.step}, Epoch {epoch}, Batch {batch_idx}, "
                        f"PG Loss: {batch_info['pg.loss']:.6f}, "
                        f"Value Loss: {batch_info['values.loss']:.6f}, "
                        f"Advantage Mean: {batch_info['advantages.mean']:.6f}")
                
                accum_step += 1
                
                if accum_step == self.grad_accum_steps or batch_idx == len(dataloader) - 1:
                    self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.lm_optimizer.step()
                    
                    batch_update_info = dict_mean(epoch_info_list[-accum_step:])
                    info_list.append(batch_update_info)
                    if self.accelerator.is_main_process:
                        wandb.log(batch_update_info)

                    accum_step = 0
                    self.lm_optimizer.zero_grad()

                
        if info_list:
            info.update(dict_mean(info_list))
        
        return info

    def save(self, path):
        torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent
