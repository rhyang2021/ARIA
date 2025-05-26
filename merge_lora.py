import os
import argparse
from datetime import timedelta
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs

from aria.models import ReinforceAgent, OnlineReinforceAgent


def convert_checkpoint_to_hf_format(checkpoint_path, base_model_name, output_dir, use_lora=False, device="cuda"):
    """
    Load a model from checkpoint, convert it to Hugging Face format, and save it.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        base_model_name (str): Base model name, e.g., "mistralai/Mistral-7B-v0.1"
        output_dir (str): Directory to save the converted model
        use_lora (bool): Whether LoRA was used for fine-tuning
        device (str): Device to use, "cuda" or "cpu"
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create a simulated accelerator object
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device
    
    # Initialize agent
    print(f"Initializing OnlineReinforceAgent with base model: {base_model_name}")
    agent = OnlineReinforceAgent(
        device=device,
        accelerator=accelerator,
        policy_lm=base_model_name,
        use_lora=use_lora
    )
    
    # Load checkpoint
    print("Loading model weights from checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Adjust loading method based on checkpoint structure
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    agent.model.load_state_dict(state_dict)
    
    # Handle LoRA model
    if use_lora:
        try:
            from peft import PeftModel
            print("Merging LoRA weights into base model")
            merged_model = agent.model.merge_and_unload()
            base_model = merged_model
        except Exception as e:
            print(f"Failed to merge LoRA weights: {e}")
            print("Continuing with unmerged model")
            base_model = agent.model
    else:
        base_model = agent.model
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    print(f"Saving model to: {output_dir}")
    base_model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    
    print(f"Model successfully converted and saved to: {output_dir}")
    
    # Validate saved model
    print("Validating saved model...")
    try:
        loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
        loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print("Model validation successful! You can load the model with the following code:")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    except Exception as e:
        print(f"Model validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ARIA agent checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint", type=str, 
                        default="./trainer.pt", 
                        help="Path to the checkpoint file")
    parser.add_argument("--base_model", type=str, 
                        default="LlaMA-3-8b-Instruct", 
                        help="Base model name or path")
    parser.add_argument("--output_dir", type=str, 
                        default="./",
                        help="Directory to save the converted model")
    parser.add_argument("--use_lora", action="store_true", 
                        help="Whether LoRA was used for fine-tuning")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use, 'cuda' or 'cpu'")
    
    args = parser.parse_args()
    
    convert_checkpoint_to_hf_format(
        args.checkpoint,
        args.base_model,
        args.output_dir,
        args.use_lora,
        args.device
    )