#!/usr/bin/env python
# coding=utf-8
import ast
import json
import os
import sys
import warnings
from typing import Dict, Optional, Union
from dataclasses import dataclass, field

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

# Import custom components
from reward_config import RewardConfig
from pointwiserm_trainer import PointwiseRewardTrainer, prepare_pointwise_dataset

# Copy ScriptArguments definition from original script
@dataclass
class ScriptArguments:
    """
    Parameters for training pointwise reward model
    """
    # Dataset parameters
    dataset_name: str = field(
        default="json", 
        metadata={"help": "Dataset name, set to 'json' when using local files"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration name"}
    )
    dataset_train_split: str = field(
        default="train", 
        metadata={"help": "Training split name"}
    )
    dataset_test_split: Optional[str] = field(
        default="validation", 
        metadata={"help": "Validation split name"}
    )
    data_files: Optional[str] = field(
        default=None,
        metadata={"help": "Data file path, can be string or dictionary format string, e.g.: \"{'train': 'train.jsonl', 'validation': 'val.jsonl'}\""} 
    )
    
    # Wandb related parameters
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project name"}
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run name"}
    )
    wandb_watch: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb monitoring mode: 'gradients', 'all', 'false'"}
    )
    
    # Other parameters
    include_tokens_per_sample: bool = field(
        default=False,
        metadata={"help": "Whether to include token count per sample in dataset"}
    )
    inference_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for inference"}
    )
    
    # Add all unrecognized parameters
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Early stopping patience: stop training after this many evaluations without improvement"}
    )
    early_stopping_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Early stopping threshold: improvement must exceed this amount to be considered valid"}
    )
    lora_bias: Optional[str] = field(
        default="none",
        metadata={"help": "LoRA bias type: 'none', 'all', 'lora_only'"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute data type"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use nested quantization"}
    )
    bnb_4bit_use_double_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use double quantization"}
    )
    bnb_4bit_quant_storage: Optional[str] = field(
        default="float16",
        metadata={"help": "Quantization storage type"}
    )
    ddp_type: Optional[str] = field(
        default=None,
        metadata={"help": "DDP type"}
    )
    pad_to_max_length: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to pad to maximum length"}
    )


def check_score_normalization(dataset: Union[Dataset, DatasetDict], split: str, verbose: bool = True) -> None:
    """Check if scores are normalized and provide warning information"""
    if split not in dataset:
        return
    
    scores = dataset[split]["score"]
    min_score = min(scores)
    max_score = max(scores)
    
    if min_score < 0 or max_score > 10:
        warnings.warn(
            f"Score range [{min_score}, {max_score}] seems unusual. "
            "Consider normalizing scores to 0-1 range for better training stability."
        )
    elif verbose:
        print(f"Score range: [{min_score}, {max_score}]")
        
    # If all scores are the same, the model won't be able to learn meaningful patterns
    if min_score == max_score:
        warnings.warn(
            f"All scores are the same ({min_score}). The model won't be able to learn meaningful patterns!"
        )


def analyze_score_distribution(dataset: Union[Dataset, DatasetDict], split: str) -> None:
    """Analyze score distribution and provide weighted loss suggestions"""
    if split not in dataset:
        return
    
    scores = dataset[split]["score"]
    total = len(scores)
    
    # Calculate sample proportion in each range
    middle_count = sum(1 for s in scores if 0.45 < s < 0.55)
    extreme_count = sum(1 for s in scores if s < 0.01 or s > 0.8)
    other_count = total - middle_count - extreme_count
    
    middle_ratio = middle_count / total
    extreme_ratio = extreme_count / total
    other_ratio = other_count / total
    
    print(f"\nScore distribution analysis ({split} split):")
    print(f"  Middle range (0.45-0.55): {middle_count} samples ({middle_ratio:.1%})")
    print(f"  Extreme range (<0.2 or >0.8): {extreme_count} samples ({extreme_ratio:.1%})")
    print(f"  Other range: {other_count} samples ({other_ratio:.1%})")
    
    # Provide suggestions
    if middle_ratio > 0.5:
        print("\nSuggestion: High proportion of middle range samples, consider enabling weighted loss:")
        print("  --use_weighted_loss=True --middle_range_weight=0.3 --extreme_range_weight=3.0")
    elif extreme_ratio < 0.1:
        print("\nSuggestion: Low proportion of extreme range samples, consider enabling weighted loss:")
        print("  --use_weighted_loss=True --middle_range_weight=1.0 --extreme_range_weight=5.0")
    else:
        print("\nScore distribution is relatively balanced, weighted loss may not be necessary, or adjust weights as needed")


def validate_dataset_structure(dataset: Union[Dataset, DatasetDict], splits: list) -> None:
    """Validate whether dataset structure meets requirements"""
    for split in splits:
        if split not in dataset:
            # If the specified split doesn't exist, warn but don't error
            warnings.warn(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
            continue
            
        # Check necessary columns
        columns = dataset[split].column_names
        if "text" not in columns or "score" not in columns:
            raise ValueError(
                f"Point-wise reward modeling requires a dataset with 'text' and 'score' columns in the '{split}' split. "
                f"Available columns: {columns}"
            )
        
        # Check text column type
        text_type = type(dataset[split]["text"][0])
        if text_type is not str:
            warnings.warn(
                f"'text' column in '{split}' split contains {text_type} instead of strings. "
                "This might cause issues during tokenization."
            )
        
        # Check score column type
        score_type = type(dataset[split]["score"][0])
        if score_type not in (int, float):
            warnings.warn(
                f"'score' column in '{split}' split contains {score_type} instead of numbers. "
                "This might cause issues during training."
            )


if __name__ == "__main__":
    # Add parameter parsing type hints
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    
    # Parse command line arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If JSON configuration file is provided
        script_args, training_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Ensure pointwise reward model is enabled
    if not training_args.is_pointwise:
        warnings.warn(
            "Setting --is_pointwise=True as it's required for point-wise reward modeling.",
            UserWarning,
        )
        training_args.is_pointwise = True
    
    # Set gradient checkpointing
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model and tokenizer
    ################
    print("Loading model and tokenizer...")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code, 
        use_fast=True
    )
    
    # Ensure tokenizer has pad_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set pad_token_id = eos_token_id: {tokenizer.pad_token_id}")
        else:
            raise ValueError("Tokenizer has no pad_token or eos_token, please set manually")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, 
        num_labels=1, 
        trust_remote_code=model_args.trust_remote_code, 
        **model_kwargs
    )
    
    # Align tokenizer and model pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # If it's a base model, use ChatML as default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
        print("Applied ChatML template to model")

    # PEFT warning
    if model_args.use_peft:
        if model_args.lora_task_type != "SEQ_CLS":
            warnings.warn(
                "You are using a task_type different from SEQ_CLS. For reward models, please use --lora_task_type SEQ_CLS to avoid potential issues.",
                UserWarning,
            )
        print(f"Will use LoRA for training, parameters: r={model_args.lora_r}, alpha={model_args.lora_alpha}")

    ##############
    # Load dataset
    ##############
    print(f"Loading dataset: {script_args.dataset_name}")
    
    # Process data_files parameter, support string dictionary format
    data_files = script_args.data_files
    if isinstance(data_files, str) and data_files.startswith("{") and data_files.endswith("}"):
        try:
            # Try to parse dictionary string, e.g.: "{'train': 'train.jsonl', 'validation': 'val.jsonl'}"
            data_files = ast.literal_eval(data_files)
        except (SyntaxError, ValueError) as e:
            warnings.warn(f"Unable to parse data_files dictionary: {e}, will treat as single file")
    
    # Load dataset
    try:
        dataset = load_dataset(
            script_args.dataset_name, 
            name=script_args.dataset_config,
            data_files=data_files,
            trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

    # Set default split names
    default_train_split = "train"
    default_test_split = "validation" if "validation" in dataset else "test" if "test" in dataset else None
    
    train_split = getattr(script_args, "dataset_train_split", default_train_split) or default_train_split
    test_split = getattr(script_args, "dataset_test_split", default_test_split) or default_test_split
    
    print(f"Using training split: {train_split}")
    if test_split:
        print(f"Using validation split: {test_split}")
    else:
        print("No validation split specified")
    
    # Validate dataset structure
    validate_dataset_structure(dataset, [train_split, test_split] if test_split else [train_split])
    
    # Check score normalization
    check_score_normalization(dataset, train_split)
    if test_split:
        check_score_normalization(dataset, test_split, verbose=False)
    
    # Analyze score distribution
    print("\nAnalyzing training set score distribution...")
    analyze_score_distribution(dataset, train_split)

    # Print dataset statistics
    print(f"Training set size: {len(dataset[train_split])}")
    if test_split and test_split in dataset:
        print(f"Validation set size: {len(dataset[test_split])}")

    ##############
    # Prepare dataset
    ##############
    print("Preparing dataset...")
    train_dataset = prepare_pointwise_dataset(
        dataset[train_split], 
        tokenizer, 
        max_length=training_args.max_length
    )

    if test_split and test_split in dataset and training_args.eval_strategy != "no":
        eval_dataset = prepare_pointwise_dataset(
            dataset[test_split], 
            tokenizer, 
            max_length=training_args.max_length
        )
    else:
        eval_dataset = None

    ##########
    # Training
    ##########
    print("Initializing trainer...")
    trainer = PointwiseRewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )

    # Display weighted loss configuration
    if training_args.use_weighted_loss:
        print(f"\nWeighted loss enabled - Configuration:")
        print(f"  Middle range weight (middle_range_weight): {training_args.middle_range_weight}")
        print(f"  Extreme range weight (extreme_range_weight): {training_args.extreme_range_weight}")
        print(f"  Middle range threshold (middle_range_threshold): {training_args.middle_range_threshold}")
        
        # Call visualization
        trainer.visualize_weights()
    
    if training_args.use_weighted_loss:
        print("\nUsing weighted loss training, please monitor the following metrics during training:")
        print("  - Middle range MSE (metrics/middle_range_mse): Mean squared error for samples near 0.5")
        print("  - Extreme range MSE (metrics/extreme_range_mse): Mean squared error for samples near 0 or 1")
        print("  - Correlation coefficient (metrics/pearson_correlation): Correlation between predicted and true scores")

        print("Starting training...")
        trainer.train()

        ############################
        # Save model and push to Hub
        ############################
        print(f"Training completed, saving model to {training_args.output_dir}...")
        trainer.save_model(training_args.output_dir)
        
        # Evaluate model
        if training_args.eval_strategy != "no" and test_split in dataset:
            print("Evaluating model...")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            print(f"Evaluation results: {metrics}")

        # Push to Hub
        if training_args.push_to_hub:
            print("Pushing model to Hugging Face Hub...")
            model_card_kwargs = {
                "task_type": "reward-modeling-pointwise",
                "model-index": [
                    {
                        "name": training_args.hub_model_id if training_args.hub_model_id else training_args.output_dir,
                        "results": [],
                    }
                ],
                "tags": ["reward-model", "point-wise", "trl"],
            }
            trainer.push_to_hub(dataset_name=script_args.dataset_name, **model_card_kwargs)
            print("Model successfully pushed to Hub")
        
        print("🎉 Training completed!")