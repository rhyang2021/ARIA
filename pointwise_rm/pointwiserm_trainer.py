# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field, FrozenInstanceError, replace
from typing import Optional, Any, Callable, Dict, List, Union
from collections import defaultdict
import pandas as pd

from transformers import TrainingArguments, PreTrainedTokenizerBase, PreTrainedModel
from transformers import Trainer
import torch
import torch.nn as nn
from datasets import Dataset
import logging
from accelerate import PartialState
from accelerate.utils import gather_object

# Import RewardTrainer and RewardConfig
from trl import RewardTrainer
from reward_config import RewardConfig

logger = logging.getLogger(__name__)

@dataclass
class PointwiseRewardDataCollatorWithPadding:
    """
    Point-wise reward model data collator for processing data containing single text and rating.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_features = []
        scores = []
        
        # Check input data format
        for feature in features:
            if "input_ids" not in feature or "attention_mask" not in feature:
                raise ValueError(
                    "Features should contain `input_ids` and `attention_mask`"
                )
            
            text_features.append({
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            })
            
            # Collect score labels
            if "labels" in feature:
                scores.append(feature["labels"])
        
        # Pad text data
        batch_inputs = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Build final batch
        batch = {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "return_loss": True,
        }
        
        # Add score data (if available)
        if scores:
            batch["labels"] = torch.tensor(scores, dtype=torch.float)
        
        return batch


def _tokenize_pointwise(batch: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizerBase") -> Dict[str, List[Any]]:
    """Tokenize a batch from a point-wise reward dataset."""
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],  # For storing scores/ratings
    }
    
    for text, score in zip(batch["text"], batch["score"]):
        tokenized = tokenizer(text)
        new_examples["input_ids"].append(tokenized["input_ids"])
        new_examples["attention_mask"].append(tokenized["attention_mask"])
        new_examples["labels"].append(score)

    return new_examples


def decode_and_strip_padding(input_ids, tokenizer):
    """Decode input IDs and remove padding tokens"""
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    
    texts = []
    for ids in input_ids:
        if tokenizer.pad_token_id in ids:
            ids = ids[:list(ids).index(tokenizer.pad_token_id)]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    
    return texts


# Create a simple dataset class
class SimpleDataset:
    """A simple dataset class to bypass RewardTrainer's data processing"""
    
    def __init__(self):
        self.column_names = ["input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"]
        self.features = []
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return {}
    
    def map(self, *args, **kwargs):
        # This method will be called by RewardTrainer, but we do nothing
        return self
    
    def filter(self, *args, **kwargs):
        # This method will be called by RewardTrainer, but we do nothing
        return self


class PointwiseRewardTrainer(Trainer):
    """
    Point-wise reward model trainer designed for single text rating tasks.
    Supports weighted loss to emphasize extreme scores and reduce weight for middle-range scores.
    
    Note: This class inherits directly from Trainer instead of RewardTrainer to avoid data processing conflicts.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        peft_config = None,
    ):
        """Initialize PointwiseRewardTrainer"""
        # Ensure args is not None
        if args is None:
            args = RewardConfig()
        
        # Get weighted loss parameters from configuration
        self.use_weighted_loss = getattr(args, "use_weighted_loss", False)
        self.middle_range_weight = getattr(args, "middle_range_weight", 0.5)
        self.extreme_range_weight = getattr(args, "extreme_range_weight", 2.0)
        self.middle_range_threshold = getattr(args, "middle_range_threshold", 0.05)
        
        # If no data collator is provided, create point-wise data collator
        if data_collator is None and processing_class is not None:
            data_collator = PointwiseRewardDataCollatorWithPadding(processing_class)
            
            # Ensure args.remove_unused_columns is set correctly
            if getattr(args, "remove_unused_columns", True):
                import warnings
                
                try:
                    args = replace(args, remove_unused_columns=False)
                except FrozenInstanceError:
                    warnings.warn(
                        "When using PointwiseRewardDataCollator, you should set `remove_unused_columns=False`.",
                        UserWarning,
                    )
    
        # Initialize attributes for logging
        self.current_batch_labels = None
        self.current_batch_predictions = None
        
        # If peft_config is provided, apply PEFT
        if peft_config is not None:
            try:
                from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
                
                if not isinstance(model, PeftModel):
                    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                        model = prepare_model_for_kbit_training(
                            model, 
                            use_gradient_checkpointing=getattr(args, "gradient_checkpointing", False)
                        )
                    
                    model = get_peft_model(model, peft_config)
            except ImportError:
                warnings.warn(
                    "PEFT is not installed but you passed a `peft_config`. "
                    "You should install it with `pip install peft`.",
                    UserWarning,
                )
        
        # Disable dropout in the model
        if getattr(args, "disable_dropout", True):
            self._disable_dropout(model)
            
        # Call Trainer's initialization directly
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,  # Note: Trainer uses tokenizer instead of processing_class
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Add tags
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(["trl", "reward-trainer"])

    def _disable_dropout(self, model):
        """Disable dropout in the model"""
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute loss for point-wise reward model.
        Supports weighted MSE loss to emphasize specific score ranges.
        """
        # Get model predictions
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        predicted_scores = outputs["logits"]
        
        # Get true scores
        if "labels" in inputs:
            true_scores = inputs["labels"].view_as(predicted_scores)
            
            # Store batch data for logging
            self.current_batch_labels = true_scores.detach()
            self.current_batch_predictions = predicted_scores.detach()
            
            # Calculate loss based on configuration
            if self.use_weighted_loss:
                # Create weight tensor
                weights = torch.ones_like(true_scores)
                
                # Apply lower weight for middle-range samples
                middle_threshold = self.middle_range_threshold
                mask_middle = (true_scores > (0.5 - middle_threshold)) & (true_scores < (0.5 + middle_threshold))
                
                # Apply higher weight for extreme samples
                # mask_extreme = (true_scores < 0.2) | (true_scores > 0.8)
                mask_extreme = (true_scores < 0.01) | (true_scores > 0.8)
                
                # Set weights
                weights[mask_middle] = self.middle_range_weight
                weights[mask_extreme] = self.extreme_range_weight
                
                # Calculate weighted MSE loss
                loss = torch.mean(weights * (predicted_scores - true_scores) ** 2)
            else:
                # Use standard MSE loss
                loss = torch.nn.functional.mse_loss(predicted_scores, true_scores)
            
            if return_outputs:
                return loss, {"predicted_scores": predicted_scores}
            return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prediction step for evaluation and inference"""
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        # Prediction logic for point-wise data
        with torch.no_grad():
            loss, outputs_dict = self.compute_loss(model, inputs, return_outputs=True)
            predicted_scores = outputs_dict["predicted_scores"]
            
        if prediction_loss_only:
            return (loss, None, None)
            
        return loss, predicted_scores, inputs.get("labels", None)
    
    def evaluate(self, *args, **kwargs):
        num_print_samples = kwargs.pop("num_print_samples", 4)
        self.visualize_samples(num_print_samples)
        return super().evaluate(*args, **kwargs)
    
    def visualize_samples(self, num_print_samples: int):
        """Visualize sample predictions"""
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        
        for _, inputs in enumerate(eval_dataloader):
            _, predicted_scores, true_scores = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            
            text = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)  # Note: using self.tokenizer here
            
            table["text"].extend(gather_object(text))
            table["predicted_score"].extend(gather_object([round(score.item(), 4) for score in predicted_scores]))
            if true_scores is not None:
                table["true_score"].extend(gather_object([round(score.item(), 4) for score in true_scores]))
            
            if num_print_samples >= 0 and len(table["text"]) >= num_print_samples:
                break
                
        df = pd.DataFrame(table)
        
        # Check if in distributed environment
        if hasattr(self, "accelerator") and self.accelerator.process_index == 0:
            # Use simple printing
            print(df[:num_print_samples])
                
            # Log to wandb
            if hasattr(self.args, "report_to") and "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"completions": wandb.Table(dataframe=df)})
                except ImportError:
                    pass

    def visualize_weights(self, dataset=None):
        """Visualize sample weight distribution"""
        if not self.use_weighted_loss:
            logger.info("Weighted loss not enabled, no need to visualize weights")
            return
            
        if dataset is None:
            dataset = self.train_dataset
            
        # Collect sample scores and weights
        scores = []
        weights = []
        
        for i in range(min(1000, len(dataset))):  # Limit sample size
            sample = dataset[i]
            score = sample["labels"].item() if isinstance(sample["labels"], torch.Tensor) else sample["labels"]
            scores.append(score)
            
            # Calculate weight
            weight = 1.0
            if 0.5 - self.middle_range_threshold < score < 0.5 + self.middle_range_threshold:
                weight = self.middle_range_weight
            # elif score < 0.2 or score > 0.8:
            elif score < 0.01 or score > 0.8:
                weight = self.extreme_range_weight
            weights.append(weight)
        
        # Use matplotlib to plot weight distribution
        try:
            import wandb
            if wandb.run is not None:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(scores, weights, alpha=0.5)
                ax.set_xlabel('Sample Score')
                ax.set_ylabel('Sample Weight')
                ax.set_title('Sample Score and Weight Distribution')
                ax.grid(True)
                
                # Add weight parameter information
                textstr = f"Middle range weight: {self.middle_range_weight}\n" \
                        f"Extreme range weight: {self.extreme_range_weight}\n" \
                        f"Middle range threshold: {self.middle_range_threshold}"
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                
                wandb.log({"weight_distribution": wandb.Image(fig)})
                plt.close()
        except ImportError:
            logger.warning("Matplotlib or wandb not available, skipping weight visualization")


# Helper function: prepare dataset for point-wise reward model
def prepare_pointwise_dataset(dataset, tokenizer, max_length):
    """Prepare dataset for point-wise reward model"""
    from accelerate import PartialState
    
    with PartialState().main_process_first():
        # Apply tokenization to dataset
        dataset = dataset.map(
            _tokenize_pointwise,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=4,  # Adjust as needed
        )
        
        # Filter samples that are too long
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) <= max_length,
            num_proc=4,  # Adjust as needed
        )
    
    return dataset