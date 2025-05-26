#!/bin/bash

# Kill existing GPU processes
pid=$(ps -ef | grep 'gpu' | grep 'python' | grep -v grep | awk '{print $2}')
echo $pid
kill -9 $pid

# Complete parameter script for point-wise reward model training
export TORCH_DISTRIBUTED_BACKEND=gloo

# Set GPUs to use
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # Use first GPU, for multiple GPUs set to "0,1,2,3" etc.

# Set your own cache directories
export TRANSFORMERS_CACHE="/path/to/your/cache"
export HF_DATASETS_CACHE="/path/to/your/cache"

###################
# Basic Configuration Parameters
###################
# MODEL_NAME="/path/to/your/base/model"
MODEL_NAME="/path/to/your/trained/model"
MODEL_REVISION="main"                          # Added this missing parameter
DATASET_NAME="json"                            # Dataset name, use json for local json/jsonl files
DATASET_CONFIG=""                              # Dataset configuration name
# String needs to be wrapped in quotes
# DATA_FILES="{'train': '/path/to/your/train.jsonl', 'validation': '/path/to/your/validation.jsonl'}"
DATA_FILES="{'train': '/path/to/your/online_train.jsonl', 'validation': '/path/to/your/online_test.jsonl'}"
DATASET_TRAIN_SPLIT="train"                    # Training data split name
DATASET_TEST_SPLIT="validation"                # Test/validation data split name
# OUTPUT_DIR="/path/to/your/output"            # Output directory
OUTPUT_DIR="./models/single_q_rm_online_iter2_no_weight_ckp_iter1"

###################
# Training Parameters
###################
PER_DEVICE_TRAIN_BATCH_SIZE=4                # Training batch size per device
PER_DEVICE_EVAL_BATCH_SIZE=4                 # Evaluation batch size per device
GRADIENT_ACCUMULATION_STEPS=2                 # Gradient accumulation steps
LEARNING_RATE=2e-5                             # Learning rate
WEIGHT_DECAY=0.05                              # Weight decay
NUM_TRAIN_EPOCHS=5                             # Number of training epochs
MAX_STEPS=-1                                   # Maximum training steps (overrides epochs)
MAX_LENGTH=4096                               # Maximum sequence length
WARMUP_STEPS=0                                 # Warmup steps
WARMUP_RATIO=0.1                              # Warmup ratio
LR_SCHEDULER_TYPE="cosine"                     # Learning rate scheduler type
LOGGING_STEPS=10                               # Logging steps
SAVE_STEPS=200                                 # Model save steps
EVAL_STEPS=100                                 # Evaluation steps
SAVE_STRATEGY="steps"                          # Save strategy: "steps", "epoch" or "no"
EVAL_STRATEGY="steps"                          # Evaluation strategy: "steps", "epoch" or "no"
SAVE_TOTAL_LIMIT=3                             # Maximum number of checkpoints to save
SEED=42                                        # Random seed
EARLY_STOPPING_PATIENCE=5                      # Early stopping patience value
EARLY_STOPPING_THRESHOLD=0.0                   # Early stopping threshold
DO_EVAL="true"                                # Convert boolean to string
GREATER_IS_BETTER="true"                      # Convert boolean to string                      

###################
# Reward Model Specific Parameters
###################
IS_POINTWISE="true"                           # Convert boolean to string
INCLUDE_TOKENS_PER_SAMPLE="true"              # Convert boolean to string
INFERENCE_BATCH_SIZE=64                        # Inference batch size

###################
# Weighted Loss and Normalization Parameters
###################
USE_WEIGHTED_LOSS="true"                      # Whether to use weighted loss
MIDDLE_RANGE_WEIGHT=0.5                        # Middle range sample weight
EXTREME_RANGE_WEIGHT=2.0                       # Extreme range sample weight
# MIDDLE_RANGE_WEIGHT=1                        # Middle range sample weight
# EXTREME_RANGE_WEIGHT=1                      # Extreme range sample weight
MIDDLE_RANGE_THRESHOLD=0.05                    # Middle range threshold

###################
# PEFT/LoRA Parameters
###################
USE_PEFT="false"                              # Disable LoRA
LORA_R=16                                      # LoRA rank
LORA_ALPHA=32                                  # LoRA alpha parameter
LORA_DROPOUT=0.1                              # LoRA dropout ratio
LORA_TASK_TYPE="SEQ_CLS"                       # LoRA task type
LORA_BIAS="none"                               # LoRA bias type

###################
# Quantization Parameters
###################
LOAD_IN_4BIT="false"                          # Convert boolean to string
LOAD_IN_8BIT="false"                          # Convert boolean to string
BNB_4BIT_QUANT_TYPE="nf4"                      # 4bit quantization type: "nf4" or "fp4"
BNB_4BIT_COMPUTE_DTYPE="float16"               # Compute data type
USE_NESTED_QUANT="false"                      # Convert boolean to string
BNBONE_THRESHOLD=4.0                           # Quantization threshold
TORCH_DTYPE="float16"                          # Use float16 data type

###################
# Distributed Training Parameters
###################
FP16="false"
BF16="true"                      # Enable bf16, matching DeepSpeed configuration
TORCH_DTYPE="bfloat16"           # Match BF16
GRADIENT_CHECKPOINTING="true"    # Enable gradient checkpointing
#DDPYPE="NO_REDUCE"  # Already set, but may need other configurations
#DDP_FIND_UNUSED_PARAMETERS="true"  # New parameter

###################
# Logging and Hugging Face Hub Parameters
###################
REPORT_TO="wandb"                              # Changed to "none" to avoid wandb issues
PUSH_TO_HUB="false"                          # Convert boolean to string
#HUB_MODEL_ID=""                                # Hub model ID
#HUB_STRATEGY="every_save"                      # Hub push strategy
#HUB_TOKEN=""                                   # Hub API token

###################
# Other Parameters
###################
TRUST_REMOTE_CODE="false"                    # Convert boolean to string
#DEEPSPEED=""                                   # DeepSpeed configuration file
#CACHE_DIR=""                                   # Cache directory
FULL_DETERMINISM="false"                     # Convert boolean to string
GROUP_BY_LENGTH="true"                       # Convert boolean to string
LENGTH_COLUMN_NAME="length"                    # Length column name
PAD_TO_MAX_LENGTH="false"                    # Convert boolean to string
DISABLE_TQDM="false"                         # Convert boolean to string
REMOVE_UNUSED_COLUMNS="false"                 # Convert boolean to string
NUM_WORKERS=8                                  # Number of dataloader workers
DATALOADER_PIN_MEMORY="true"                 # Convert boolean to string
OPTIM="adamw_torch"                            # Optimizer
DS_CONFIG_PATH="/path/to/your/ds_config.json"

deepspeed --num_gpus=8 \
    /path/to/your/run_pointwise_rm.py \
    --deepspeed ${DS_CONFIG_PATH} \
    --model_name_or_path "${MODEL_NAME}" \
    --model_revision "${MODEL_REVISION}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --data_files "${DATA_FILES}" \
    --dataset_train_split "${DATASET_TRAIN_SPLIT}" \
    --dataset_test_split "${DATASET_TEST_SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --max_length ${MAX_LENGTH} \
    --warmup_steps ${WARMUP_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "${SAVE_STRATEGY}" \
    --eval_strategy "${EVAL_STRATEGY}" \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --seed ${SEED} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --early_stopping_threshold ${EARLY_STOPPING_THRESHOLD} \
    --do_eval ${DO_EVAL} \
    --greater_is_better ${GREATER_IS_BETTER} \
    --is_pointwise ${IS_POINTWISE} \
    --include_tokens_per_sample ${INCLUDE_TOKENS_PER_SAMPLE} \
    --inference_batch_size ${INFERENCE_BATCH_SIZE} \
    --use_weighted_loss ${USE_WEIGHTED_LOSS} \
    --middle_range_weight ${MIDDLE_RANGE_WEIGHT} \
    --extreme_range_weight ${EXTREME_RANGE_WEIGHT} \
    --middle_range_threshold ${MIDDLE_RANGE_THRESHOLD} \
    --use_peft ${USE_PEFT} \
    --load_in_4bit ${LOAD_IN_4BIT} \
    --load_in_8bit ${LOAD_IN_8BIT} \
    --bnb_4bit_quant_type "${BNB_4BIT_QUANT_TYPE}" \
    --bnb_4bit_compute_dtype "${BNB_4BIT_COMPUTE_DTYPE}" \
    --use_nested_quant ${USE_NESTED_QUANT} \
    --bnb_4bit_use_double_quant ${USE_NESTED_QUANT} \
    --bnb_4bit_quant_storage "${BNB_4BIT_COMPUTE_DTYPE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --fp16 ${FP16} \
    --bf16 ${BF16} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --report_to "${REPORT_TO}" \
    --push_to_hub ${PUSH_TO_HUB} \
    --trust_remote_code ${TRUST_REMOTE_CODE} \
    --ddp_type "${DDPYPE}" \
    --full_determinism ${FULL_DETERMINISM} \
    --group_by_length ${GROUP_BY_LENGTH} \
    --length_column_name "${LENGTH_COLUMN_NAME}" \
    --pad_to_max_length ${PAD_TO_MAX_LENGTH} \
    --disable_tqdm ${DISABLE_TQDM} \
    --remove_unused_columns ${REMOVE_UNUSED_COLUMNS} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --dataloader_pin_memory ${DATALOADER_PIN_MEMORY} \
    --optim "${OPTIM}"
