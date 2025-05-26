#!/usr/bin/env bash
echo '——————————kill gpu process——————————'
pid=$(ps -ef | grep 'gpu' | grep 'python' | grep -v grep | awk '{print $2}')
echo $pid
kill -9 $pid

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENAI_API_KEY='api_key'

# Set configuration file path
DS_CONFIG_PATH="config/ds_config.json"

# Launch directly with DeepSpeed
deepspeed --num_gpus=8 \
  run_online.py \
  --config-name onlinereinforce_llm \
  --deepspeed_config=${DS_CONFIG_PATH}

# For accelerate
# accelerate launch --config_file config/accelerate_config/default_config.yaml \
                # run_offline.py \
                # --config-name reinforce_llm
