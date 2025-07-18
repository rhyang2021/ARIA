#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY="Your_API_key"
ENV_NAMES=("twenty_questions" "guess_my_city")
MODEL_NAME="llama3-8B"
MODEL_PORT=8035
for env_name in "${ENV_NAMES[@]}"; do
    python main.py \
        --env_name "$env_name" \
        --model_id "$MODEL_NAME" \
        --model_port "$MODEL_PORT" \
        --repeat 200 \
        --output_dir "../results/single_agent/$MODEL_NAME" 
done
