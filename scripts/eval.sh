#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <task1> <task2> ..."
    echo "Example: $0 arc_challenge winogrande"
    exit 1
fi

cd "$(dirname "$0")/.."

TASKS=$(IFS=,; echo "$*")
NUM_FEWSHOT=0
BATCH_SIZE=2
TEMPERATURE=0
CHECKPOINT=195

echo "Running evaluations on tasks: $TASKS"
echo "================================================"

echo "Evaluating base model: google/gemma-3-1b-pt"
lm_eval --model hf \
  --model_args pretrained=google/gemma-3-1b-pt,trust_remote_code=True,parallelize=True,device_map=auto \
  --tasks "$TASKS" \
  --num_fewshot $NUM_FEWSHOT \
  --batch_size $BATCH_SIZE \
  --gen_kwargs "temperature=$TEMPERATURE" \
  --output_path reports/gemma-3-1b-pt.json

echo "================================================"
echo "Evaluating: base_token_gating_gemma3_1b_pt"
lm_eval --model hf \
  --model_args pretrained=./outputs/base_token_gating_gemma3_1b_pt/checkpoint-$CHECKPOINT,trust_remote_code=True,parallelize=True,device_map=auto \
  --tasks "$TASKS" \
  --num_fewshot $NUM_FEWSHOT \
  --batch_size $BATCH_SIZE \
  --apply_chat_template \
  --gen_kwargs "temperature=$TEMPERATURE" \
  --output_path reports/base_token_gating_gemma3_1b_pt.json

echo "================================================"
echo "Evaluating: random_gemma3_1b_pt"
lm_eval --model hf \
  --model_args pretrained=./outputs/random_gemma3_1b_pt/checkpoint-$CHECKPOINT,trust_remote_code=True,parallelize=True,device_map=auto \
  --tasks "$TASKS" \
  --num_fewshot $NUM_FEWSHOT \
  --batch_size $BATCH_SIZE \
  --apply_chat_template \
  --gen_kwargs "temperature=$TEMPERATURE" \
  --output_path reports/random_gemma3_1b_pt.json

echo "================================================"
echo "Evaluating: loss_gating_gemma3_1b_pt"
lm_eval --model hf \
  --model_args pretrained=./outputs/loss_gating_gemma3_1b_pt/checkpoint-$CHECKPOINT,trust_remote_code=True,parallelize=True,device_map=auto \
  --tasks "$TASKS" \
  --num_fewshot $NUM_FEWSHOT \
  --batch_size $BATCH_SIZE \
  --apply_chat_template \
  --gen_kwargs "temperature=$TEMPERATURE" \
  --output_path reports/loss_gating_gemma3_1b_pt.json

echo "================================================"
echo "All evaluations complete. Results in reports/"
