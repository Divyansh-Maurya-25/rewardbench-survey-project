#!/bin/bash

# RewardBench Survey Project
# Runs the four core pretrained checkpoints used in the project.
# Assumes this script is run from inside the official reward-bench repository.

mkdir -p ../raw_outputs

echo "Running DeBERTa RM..."
python scripts/run_rm.py \
  --model=OpenAssistant/reward-model-deberta-v3-large-v2 \
  --chat_template=raw \
  --batch_size=16 \
  2>&1 | tee ../raw_outputs/deberta_rm.log

echo "Running Qwen 0.5B DPO..."
python scripts/run_dpo.py \
  --model=Qwen/Qwen1.5-0.5B-Chat \
  --ref_model=Qwen/Qwen1.5-0.5B \
  --batch_size=1 \
  2>&1 | tee ../raw_outputs/qwen_05b_dpo_bs1.log

echo "Running ArmoRM 8B..."
python scripts/run_rm.py \
  --model=RLHFlow/ArmoRM-Llama3-8B-v0.1 \
  --batch_size=1 \
  --trust_remote_code \
  --do_not_save \
  2>&1 | tee ../raw_outputs/armorm_patched_bs1.log

echo "Running Zephyr 7B DPO..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dpo.py \
  --model=HuggingFaceH4/zephyr-7b-beta \
  --ref_model=HuggingFaceH4/mistral-7b-sft-beta \
  --batch_size=1 \
  2>&1 | tee ../raw_outputs/zephyr_7b_beta_dpo_bs1.log
