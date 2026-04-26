#!/bin/bash

# RewardBench Survey Project
# Core experiment commands for classifier reward models and DPO-style models.

# 1. Lightweight classifier reward model
python scripts/run_rm.py \
  --model=OpenAssistant/reward-model-deberta-v3-large-v2 \
  --chat_template=raw \
  --batch_size=8

# 2. Strong classifier/custom reward model
python scripts/run_rm.py \
  --model=RLHFlow/ArmoRM-Llama3-8B-v0.1 \
  --batch_size=1 \
  --trust_remote_code

# 3. Small DPO-style model
python scripts/run_dpo.py \
  --model=Qwen/Qwen1.5-0.5B-Chat \
  --ref_model=Qwen/Qwen1.5-0.5B \
  --batch_size=8

# 4. Stronger DPO-style model
python scripts/run_dpo.py \
  --model=HuggingFaceH4/zephyr-7b-beta \
  --ref_model=mistralai/Mistral-7B-v0.1 \
  --batch_size=1
