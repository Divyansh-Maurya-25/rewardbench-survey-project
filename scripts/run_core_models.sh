#!/bin/bash

# RewardBench Survey Project
# Core commands used for the final experiments.
# These commands assume RewardBench is installed and the official
# RewardBench repository has been cloned.

# 1. DeBERTa classifier reward model
python scripts/run_rm.py \
  --model=OpenAssistant/reward-model-deberta-v3-large-v2 \
  --chat_template=raw \
  --batch_size=16

# 2. Qwen small DPO-style model
python scripts/run_dpo.py \
  --model=Qwen/Qwen1.5-0.5B-Chat \
  --ref_model=Qwen/Qwen1.5-0.5B \
  --batch_size=1

# 3. ArmoRM strong reward model
# Note: In Colab, ArmoRM required the local ArmoRM pipeline patch
# included in the notebook.
python scripts/run_rm.py \
  --model=RLHFlow/ArmoRM-Llama3-8B-v0.1 \
  --batch_size=1 \
  --trust_remote_code \
  --do_not_save

# 4. Zephyr DPO-style model
python scripts/run_dpo.py \
  --model=HuggingFaceH4/zephyr-7b-beta \
  --ref_model=mistralai/Mistral-7B-v0.1 \
  --batch_size=1
