# RewardBench Survey Project

This repository contains code, notebooks, results, and figures for a survey project on reward model evaluation using RewardBench.

## Project Goal

The goal of this project is to compare representative classifier-based reward models and DPO-style implicit reward models across the main RewardBench evaluation sections:

- Chat
- Chat Hard
- Safety
- Reasoning

The project focuses on section-level performance trends, model-family differences, and practical tradeoffs in reward model evaluation.

## Models Evaluated

### Classifier Reward Models

- OpenAssistant/reward-model-deberta-v3-large-v2
- RLHFlow/ArmoRM-Llama3-8B-v0.1

### DPO-style / Implicit Reward Models

- Qwen/Qwen1.5-0.5B-Chat with Qwen/Qwen1.5-0.5B as reference
- HuggingFaceH4/zephyr-7b-beta with mistralai/Mistral-7B-v0.1 as reference

## Repository Structure

```text
notebooks/
  rewardbench_survey_colab.ipynb

scripts/
  run_core_models.sh
  make_figures.py

results/
  model_comparison.csv
  section_scores.csv
  raw_outputs/

figures/
  section_scores.png
  model_family_average.png

report/
  main.tex
  references.bib


Setup
The experiments use the official RewardBench codebase.
git clone https://github.com/allenai/reward-bench.git
cd reward-bench
pip install -e .

Running Experiments
Reward model example:
python scripts/run_rm.py \
  --model=OpenAssistant/reward-model-deberta-v3-large-v2 \
  --chat_template=raw \
  --batch_size=8
DPO-style model example:
python scripts/run_dpo.py \
  --model=Qwen/Qwen1.5-0.5B-Chat \
  --ref_model=Qwen/Qwen1.5-0.5B \
  --batch_size=8

Results
Processed results are stored in:
results/model_comparison.csv
results/section_scores.csv

Figures are stored in:
figures/
Notes
This repository does not replace the original RewardBench repository. Instead, it provides standalone scripts, notebooks, results, and analysis files for this course survey project.

Commit it with:

```text
Update README with project overview
