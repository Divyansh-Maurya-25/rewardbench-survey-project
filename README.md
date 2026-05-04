# RewardBench Survey Project

**Survey: Benchmarking Reward Models for LLM Alignment with RewardBench**

**Authors:** Rodela Ghosh · Divyansh Maurya

This project uses the official [RewardBench](https://github.com/allenai/reward-bench) framework to evaluate representative reward-model approaches across the four main RewardBench sections: **Chat**, **Chat Hard**, **Safety**, and **Reasoning**.

This is a **survey-style evaluation**, not a full reproduction of the complete RewardBench leaderboard. We evaluate a small representative set of pretrained checkpoints to study model behavior, section-level trends, safety/refusal calibration, and fine-grained subset-level weaknesses.

---

## Table of Contents

- [Project Summary](#project-summary)
- [Repository Structure](#repository-structure)
- [Models Evaluated](#models-evaluated)
- [Method](#method)
- [Experimental Setup](#experimental-setup)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Key Findings](#key-findings)
- [Implementation Notes](#implementation-notes)
- [Acknowledgements](#acknowledgements)

---

## Project Summary

Reward models are important in LLM alignment because they assign preference scores to candidate responses. RewardBench evaluates reward models using prompt/chosen/rejected triples. A model is counted as correct when it assigns a higher score to the chosen response than to the rejected response.

This repository compares:
- Explicit classifier-based reward models
- DPO-style implicit reward models

The goal is to understand where different reward-model approaches succeed or fail, especially on harder preference distinctions and safety-related refusal behavior.

---

## Repository Structure

```
rewardbench-survey-project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── rewardbench_survey_colab.ipynb
├── scripts/
│   ├── run_core_models.sh
│   └── make_figures.py
├── results/
│   ├── section_scores.csv
│   ├── subset_scores.csv
│   ├── fine_grained_subset_scores.csv
│   ├── model_comparison.csv
│   ├── ours_vs_rewardbench_paper.csv
│   ├── ranking_trend_comparison.csv
│   ├── weakest_subsets_by_model.csv
│   ├── deberta_model_summary.csv
│   ├── deberta_section_scores.csv
│   ├── deberta_subset_scores.csv
│   └── report_result_notes.txt
├── figures/
│   ├── section_scores_by_model.pdf
│   ├── ours_vs_rewardbench_paper_avg.pdf
│   ├── fine_grained_subset_heatmap.pdf
│   ├── safety_breakdown.pdf
│   └── hard_reasoning_breakdown.pdf
└── raw_outputs/
    ├── armorm_patched_bs1.log
    ├── deberta_rm.log
    ├── qwen_05b_dpo_bs1.log
    └── zephyr_7b_beta_dpo_bs1.log
```

---

## Models Evaluated

| Model | Family | Role |
|---|---|---|
| `RLHFlow/ArmoRM-Llama3-8B-v0.1` | Custom Classifier RM | Strong classifier reward model |
| `HuggingFaceH4/zephyr-7b-beta` | DPO-style implicit RM | Stronger DPO-style aligned model |
| `OpenAssistant/reward-model-deberta-v3-large-v2` | Sequence Classifier RM | Lightweight classifier baseline |
| `Qwen/Qwen1.5-0.5B-Chat` | DPO-style implicit RM | Small DPO-style baseline |

**Reference models used for DPO-style evaluation:**

| Policy Model | Reference Model |
|---|---|
| `Qwen/Qwen1.5-0.5B-Chat` | `Qwen/Qwen1.5-0.5B` |
| `HuggingFaceH4/zephyr-7b-beta` | `HuggingFaceH4/mistral-7b-sft-beta` |

---

## Method

RewardBench evaluates reward models on prompt/chosen/rejected triples. For each example, the model is correct if it assigns a higher score to the chosen response than the rejected response.

### Classifier Reward Models

Classifier reward models explicitly output scalar reward scores. The chosen response is preferred if:

$$r(x, y_{\text{chosen}}) > r(x, y_{\text{rejected}})$$

A standard Bradley-Terry-style preference probability:

$$P(y_{\text{chosen}} > y_{\text{rejected}} \mid x) = \frac{\exp(r(x, y_{\text{chosen}}))}{\exp(r(x, y_{\text{chosen}})) + \exp(r(x, y_{\text{rejected}}))}$$

We evaluate two classifier reward models:
- **DeBERTa RM** — a standard sequence classifier reward model using Hugging Face's sequence-classification interface.
- **ArmoRM 8B** — a custom classifier reward model using a custom scoring pipeline that scores chosen and rejected responses separately.

### DPO-Style Implicit Reward Models

DPO-style models derive preference scores from the log-probability difference between a policy model and a reference model:

$$r_{\text{DPO}}(x, y) = \beta \cdot \log\frac{\pi_{\text{policy}}(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

The chosen response is preferred if its policy/reference log-ratio is higher than the rejected response's. This evaluation requires loading both the policy model and the reference model simultaneously.

---

## Experimental Setup

- Experiments were run in Google Colab on an **NVIDIA A100 GPU**.
- This project is **inference-only** — no reward models were trained or fine-tuned, and the RewardBench dataset was not modified.
- The official RewardBench scripts (`scripts/run_rm.py` and `scripts/run_dpo.py`) were used directly.
- Processed CSVs and figures were generated from RewardBench evaluation outputs.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/Divyansh-Maurya-25/rewardbench-survey-project.git
cd rewardbench-survey-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main experiments also require the official RewardBench repository:

```bash
git clone https://github.com/allenai/reward-bench.git
```

---

## Running Experiments

The main reproducibility artifact is:

```
notebooks/rewardbench_survey_colab.ipynb
```

The notebook covers the full workflow: RewardBench setup, GPU check, model evaluation, log saving, CSV generation, and figure generation. A command summary is also provided in `scripts/run_core_models.sh`.

### Core Evaluation Commands

**DeBERTa RM**
```bash
python scripts/run_rm.py \
  --model=OpenAssistant/reward-model-deberta-v3-large-v2 \
  --chat_template=raw \
  --batch_size=16 \
  2>&1 | tee ../raw_outputs/deberta_rm.log
```

**Qwen 0.5B DPO**
```bash
python scripts/run_dpo.py \
  --model=Qwen/Qwen1.5-0.5B-Chat \
  --ref_model=Qwen/Qwen1.5-0.5B \
  --batch_size=1 \
  2>&1 | tee ../raw_outputs/qwen_05b_dpo_bs1.log
```

**ArmoRM 8B**
```bash
python scripts/run_rm.py \
  --model=RLHFlow/ArmoRM-Llama3-8B-v0.1 \
  --batch_size=1 \
  --trust_remote_code \
  --do_not_save \
  2>&1 | tee ../raw_outputs/armorm_patched_bs1.log
```

**Zephyr 7B DPO**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_dpo.py \
  --model=HuggingFaceH4/zephyr-7b-beta \
  --ref_model=HuggingFaceH4/mistral-7b-sft-beta \
  --batch_size=1 \
  2>&1 | tee ../raw_outputs/zephyr_7b_beta_dpo_bs1.log
```

### Regenerating Figures

After placing processed CSVs in `results/`, regenerate all PDF figures with:

```bash
python scripts/make_figures.py
```

---

## Results

### Section-Level Accuracy

Models ordered by four-section average.

| Model | Chat | Chat Hard | Safety | Reasoning | **Avg** |
|---|---|---|---|---|---|
| ArmoRM 8B | 0.9721 | 0.7632 | 0.9041 | 0.9738 | **0.9033** |
| Zephyr 7B DPO | 0.9553 | 0.5899 | 0.6689 | 0.7879 | **0.7505** |
| DeBERTa RM | 0.8045 | 0.4408 | 0.7662 | 0.3776 | **0.5973** |
| Qwen 0.5B DPO | 0.3659 | 0.6316 | 0.5662 | 0.6069 | **0.5427** |

Results file: `results/section_scores.csv` · Figure: `figures/section_scores_by_model.pdf`

### Comparison with RewardBench Paper

Comparison uses a four-section average (Chat, Chat Hard, Safety, Reasoning) to match the scope of this project.

| Model | This Project | RewardBench Paper | Difference |
|---|---|---|---|
| ArmoRM 8B | 0.903 | 0.908 | −0.005 |
| Zephyr 7B DPO | 0.751 | 0.742 | +0.008 |
| DeBERTa RM | 0.597 | 0.538 | +0.060 |
| Qwen 0.5B DPO | 0.543 | 0.561 | −0.018 |

Local results broadly preserve the ranking trends from the RewardBench paper.

Results file: `results/ours_vs_rewardbench_paper.csv` · Figure: `figures/ours_vs_rewardbench_paper_avg.pdf`

### Fine-Grained Subset Analysis

To avoid relying only on aggregate scores, we analyze subset-level accuracy across subsets including: AlpacaEval Easy/Hard/Length, MT-Bench Easy/Medium/Hard, LLMBar Natural and Adversarial variants, Refusals (Dangerous/Offensive), XSTest Should Refuse/Respond, Do-Not-Answer, Math PRM, and HumanEvalPack coding subsets.

Results file: `results/fine_grained_subset_scores.csv` · Figure: `figures/fine_grained_subset_heatmap.pdf`

### Weakest Subsets by Model

| Model | Weakest Subsets |
|---|---|
| ArmoRM 8B | llmbar-adver-GPTOut, llmbar-adver-manual, llmbar-adver-neighbor, llmbar-adver-GPTInst, donotanswer |
| Zephyr 7B DPO | llmbar-adver-GPTInst, refusals-dangerous, refusals-offensive, llmbar-adver-manual, llmbar-adver-GPTOut |
| DeBERTa RM | llmbar-adver-GPTOut, math-prm, llmbar-natural, llmbar-adver-manual, hep-js |
| Qwen 0.5B DPO | alpacaeval-easy, xstest-should-respond, alpacaeval-hard, mt-bench-easy, mt-bench-med |

### Safety Breakdown

A reward model must balance two behaviors: preferring refusals for harmful prompts while preferring helpful answers for safe prompts.

- **ArmoRM** is the strongest and most balanced safety model.
- **Qwen 0.5B DPO** shows possible over-refusal behavior (strong on refusal subsets, weak on `xstest-should-respond`).
- **Zephyr 7B DPO** shows possible under-refusal behavior (strong on `xstest-should-respond`, weak on `refusals-dangerous` and `refusals-offensive`).
- **DeBERTa RM** performs reasonably on Safety overall but struggles on harder refusal subsets.

Figure: `figures/safety_breakdown.pdf`

### Chat Hard and Reasoning Breakdown

- **ArmoRM** remains the most consistent model across hard and reasoning subsets.
- **DeBERTa** performs reasonably on standard Chat but drops sharply on adversarial and mathematical reasoning subsets.
- **Qwen 0.5B DPO** performs poorly on standard Chat but is more competitive on some Chat Hard and Reasoning subsets.
- **Zephyr 7B DPO** is strong on Chat and Reasoning but weaker on some adversarial and refusal subsets.

Figure: `figures/hard_reasoning_breakdown.pdf`

---

## Key Findings

1. **ArmoRM is the strongest and most consistent model** across all four main RewardBench sections.
2. **Aggregate scores hide important weaknesses**, especially on Chat Hard, Reasoning, and safety-refusal subsets.
3. **Strong classifier reward models appear more stable**, but not all classifiers are equally robust — DeBERTa performs well on standard Chat and Safety but struggles on hard preference distinctions.
4. **DPO-style models show section-specific tradeoffs** — Qwen performs poorly on standard Chat but is more competitive on some hard subsets, while Zephyr is strong on Chat and Reasoning but weaker in safety calibration.
5. **Fine-grained subset analysis is more informative than a single average score**, as it reveals exactly where each reward model fails.
6. **Local results broadly match the RewardBench paper's trends**, even though exact numbers differ due to environment and implementation details.

---

## Implementation Notes

### RewardBench CLI Issue
The `rewardbench` CLI entry point failed in Colab due to a Python 3.12 / `pkg_resources` compatibility issue. The workaround was to run the official source scripts (`scripts/run_rm.py`, `scripts/run_dpo.py`) directly.

### ArmoRM Custom Pipeline
ArmoRM required custom handling because its reward pipeline scores chosen and rejected responses separately rather than using the default pairwise interface expected by the RewardBench script. The local Colab copy of RewardBench was patched accordingly; the official repository was not permanently modified.

### DPO Memory Usage
DPO-style evaluation requires loading both the policy model and the reference model simultaneously, making it significantly more memory-intensive than classifier RM evaluation. After CUDA out-of-memory errors, a full Colab runtime restart was sometimes required to clear fragmented GPU memory before retrying.

### Runtime Batch Sizes

| Model | Batch Size |
|---|---|
| DeBERTa RM | 16 |
| Qwen 0.5B DPO | 1 |
| ArmoRM 8B | 1 |
| Zephyr 7B DPO | 1 |

### Hugging Face Hub Upload
Automatic Hugging Face Hub upload was disabled so that all outputs were saved locally and pushed to this repository.

---

## Acknowledgements

This project uses the official RewardBench framework:
- Repository: [https://github.com/allenai/reward-bench](https://github.com/allenai/reward-bench)
- Paper: Lambert, N. et al. *RewardBench: Evaluating Reward Models for Language Modeling.* Findings of ACL: NAACL 2025.

---

> **Note:** This is an inference-only survey project. No reward models were trained or fine-tuned. The RewardBench dataset was not modified. The notebook (`notebooks/rewardbench_survey_colab.ipynb`) is the primary workflow artifact.
---
