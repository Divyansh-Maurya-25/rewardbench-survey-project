# RewardBench Survey Project

A survey project comparing reward model approaches using the [RewardBench](https://github.com/allenai/reward-bench) evaluation framework, with analysis across Chat, Chat Hard, Safety, and Reasoning sections.

---

## Project Goal

The goal is to compare representative reward-model families across RewardBench's four evaluation sections, focusing not just on overall performance but on **section-level and subset-level behavior** — which reveals deeper strengths and weaknesses that aggregate scores hide.

---

## Models Evaluated

| Model | Family | Notes |
|---|---|---|
| `OpenAssistant/reward-model-deberta-v3-large-v2` | Classifier RM | Lightweight classifier baseline |
| `Qwen/Qwen1.5-0.5B-Chat` | DPO-style | Small aligned chat model |
| `RLHFlow/ArmoRM-Llama3-8B-v0.1` | Classifier RM | Strong RM using ArmoRM's custom pipeline |
| `HuggingFaceH4/zephyr-7b-beta` | DPO-style | Stronger aligned DPO-style model |

---

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
  subset_scores.csv
  raw_outputs/

figures/
  section_scores_by_model.png
  model_family_average.png
  safety_breakdown.png
  hard_reasoning_breakdown.png
```

---

## Setup

```bash
pip install rewardbench
git clone https://github.com/allenai/reward-bench.git
cd reward-bench
```

> **Note:** Due to a Python 3.12 / `pkg_resources` compatibility issue with the `rewardbench` CLI in Colab, experiments were run directly via the source scripts instead:
>
> ```bash
> python scripts/run_rm.py
> python scripts/run_dpo.py
> ```

---

## Running Experiments

The main reproducibility artifact is:

```text
notebooks/rewardbench_survey_colab.ipynb
```

The notebook contains the full workflow:

1. RewardBench environment setup
2. GPU / runtime check
3. Local output folder creation
4. Hugging Face Hub upload patch (disabled for local storage)
5. ArmoRM custom pipeline patch
6. Model evaluation commands
7. CSV result generation
8. Figure generation

A command summary is also available in:

```text
scripts/run_core_models.sh
```

---

## Results

### Section-Level Scores

| Model | Chat | Chat Hard | Safety | Reasoning |
|---|---:|---:|---:|---:|
| DeBERTa RM | 0.8045 | 0.4408 | 0.7662 | 0.3776 |
| Qwen 0.5B DPO | 0.3659 | 0.6316 | 0.5662 | 0.6069 |
| ArmoRM 8B | 0.9721 | 0.7632 | 0.9041 | 0.9738 |
| Zephyr 7B DPO | 0.9246 | 0.6601 | 0.6297 | 0.7612 |

Processed result files:

```text
results/model_comparison.csv
results/section_scores.csv
results/subset_scores.csv
```

Figures:

```text
figures/section_scores_by_model.png
figures/model_family_average.png
figures/safety_breakdown.png
figures/hard_reasoning_breakdown.png
```

---

## Key Findings

- **ArmoRM** is the strongest overall model across Chat, Safety, and Reasoning.
- **DeBERTa** performs well on standard Chat and Safety but struggles on Chat Hard and Reasoning.
- **Qwen 0.5B** shows an interesting imbalance: weak on Chat, but stronger than DeBERTa on Chat Hard and Reasoning.
- **Zephyr** performs strongly on Chat and Reasoning but shows weaker Safety calibration, especially on refusal-related subsets.
- Subset-level plots reveal that a single aggregate score can hide significant differences in refusal behavior, adversarial judgment, and reasoning ability.

---

## Runtime Notes

Experiments were run in Google Colab on an NVIDIA A100 GPU.

DPO-style evaluation requires more memory than classifier RM evaluation because it loads both a policy model and a reference model, computing token-level log probabilities for chosen and rejected completions.

### Batch sizes used

| Model | Batch Size |
|---|---:|
| DeBERTa RM | 16 |
| Qwen 0.5B DPO | 1 |
| ArmoRM 8B | 1 |
| Zephyr 7B DPO | 1 |

> After CUDA out-of-memory errors, a full runtime restart was required before retrying — GPU memory can remain fragmented otherwise.

---

## Implementation Notes

### 1. RewardBench CLI issue
The `rewardbench` CLI entry point failed due to a Python 3.12 + `pkg_resources` incompatibility. Resolved by running `scripts/run_rm.py` and `scripts/run_dpo.py` directly.

### 2. Hugging Face Hub upload
Hub upload was disabled to keep results stored locally and in this repository.

### 3. ArmoRM pipeline patch
`run_rm.py` assumed an input format incompatible with ArmoRM. Fixed by patching the script to score chosen and rejected responses separately before comparing them.

### 4. DPO memory constraint
DPO evaluation required batch size 1 due to the dual-model memory footprint, resulting in slower execution.

All patches were applied only to the temporary Colab copy of RewardBench. The official repository was not modified.

---

## Regenerating Figures

After placing CSV files in `results/`, regenerate all figures with:

```bash
python scripts/make_figures.py
```

---

## Notes on Result Files

CSVs contain processed results copied from final RewardBench evaluation outputs. The original RewardBench dataset is not included — only processed result tables and generated figures.

> **Colab users:** Colab storage resets after runtime restart. Download results or push to GitHub before ending your session.

---

## Acknowledgements

This project uses the RewardBench evaluation framework:
[https://github.com/allenai/reward-bench](https://github.com/allenai/reward-bench)

---

## Final Notes

- The notebook (`notebooks/rewardbench_survey_colab.ipynb`) is the primary reproducibility artifact
- This repository supplements, but does not replace, the original RewardBench repository
- All results were derived from RewardBench evaluation outputs
