The notebook covers the full workflow: environment setup, RewardBench installation, GPU check, result folder creation, Hub upload disabling, ArmoRM custom pipeline handling, model evaluation, log saving, CSV parsing, figure generation, and result summaries.

A command summary is also available in `scripts/run_core_models.sh`.

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

---

## Method

RewardBench evaluates reward models on prompt/chosen/rejected triples. A model is correct when it assigns a higher score to the chosen response than the rejected one.

Three model interface types are evaluated:

- **Sequence Classifier RM** — standard HuggingFace classification head outputting a scalar reward
- **Custom Classifier RM** — reward scoring via a custom pipeline (ArmoRM)
- **DPO-style implicit RM** — preference scores derived from policy/reference model log-probability differences; more memory-intensive since both models must be loaded

---

## Results

### Section-Level Accuracy

| Model | Chat | Chat Hard | Safety | Reasoning | Avg |
|---|---|---|---|---|---|
| ArmoRM 8B | 0.9721 | 0.7632 | 0.9041 | 0.9738 | **0.9033** |
| Zephyr 7B DPO | 0.9553 | 0.5899 | 0.6689 | 0.7879 | 0.7505 |
| DeBERTa RM | 0.8045 | 0.4408 | 0.7662 | 0.3776 | 0.5973 |
| Qwen 0.5B DPO | 0.3659 | 0.6316 | 0.5662 | 0.6069 | 0.5427 |

Results stored in `results/section_scores.csv`. Figure: `figures/section_scores_by_model.pdf`.

### Comparison with RewardBench Paper

| Model | This Project | RewardBench Paper | Difference |
|---|---|---|---|
| ArmoRM 8B | 0.903 | 0.908 | −0.005 |
| Zephyr 7B DPO | 0.751 | 0.742 | +0.008 |
| DeBERTa RM | 0.597 | 0.538 | +0.060 |
| Qwen 0.5B DPO | 0.543 | 0.561 | −0.018 |

> The original paper's overall score includes Prior Sets; this project uses a four-section average (Chat, Chat Hard, Safety, Reasoning) for the comparison.

Results in `results/ours_vs_rewardbench_paper.csv`. Figure: `figures/ours_vs_rewardbench_paper_avg.pdf`.

### Fine-Grained Subset Analysis

Subset-level analysis spans AlpacaEval, MT-Bench, LLMBar adversarial, XSTest safety, Do-Not-Answer, Math PRM, and HumanEvalPack coding subsets. Results in `results/fine_grained_subset_scores.csv`. Figure: `figures/fine_grained_subset_heatmap.pdf`.

### Safety Breakdown

Safety performance reflects four distinct behaviors: correct refusal, correct response to safe requests, over-refusal, and under-refusal. Figure: `figures/safety_breakdown.pdf`.

Key observations:
- **ArmoRM** is the strongest overall on safety
- **Zephyr** is much stronger on `xstest-should-respond` than on `refusal-dangerous`/`refusal-offensive`
- **Qwen** shows the opposite: stronger refusal but much weaker response behavior on safe prompts

### Chat Hard & Reasoning Breakdown

Figure: `figures/hard_reasoning_breakdown.pdf`.

Key observations:
- DeBERTa performs reasonably on standard Chat but drops sharply on adversarial and mathematical reasoning subsets
- Qwen performs poorly on standard Chat but is more competitive on some Chat Hard and Reasoning subsets

### Weakest Subsets by Model

Full table in `results/weakest_subsets_by_model.csv`.

| Model | Weakest Subsets |
|---|---|
| ArmoRM 8B | `llmbar-adver-GPTOut`, `llmbar-adver-manual`, `llmbar-adver-neighbor` |
| DeBERTa RM | `llmbar-adver-GPTOut`, `math-prm`, `llmbar-natural`, `llmbar-adver-manual`, `hep-js` |
| Qwen 0.5B DPO | `alpacaeval-easy`, `xstest-should-respond`, `alpacaeval-hard`, `mt-bench-easy`, `mt-bench-med` |
| Zephyr 7B DPO | `llmbar-adver-GPTInst`, `refusals-dangerous`, `refusals-offensive`, `llmbar-adver-manual`, `llmbar-adver-GPTOut` |

---

## Key Findings

1. **ArmoRM is the strongest and most consistent model**, achieving the best four-section average with strong performance across Chat, Safety, and Reasoning.
2. **Zephyr is strong on Chat and Reasoning but weaker on Safety**, especially on refusal-dangerous and refusal-offensive subsets.
3. **DeBERTa performs well on standard Chat and Safety but struggles on hard preference distinctions** — its low Chat Hard and Reasoning scores reveal the limits of lightweight classifier-based reward models.
4. **Qwen 0.5B has uneven behavior** — poor on standard Chat but comparatively stronger on Chat Hard and Reasoning than its Chat score would suggest.
5. **Fine-grained analysis is more informative than aggregate scores**, revealing meaningful differences in refusal behavior, adversarial robustness, and reasoning ability.
6. **Local results broadly preserve the RewardBench paper trend** — ArmoRM remains the strongest, Zephyr strong but less balanced, and smaller baselines show more section-specific weaknesses.

---

## Runtime Notes

| Model | Batch Size |
|---|---|
| DeBERTa RM | 16 |
| Qwen 0.5B DPO | 1 |
| ArmoRM 8B | 1 |
| Zephyr 7B DPO | 1 |

DPO-style evaluation requires loading both a policy and reference model, making it more memory-intensive. After CUDA out-of-memory errors, a full Colab runtime restart was sometimes required before retrying.

---

## Implementation Notes

**RewardBench CLI issue:** The `rewardbench` CLI entry point failed in Colab due to a Python 3.12 / `pkg_resources` incompatibility. Workaround: run source scripts directly (`run_rm.py`, `run_dpo.py`).

**HuggingFace Hub upload:** Disabled so all outputs were saved locally and pushed to this repository.

**ArmoRM custom pipeline:** The local Colab copy of RewardBench was patched to score chosen and rejected responses separately before comparison.

**DPO memory usage:** Loading both policy and reference models made Qwen and Zephyr runs significantly more memory-intensive than classifier RM runs.

All patches were applied only to the temporary Colab copy of RewardBench. The official repository was not modified.

---

## Regenerating Figures

Figures were generated from the processed CSVs using the plotting cells in `notebooks/rewardbench_survey_colab.ipynb`. A helper script is also available:

```bash
python scripts/make_figures.py
```

The notebook remains the primary figure-generation artifact.

---

## Reproducing the Project

1. Clone the official RewardBench repository: `git clone https://github.com/allenai/reward-bench.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `notebooks/rewardbench_survey_colab.ipynb`
4. Reference commands: `scripts/run_core_models.sh`
5. Compare generated results with the processed CSVs in `results/`

---

## Acknowledgements

This project uses the official [RewardBench framework](https://github.com/allenai/reward-bench).

> Lambert, N. et al. *RewardBench: Evaluating Reward Models for Language Modeling.*> ```

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
