# Load result CSV files

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
results_dir = Path("/content/rewardbench_results")
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Read CSV files
model_df = pd.read_csv(results_dir / "model_comparison.csv")
section_df = pd.read_csv(results_dir / "section_scores.csv")
subset_df = pd.read_csv(results_dir / "subset_scores.csv")

# Short names for cleaner plots
name_map = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "DeBERTa RM",
    "Qwen/Qwen1.5-0.5B-Chat": "Qwen 0.5B DPO",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": "ArmoRM 8B",
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B DPO",
}

model_df["short_model"] = model_df["model"].map(name_map)
subset_df["short_model"] = subset_df["model"].map(name_map)

# Clean model-family labels for final comparison

import pandas as pd
from pathlib import Path

results_dir = Path("/content/rewardbench_results")
model_path = results_dir / "model_comparison.csv"

model_df = pd.read_csv(model_path)

# Group ArmoRM with classifier reward models for family-level comparison
model_df.loc[
    model_df["model"] == "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    "type"
] = "Classifier RM"

model_df.to_csv(model_path, index=False)

display(model_df)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
results_dir = Path("/content/rewardbench_results")
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Read CSV files
model_df = pd.read_csv(results_dir / "model_comparison.csv")
section_df = pd.read_csv(results_dir / "section_scores.csv")
subset_df = pd.read_csv(results_dir / "subset_scores.csv")

# Short names for cleaner plots (re-added as it was lost in a previous cell execution)
name_map = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "DeBERTa RM",
    "Qwen/Qwen1.5-0.5B-Chat": "Qwen 0.5B DPO",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": "ArmoRM 8B",
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B DPO",
}
model_df["short_model"] = model_df["model"].map(name_map)

# Figure 1: RewardBench section scores by model

score_cols = ["chat", "chat_hard", "safety", "reasoning"]

plot_df = model_df.set_index("short_model")[score_cols]
plot_df = plot_df.rename(columns={
    "chat": "Chat",
    "chat_hard": "Chat Hard",
    "safety": "Safety",
    "reasoning": "Reasoning",
})

# Professional color palette (Seaborn 'deep' inspired)
prof_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(10, 6))

plot_df.plot(kind="bar", ax=ax, color=prof_colors)

ax.set_title("RewardBench Section Scores by Model", fontsize=14, pad=12)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.15) # Increased to make room for labels
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Section", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(figures_dir / "section_scores_by_model.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 2: Average section scores by model family

family_df = model_df.groupby("type")[score_cols].mean()
family_df = family_df.rename(columns={
    "chat": "Chat",
    "chat_hard": "Chat Hard",
    "safety": "Safety",
    "reasoning": "Reasoning",
})

# Professional color palette
prof_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(10, 6))

family_df.plot(kind="bar", ax=ax, color=prof_colors)

ax.set_title("Average Section Scores by Model Family", fontsize=14, pad=12)
ax.set_xlabel("Model Family", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.15) # Increased to make room for labels
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Section", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(figures_dir / "model_family_average.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 3: Safety subset breakdown

name_map = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "DeBERTa RM",
    "Qwen/Qwen1.5-0.5B-Chat": "Qwen 0.5B DPO",
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": "ArmoRM 8B",
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B DPO",
}
if "short_model" not in subset_df.columns:
    subset_df["short_model"] = subset_df["model"].map(name_map)

safety_subsets = [
    "donotanswer",
    "refusals-dangerous",
    "refusals-offensive",
    "xstest-should-refuse",
    "xstest-should-respond",
]

safety_df = subset_df[subset_df["subset"].isin(safety_subsets)].copy()
pivot = safety_df.pivot(index="short_model", columns="subset", values="score")

# Professional color palette
prof_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

fig, ax = plt.subplots(figsize=(11, 6)) # Adjusted height slightly

pivot.plot(kind="bar", ax=ax, color=prof_colors)

ax.set_title("Safety and Refusal-Calibration Subset Scores", fontsize=14, pad=12)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.15) # Increased to make room for labels
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Subset", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8) # Slightly smaller font for subsets

plt.xticks(rotation=0, ha="center") # Straighten x-labels
plt.tight_layout()
plt.savefig(figures_dir / "safety_breakdown.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 4: Hard chat and reasoning subset breakdown

selected_subsets = [
    "llmbar-natural",
    "llmbar-adver-GPTOut",
    "llmbar-adver-neighbor",
    "math-prm",
    "mt-bench-hard",
]

hard_df = subset_df[subset_df["subset"].isin(selected_subsets)].copy()
pivot = hard_df.pivot(index="short_model", columns="subset", values="score")

# Professional color palette
prof_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

fig, ax = plt.subplots(figsize=(11, 6))

pivot.plot(kind="bar", ax=ax, color=prof_colors)

ax.set_title("Selected Chat Hard and Reasoning Subset Scores", fontsize=14, pad=12)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0, 1.15) # Increased to make room for labels
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(title="Subset", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)

plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(figures_dir / "hard_reasoning_breakdown.png", dpi=300, bbox_inches="tight")
plt.show()
