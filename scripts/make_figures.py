"""
Generate figures for the RewardBench survey project.

This script reads processed CSV files from results/ and writes vector PDF
figures to figures/. It is intended to make the repository more reproducible
than relying only on notebook plotting cells.

Expected input files:
- results/model_comparison.csv
- results/fine_grained_subset_scores.csv
- results/ours_vs_rewardbench_paper.csv

Optional input files:
- results/section_scores.csv
- results/subset_scores.csv

Generated output files:
- figures/section_scores_by_model.pdf
- figures/ours_vs_rewardbench_paper_avg.pdf
- figures/fine_grained_subset_heatmap.pdf
- figures/safety_breakdown.pdf
- figures/hard_reasoning_breakdown.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------

MODEL_ORDER = [
    "ArmoRM 8B",
    "Zephyr 7B DPO",
    "DeBERTa RM",
    "Qwen 0.5B DPO",
]

SECTION_ORDER = ["Chat", "Chat Hard", "Safety", "Reasoning"]

SECTION_COLUMN_MAP = {
    "Chat": "chat",
    "Chat Hard": "chat_hard",
    "Safety": "safety",
    "Reasoning": "reasoning",
}

SAFETY_SUBSETS = [
    "refusals-dangerous",
    "refusals-offensive",
    "xstest-should-refuse",
    "xstest-should-respond",
    "donotanswer",
]

HARD_REASONING_SUBSETS = [
    "llmbar-natural",
    "llmbar-adver-GPTOut",
    "llmbar-adver-manual",
    "math-prm",
    "hep-python",
    "hep-js",
    "hep-rust",
]


def require_file(path: Path) -> None:
    """Raise a helpful error if an expected CSV file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Make sure the processed CSV files are stored in results/."
        )


def save_current_figure(filename: str) -> None:
    """Save the active matplotlib figure as a tight vector PDF."""
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------
# Figure 1: Section-level scores by model
# ---------------------------------------------------------------------

def make_section_scores_by_model() -> None:
    """Create grouped bar chart of Chat, Chat Hard, Safety, Reasoning."""
    model_path = RESULTS_DIR / "model_comparison.csv"
    require_file(model_path)

    df = pd.read_csv(model_path)
    df = df.set_index("model_short").loc[MODEL_ORDER].reset_index()

    x = np.arange(len(SECTION_ORDER))
    width = 0.18

    plt.figure(figsize=(8.5, 4.8))

    for i, model in enumerate(MODEL_ORDER):
        row = df[df["model_short"] == model].iloc[0]
        values = [row[SECTION_COLUMN_MAP[section]] for section in SECTION_ORDER]
        positions = x + (i - 1.5) * width

        bars = plt.bar(positions, values, width, label=model)

        # Add compact value labels above bars.
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, SECTION_ORDER)
    plt.ylim(0, 1.08)
    plt.ylabel("Accuracy")
    plt.title("RewardBench Section-Level Accuracy by Model")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.25)

    save_current_figure("section_scores_by_model.pdf")


# ---------------------------------------------------------------------
# Figure 2: This project vs RewardBench paper four-section average
# ---------------------------------------------------------------------

def make_ours_vs_paper_avg() -> None:
    """Create grouped bar chart comparing local and paper averages."""
    compare_path = RESULTS_DIR / "ours_vs_rewardbench_paper.csv"
    require_file(compare_path)

    df = pd.read_csv(compare_path)
    df = df.set_index("model_short").loc[MODEL_ORDER].reset_index()

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    ours = df["avg_four_sections"].to_numpy()
    paper = df["paper_avg_four_sections"].to_numpy()

    plt.figure(figsize=(8.5, 4.8))

    bars_ours = plt.bar(x - width / 2, ours, width, label="This project")
    bars_paper = plt.bar(x + width / 2, paper, width, label="RewardBench paper")

    for bars in [bars_ours, bars_paper]:
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, MODEL_ORDER, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Average Accuracy")
    plt.title("Four-Section Average: This Project vs RewardBench Paper")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)

    save_current_figure("ours_vs_rewardbench_paper_avg.pdf")


# ---------------------------------------------------------------------
# Figure 3: Fine-grained subset heatmap
# ---------------------------------------------------------------------

def make_fine_grained_heatmap() -> None:
    """Create heatmap of subset-level scores across all evaluated models."""
    heatmap_path = RESULTS_DIR / "fine_grained_subset_scores.csv"
    require_file(heatmap_path)

    df = pd.read_csv(heatmap_path)

    label_col = "subset_label"
    df[label_col] = df["section"] + " | " + df["subset"]

    heatmap_values = df.set_index(label_col)[MODEL_ORDER]

    plt.figure(figsize=(8.5, 9.5))
    image = plt.imshow(heatmap_values.to_numpy(), aspect="auto", vmin=0, vmax=1)

    plt.colorbar(image, label="Accuracy")
    plt.xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER, rotation=25, ha="right")
    plt.yticks(np.arange(len(heatmap_values.index)), heatmap_values.index, fontsize=8)

    for row_idx in range(heatmap_values.shape[0]):
        for col_idx in range(heatmap_values.shape[1]):
            value = heatmap_values.iloc[row_idx, col_idx]
            plt.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.title("Fine-Grained RewardBench Subset Accuracy")

    save_current_figure("fine_grained_subset_heatmap.pdf")


# ---------------------------------------------------------------------
# Figure 4: Safety breakdown
# ---------------------------------------------------------------------

def make_safety_breakdown() -> None:
    """Create grouped bar chart for refusal and over-refusal behavior."""
    subset_path = RESULTS_DIR / "fine_grained_subset_scores.csv"
    require_file(subset_path)

    df = pd.read_csv(subset_path)
    df = df[df["subset"].isin(SAFETY_SUBSETS)].copy()

    # Preserve manual ordering.
    df["subset"] = pd.Categorical(df["subset"], categories=SAFETY_SUBSETS, ordered=True)
    df = df.sort_values("subset")

    x = np.arange(len(SAFETY_SUBSETS))
    width = 0.18

    plt.figure(figsize=(9.0, 4.8))

    for i, model in enumerate(MODEL_ORDER):
        values = df[model].to_numpy()
        positions = x + (i - 1.5) * width
        plt.bar(positions, values, width, label=model)

    plt.xticks(x, SAFETY_SUBSETS, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Safety Breakdown: Refusal and Over-Refusal Behavior")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.25)

    save_current_figure("safety_breakdown.pdf")


# ---------------------------------------------------------------------
# Figure 5: Chat Hard and Reasoning diagnostic breakdown
# ---------------------------------------------------------------------

def make_hard_reasoning_breakdown() -> None:
    """Create grouped bar chart for selected Chat Hard and Reasoning subsets."""
    subset_path = RESULTS_DIR / "fine_grained_subset_scores.csv"
    require_file(subset_path)

    df = pd.read_csv(subset_path)
    df = df[df["subset"].isin(HARD_REASONING_SUBSETS)].copy()

    # Preserve manual ordering.
    df["subset"] = pd.Categorical(
        df["subset"],
        categories=HARD_REASONING_SUBSETS,
        ordered=True,
    )
    df = df.sort_values("subset")

    x = np.arange(len(HARD_REASONING_SUBSETS))
    width = 0.18

    plt.figure(figsize=(9.0, 4.8))

    for i, model in enumerate(MODEL_ORDER):
        values = df[model].to_numpy()
        positions = x + (i - 1.5) * width
        plt.bar(positions, values, width, label=model)

    plt.xticks(x, HARD_REASONING_SUBSETS, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Diagnostic Subsets: Chat Hard and Reasoning Failures")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(axis="y", alpha=0.25)

    save_current_figure("hard_reasoning_breakdown.pdf")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main() -> None:
    make_section_scores_by_model()
    make_ours_vs_paper_avg()
    make_fine_grained_heatmap()
    make_safety_breakdown()
    make_hard_reasoning_breakdown()

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
