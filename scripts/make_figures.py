import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("results/model_comparison.csv")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

score_cols = ["chat", "chat_hard", "safety", "reasoning"]

for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Section-wise bar chart
plot_df = df.set_index("model")[score_cols]
ax = plot_df.plot(kind="bar", figsize=(10, 5))

ax.set_title("RewardBench Section-wise Scores by Model")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Model")
ax.legend(title="Section")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "section_scores.png", dpi=300)
plt.close()

# Model-family average chart
family_df = df.groupby("type")[score_cols].mean()
ax = family_df.plot(kind="bar", figsize=(8, 5))

ax.set_title("Average Section Scores by Model Family")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Model Family")
ax.legend(title="Section")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_family_average.png", dpi=300)
plt.close()

print("Figures saved to figures/")
