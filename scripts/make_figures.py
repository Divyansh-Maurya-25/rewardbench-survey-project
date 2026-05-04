"""
Figure generation helper for the RewardBench survey project.

The final figures in figures/ were generated from the processed CSV files
using the plotting cells in notebooks/rewardbench_survey_colab.ipynb.

Expected input files:
- results/section_scores.csv
- results/subset_scores.csv
- results/fine_grained_subset_scores.csv
- results/ours_vs_rewardbench_paper.csv
- results/weakest_subsets_by_model.csv

Expected output files:
- figures/section_scores_by_model.pdf
- figures/ours_vs_rewardbench_paper_avg.pdf
- figures/fine_grained_subset_heatmap.pdf
- figures/safety_breakdown.pdf
- figures/hard_reasoning_breakdown.pdf
"""

print("Final figure generation is documented in notebooks/rewardbench_survey_colab.ipynb.")
print("Processed CSV files are stored in results/.")
print("Generated PDF figures are stored in figures/.")
