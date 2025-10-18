import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the WW predictions (assuming you have predictions saved)
df_ww = pd.read_csv(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/processed/df_ww_clean.csv"
)

# If predictions exist in the file, use them; otherwise load from prediction output
# For this script, I'll assume predictions are already in the file from your pipeline run
# If not, you'll need to run the prediction first

# Check if predictions exist
if "predicted_category" not in df_ww.columns:
    print("⚠️  Predictions not found in df_ww_clean.csv")
    print("Please run the prediction pipeline first or load predicted data")

    # Create dummy data for visualization purposes (replace with actual predictions)
    print("\nUsing example distribution from your output...")
    categories = ["0", "1", "2", "3", "4"]
    counts = [21278, 10269, 9285, 17154, 9780]
    percentages = [31.40, 15.15, 13.70, 25.31, 14.43]
    total = sum(counts)
else:
    # Get actual predicted distribution
    pred_dist = df_ww["predicted_category"].value_counts().sort_index()
    categories = [str(i) for i in pred_dist.index]
    counts = pred_dist.values
    total = len(df_ww)
    percentages = [(c / total * 100) for c in counts]

print(f"Total passengers: {total:,}")
print("\nPredicted distribution:")
for cat, count, pct in zip(categories, counts, percentages):
    print(f"  Category {cat}: {count:,} ({pct:.2f}%)")

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars with better width and elegant colors
bar_width = 0.6
x_pos = np.arange(len(categories))

# Deloitte color palette - professional blues and green
colors = ["#86BC25", "#62B5E5", "#00A3E0", "#0076A8", "#012169"]

bars = ax.bar(
    x_pos, counts, bar_width, color=colors, alpha=0.85, edgecolor="white", linewidth=2
)

# Add percentage labels on top of bars with better positioning
for i, (bar, pct, count) in enumerate(zip(bars, percentages, counts)):
    height = bar.get_height()
    # Position text higher to avoid overlap
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + max(counts) * 0.03,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    # Add count below percentage
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + max(counts) * 0.005,
        f"({count:,})",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#666666",
    )

# Styling
ax.set_xlabel(
    "Predicted Spending Category", fontsize=13, fontweight="600", color="#333333"
)
ax.set_ylabel("Number of Passengers", fontsize=13, fontweight="600", color="#333333")
ax.set_title(
    "Worldwide Predictions: Spending Distribution",
    fontsize=16,
    fontweight="bold",
    pad=20,
    color="#2C3E50",
)

# Add category ranges as x-axis labels
category_labels = ["€0-10", "€10-50", "€50-150", "€150-300", "€300-500"]
ax.set_xticks(x_pos)
ax.set_xticklabels(category_labels, fontsize=11, color="#333333")

# Format y-axis with thousands separator
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
ax.tick_params(axis="y", labelsize=10, colors="#666666")

# Add subtle gridlines
ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5, color="#CCCCCC")
ax.set_axisbelow(True)

# Set y-axis limit to give more space at the top
ax.set_ylim(0, max(counts) * 1.15)

# Add total count annotation with cleaner styling
ax.text(
    0.02,
    0.98,
    f"Total: {total:,} passengers",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#F8F9FA",
        edgecolor="#DEE2E6",
        alpha=0.9,
        linewidth=1.5,
    ),
    color="#495057",
)

# Add confidence annotation
avg_confidence = 82.21  # From your output
ax.text(
    0.98,
    0.89,
    f"Avg Confidence: {avg_confidence:.1f}%",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#D1F2EB",
        edgecolor="#A9DFBF",
        alpha=0.9,
        linewidth=1.5,
    ),
    color="#1E8449",
)

# Add model annotation
ax.text(
    0.98,
    0.98,
    "XGBoost Model\nF1: 92%",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#E8F8F5",
        edgecolor="#A9DFBF",
        alpha=0.9,
        linewidth=1.5,
    ),
    color="#1E8449",
)

# Remove top and right spines for cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")

# Clean up layout
plt.tight_layout()

# Save in high resolution for presentation
plt.savefig(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/outputs/predicted_spending_distribution.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print(
    "\n✓ Predicted spending distribution chart saved to outputs/predicted_spending_distribution.png"
)
plt.show()

# ========================================
# Create a comparison version (EU vs WW)
# ========================================

# Load EU data for comparison
df_eu = pd.read_csv(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/processed/df_eu_clean.csv"
)
eu_dist = df_eu["amount_spent_cat"].value_counts().sort_index()
eu_percentages = (eu_dist / len(df_eu) * 100).values

fig2, ax2 = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(categories))
width = 0.35

# EU bars (training data)
bars1 = ax2.bar(
    x_pos - width / 2,
    eu_percentages,
    width,
    label="EU (Training)",
    color="#62B5E5",
    alpha=0.85,
    edgecolor="white",
    linewidth=1.5,
)

# WW bars (predictions)
bars2 = ax2.bar(
    x_pos + width / 2,
    percentages,
    width,
    label="WW (Predicted)",
    color="#86BC25",
    alpha=0.85,
    edgecolor="white",
    linewidth=1.5,
)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

# Styling
ax2.set_xlabel("Spending Category", fontsize=13, fontweight="600", color="#333333")
ax2.set_ylabel(
    "Percentage of Passengers", fontsize=13, fontweight="600", color="#333333"
)
ax2.set_title(
    "Spending Distribution: EU Training vs WW Predictions",
    fontsize=16,
    fontweight="bold",
    pad=20,
    color="#2C3E50",
)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(category_labels, fontsize=11, color="#333333")

# Add legend
ax2.legend(loc="upper right", fontsize=11, framealpha=0.9)

# Add gridlines
ax2.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5, color="#CCCCCC")
ax2.set_axisbelow(True)

# Set y-axis limit
ax2.set_ylim(0, max(max(eu_percentages), max(percentages)) * 1.15)

# Format y-axis as percentage
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}%"))

# Remove top and right spines
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#CCCCCC")
ax2.spines["bottom"].set_color("#CCCCCC")

plt.tight_layout()

# Save comparison version
plt.savefig(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/outputs/spending_distribution_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("✓ Comparison chart saved to outputs/spending_distribution_comparison.png")
plt.show()
