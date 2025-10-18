import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from your output
categories = ["0", "1", "2", "3", "4"]
counts = [39170, 22841, 31444, 22384, 16395]
percentages = [29.62, 17.27, 23.78, 16.93, 12.40]

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars with better width and elegant colors
bar_width = 0.6
x_pos = np.arange(len(categories))
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
        height + 1500,
        f"{pct}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    # Add count below percentage
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 500,
        f"({count:,})",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#666666",
    )

# Styling
ax.set_xlabel("Spending Category", fontsize=13, fontweight="600", color="#333333")
ax.set_ylabel("Number of Passengers", fontsize=13, fontweight="600", color="#333333")
ax.set_title(
    "EU Dataset: Class Distribution (Imbalanced)",
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
total = sum(counts)
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

# Add imbalance indicator with cleaner styling
max_count = max(counts)
min_count = min(counts)
imbalance_ratio = max_count / min_count
ax.text(
    0.02,
    0.91,
    f"Imbalance ratio: {imbalance_ratio:.1f}:1",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#FFF3CD",
        edgecolor="#FFE69C",
        alpha=0.9,
        linewidth=1.5,
    ),
    color="#856404",
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
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/outputs/class_imbalance_chart.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("✓ Class imbalance chart saved to outputs/class_imbalance_chart.png")
plt.show()
