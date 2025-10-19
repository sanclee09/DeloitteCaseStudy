import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load your preprocessed EU data
df_eu = pd.read_csv(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/processed/df_eu_clean.csv"
)

# Define the features to check correlation
numeric_cols = [
    # Raw numerical features
    "age",
    "luggage_weight_kg",
    "total_flighttime",
    "total_traveltime",
    "layover_time",
    "layover_ratio",
    "layover_ratio_log",
    # Binary features
    "is_business",
    "has_family",
    "has_connection",
    "is_male",
    "is_long_haul",
    # Categorical ordinal features
    "layover_category",
    "flight_time_category",
    "age_group",
    # Interaction features
    "business_longhaul",
    "age_business",
    "family_luggage",
    "male_business",
    # Polynomial features
    "age_squared",
    "luggage_squared",
    "flighttime_log",
]

# Filter to only columns that exist
numeric_cols = [c for c in numeric_cols if c in df_eu.columns]

# Calculate correlations with target
correlations = df_eu[numeric_cols + ["amount_spent_cat"]].corr()["amount_spent_cat"]
correlations = correlations.drop("amount_spent_cat").sort_values(ascending=False)

# Get top 10 features
top_10_features = correlations.head(10)

print("Top 10 correlations with spending:")
print(top_10_features)

# Create a more detailed correlation matrix for top 10 features
top_features_list = top_10_features.index.tolist()
correlation_matrix = df_eu[top_features_list + ["amount_spent_cat"]].corr()

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap with better styling
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Custom colormap - diverging from negative (blue) to positive (red)
cmap = sns.diverging_palette(250, 10, as_cmap=True)

sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap=cmap,
    center=0,
    square=True,
    linewidths=1.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    vmin=-1,
    vmax=1,
    ax=ax,
)

# Improve labels
feature_labels = {
    "layover_category": "Layover Category",
    "has_connection": "Has Connection",
    "layover_ratio_log": "Layover Ratio (log)",
    "layover_ratio": "Layover Ratio",
    "layover_time": "Layover Time",
    "total_traveltime": "Total Travel Time",
    "total_flighttime": "Total Flight Time",
    "flighttime_log": "Flight Time (log)",
    "flight_time_category": "Flight Time Category",
    "is_long_haul": "Long Haul Flight",
    "amount_spent_cat": "Spending Category",
}

# Get current labels and replace with cleaner versions
current_labels = [label.get_text() for label in ax.get_xticklabels()]
clean_labels = [feature_labels.get(label, label) for label in current_labels]

ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(clean_labels, rotation=0, fontsize=10)

# Title
ax.set_title(
    "Feature Correlation Matrix: Top 10 Predictors of Spending",
    fontsize=14,
    fontweight="bold",
    pad=20,
    color="#2C3E50",
)

# Highlight the target column/row
for i, label in enumerate(ax.get_yticklabels()):
    if "Spending Category" in label.get_text():
        label.set_weight("bold")
        label.set_color("#86BC25")

for i, label in enumerate(ax.get_xticklabels()):
    if "Spending Category" in label.get_text():
        label.set_weight("bold")
        label.set_color("#86BC25")

plt.tight_layout()

# Save
plt.savefig(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("\n✓ Correlation heatmap saved to outputs/correlation_heatmap.png")
plt.show()

# Create a simpler version - just top 10 features vs target (for cleaner slide)
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Prepare data for horizontal bar chart
features = [feature_labels.get(f, f) for f in top_10_features.index]
values = top_10_features.values

# Create color map based on correlation strength
colors_bars = [
    "#86BC25" if v > 0.6 else "#00A3E0" if v > 0.4 else "#62B5E5" for v in values
]

bars = ax2.barh(
    range(len(features)),
    values,
    color=colors_bars,
    alpha=0.85,
    edgecolor="white",
    linewidth=1.5,
)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(
        val + 0.01,
        i,
        f"{val:.3f}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

# Styling
ax2.set_yticks(range(len(features)))
ax2.set_yticklabels(features, fontsize=11)
ax2.set_xlabel(
    "Correlation with Spending Category",
    fontsize=12,
    fontweight="bold",
    color="#333333",
)
ax2.set_title(
    "Top 10 Features: Correlation with Spending",
    fontsize=14,
    fontweight="bold",
    pad=15,
    color="#2C3E50",
)

# Add gridlines
ax2.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.7)
ax2.set_axisbelow(True)

# Add reference line at 0.5
ax2.axvline(
    x=0.5,
    color="red",
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label="Strong correlation (>0.5)",
)
ax2.legend(loc="lower right", fontsize=9)

# Set x-axis limits
ax2.set_xlim(0, max(values) * 1.1)

# Clean spines
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#CCCCCC")
ax2.spines["bottom"].set_color("#CCCCCC")

# Invert y-axis so highest correlation is on top
ax2.invert_yaxis()

plt.tight_layout()

# Save simpler version
plt.savefig(
    "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/outputs/correlation_bar_chart.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print("✓ Correlation bar chart saved to outputs/correlation_bar_chart.png")
plt.show()
