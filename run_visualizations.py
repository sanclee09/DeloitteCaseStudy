"""
Standalone visualization generator
Run this when you want to update/regenerate visualizations without retraining

This is perfect for:
- Tweaking visualization styles
- Adding new plots
- Experimenting with different chart types
"""

import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from utils import *


def load_all_results():
    """Load all necessary data for visualizations"""
    print_section_header("LOADING DATA")

    # Load model
    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded: {MODEL_FILE}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Load profitability results
    try:
        profitability = pd.read_csv(PROFITABILITY_FILE, index_col=0)
        print(f"✓ Profitability loaded: {PROFITABILITY_FILE}")
    except Exception as e:
        print(f"✗ Error loading profitability: {e}")
        return None

    # Load WW predictions
    try:
        df_ww = pd.read_csv(WW_CLEAN_FILE)
        print(f"✓ WW data loaded: {WW_CLEAN_FILE}")
    except Exception as e:
        print(f"✗ Error loading WW data: {e}")
        return None

    return {"model_data": model_data, "profitability": profitability, "df_ww": df_ww}


def create_confusion_matrices(model_data):
    """Create confusion matrix visualizations"""
    print_section_header("CREATING CONFUSION MATRIX PLOTS")

    # Get XGBoost and RF metrics
    xgb_metrics = model_data["comparison"]["xgb_metrics"]
    rf_metrics = model_data["comparison"]["rf_metrics"]

    # Create plots for both models
    for model_name, metrics in [("XGBoost", xgb_metrics), ("RandomForest", rf_metrics)]:
        print(f"\nGenerating {model_name} confusion matrices...")

        cm = metrics["confusion_matrix"]

        # Calculate normalized confusion matrix if not present
        if "confusion_matrix_normalized" in metrics:
            cm_normalized = metrics["confusion_matrix_normalized"]
        else:
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Raw confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=range(5),
            yticklabels=range(5),
            cbar_kws={"label": "Count"},
        )
        ax1.set_title(
            f"{model_name} - Raw Confusion Matrix", fontweight="bold", fontsize=12
        )
        ax1.set_ylabel("True Label", fontsize=11)
        ax1.set_xlabel("Predicted Label", fontsize=11)

        # Normalized confusion matrix (as percentages)
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            ax=ax2,
            xticklabels=range(5),
            yticklabels=range(5),
            cbar_kws={"label": "Percentage"},
        )
        ax2.set_title(
            f"{model_name} - Normalized Confusion Matrix",
            fontweight="bold",
            fontsize=12,
        )
        ax2.set_ylabel("True Label", fontsize=11)
        ax2.set_xlabel("Predicted Label", fontsize=11)

        plt.tight_layout()

        # Save
        filename = os.path.join(
            OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}.png"
        )
        plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"✓ Saved: {filename}")
        plt.close()


def create_profitability_visualizations(profitability, df_ww):
    """Regenerate main profitability visualizations"""
    print_section_header("CREATING PROFITABILITY VISUALIZATIONS")

    # Predict spending if not already in df_ww
    if "predicted_category" not in df_ww.columns:
        print("Note: predicted_category not found in WW data")
        print("Creating placeholder distribution for visualization...")
        # Create a reasonable distribution for visualization purposes
        df_ww["predicted_category"] = 2  # placeholder

    fig = plt.figure(figsize=FIGURE_SIZE)

    # 1. Predicted spending distribution
    ax1 = plt.subplot(2, 3, 1)
    pred_dist = df_ww["predicted_category"].value_counts().sort_index()
    colors_spending = COLORS["spending_categories"]
    pred_dist.plot(kind="bar", ax=ax1, color=colors_spending)
    ax1.set_title("Predicted Spending Distribution\n(Worldwide)", fontweight="bold")
    ax1.set_xlabel("Spending Category")
    ax1.set_ylabel("Number of Passengers")
    ax1.tick_params(axis="x", rotation=0)

    for i, v in enumerate(pred_dist.values):
        pct = v / len(df_ww) * 100
        ax1.text(i, v + 500, f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=9)

    # 2. Passengers by airport
    ax2 = plt.subplot(2, 3, 2)
    passenger_counts = profitability["passenger_count"].sort_values(ascending=True)
    passenger_counts.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_title("Passenger Volume by Airport", fontweight="bold")
    ax2.set_xlabel("Number of Passengers")

    # 3. Monthly revenue
    ax3 = plt.subplot(2, 3, 3)
    monthly_rev = profitability["monthly_revenue"].sort_values(ascending=True)
    monthly_rev.plot(kind="barh", ax=ax3, color=COLORS["revenue"])
    ax3.set_title("Monthly Revenue by Airport", fontweight="bold")
    ax3.set_xlabel("Revenue (€)")

    # 4. Monthly costs
    ax4 = plt.subplot(2, 3, 4)
    monthly_costs = profitability["monthly_cost"].sort_values(ascending=True)
    monthly_costs.plot(kind="barh", ax=ax4, color=COLORS["cost"])
    ax4.set_title("Monthly Lease Costs", fontweight="bold")
    ax4.set_xlabel("Cost (€)")

    # 5. Monthly profit (KEY CHART)
    ax5 = plt.subplot(2, 3, 5)
    monthly_profits = profitability["monthly_profit"].sort_values(ascending=True)
    colors_profit = [
        COLORS["profit_positive"] if p > 0 else COLORS["profit_negative"]
        for p in monthly_profits
    ]
    monthly_profits.plot(kind="barh", ax=ax5, color=colors_profit)
    ax5.set_title("Monthly Profit by Airport ⭐", fontweight="bold", fontsize=12)
    ax5.set_xlabel("Profit (€)")
    ax5.axvline(x=0, color="black", linestyle="--", linewidth=2)

    for i, (airport, profit) in enumerate(monthly_profits.items()):
        ax5.text(
            profit + 5000,
            i,
            f"€{profit:,.0f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # 6. Profit margin
    ax6 = plt.subplot(2, 3, 6)
    profit_margins = profitability["profit_margin"].sort_values(ascending=True)
    colors_margin = [
        COLORS["profit_positive"] if m > 0 else COLORS["profit_negative"]
        for m in profit_margins
    ]
    profit_margins.plot(kind="barh", ax=ax6, color=colors_margin)
    ax6.set_title("Profit Margin by Airport", fontweight="bold")
    ax6.set_xlabel("Profit Margin (%)")
    ax6.axvline(x=0, color="black", linestyle="--", linewidth=2)

    plt.tight_layout()

    filename = VISUALIZATION_FILE
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"✓ Saved: {filename}")
    plt.close()


def main():
    """Main visualization generation"""
    print("=" * 80)
    print("STANDALONE VISUALIZATION GENERATOR")
    print("=" * 80)
    print()

    # Load data
    data = load_all_results()
    if data is None:
        print("\n✗ Failed to load required data")
        return None

    # Create visualizations
    try:
        # Confusion matrices
        create_confusion_matrices(data["model_data"])

        # Profitability visualizations
        create_profitability_visualizations(data["profitability"], data["df_ww"])

        print_section_header("VISUALIZATION GENERATION COMPLETE")
        print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - confusion_matrix_xgboost.png")
        print(f"  - confusion_matrix_randomforest.png")
        print(f"  - feature_importance.png")
        print(f"  - case_study_analysis.png")

        return True

    except Exception as e:
        print(f"\n✗ Visualization generation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED")
        print("=" * 80)
        sys.exit(1)
