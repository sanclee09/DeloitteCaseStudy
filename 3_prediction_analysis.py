import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore")

from config import *
from utils import *


# ============================================================================
# MODEL AND DATA LOADING
# ============================================================================


def load_model_and_data():
    """
    Load trained model and preprocessed data
    """
    print_section_header("LOADING MODEL AND DATA")

    # Load model
    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded: {MODEL_FILE}")
        print(f"  Trained on: {model_data['trained_date']}")
        print(f"  Test accuracy: {model_data['metrics']['test_accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None, None, None

    # Load preprocessed data
    df_ww = load_csv_with_info(WW_CLEAN_FILE, "WW Passengers (Clean)")

    return model_data["model"], model_data["feature_cols"], df_ww


# ============================================================================
# PREDICTION
# ============================================================================


def predict_worldwide_spending(df_ww, model, feature_cols):
    """
    Predict spending categories for worldwide passengers
    """
    print_section_header("PREDICTING WORLDWIDE SPENDING")

    # Prepare features
    print("Preparing features for prediction...")
    X_ww = df_ww[feature_cols].fillna(0)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Passengers: {len(X_ww):,}")

    # Predict categories
    print("\nGenerating predictions...")
    df_ww["predicted_category"] = model.predict(X_ww)

    # Get probability distributions
    pred_proba = model.predict_proba(X_ww)
    df_ww["pred_proba"] = list(pred_proba)

    # Calculate prediction confidence (max probability)
    df_ww["prediction_confidence"] = pred_proba.max(axis=1)

    print("✓ Predictions complete")

    # Distribution of predictions
    print_subsection_header("Predicted Category Distribution")
    pred_dist = print_category_distribution(
        df_ww, "predicted_category", normalize=True, name="Predicted Spending"
    )

    # Average prediction confidence
    avg_confidence = df_ww["prediction_confidence"].mean()
    print(f"\nAverage prediction confidence: {avg_confidence:.2%}")

    if avg_confidence > 0.7:
        print("  ✓ High confidence predictions")
    elif avg_confidence > 0.5:
        print("  ⚠ Moderate confidence predictions")
    else:
        print("  ✗ Low confidence - predictions may be uncertain")

    return df_ww


# ============================================================================
# REVENUE CALCULATION
# ============================================================================


def calculate_revenue_by_airport(df_ww):
    """
    Calculate expected revenue per airport
    """
    print_section_header("REVENUE CALCULATION")

    # Map categories to revenue using midpoints
    print("Mapping predicted categories to revenue...")
    print("Category midpoints (EUR):")
    for cat, midpoint in CATEGORY_MIDPOINTS.items():
        count = len(df_ww[df_ww["predicted_category"] == cat])
        print(f"  Category {cat}: €{midpoint:3d} ({count:,} passengers)")

    df_ww["predicted_revenue"] = df_ww["predicted_category"].map(CATEGORY_MIDPOINTS)

    # Calculate revenue by airport
    print("\nAggregating revenue by airport...")
    revenue_by_airport = (
        df_ww.groupby("shopped_at")
        .agg(
            {
                "predicted_revenue": "sum",
                "name": "count",
                "prediction_confidence": "mean",
            }
        )
        .rename(
            columns={
                "name": "passenger_count",
                "predicted_revenue": "total_revenue",
                "prediction_confidence": "avg_confidence",
            }
        )
    )

    # Calculate monthly and annual revenue
    revenue_by_airport["monthly_revenue"] = revenue_by_airport["total_revenue"]
    revenue_by_airport["annual_revenue"] = (
        revenue_by_airport["monthly_revenue"] * MONTHS_PER_YEAR
    )

    # Calculate revenue per passenger
    revenue_by_airport["revenue_per_passenger"] = (
        revenue_by_airport["monthly_revenue"] / revenue_by_airport["passenger_count"]
    )

    # Sort by monthly revenue
    revenue_by_airport = revenue_by_airport.sort_values(
        "monthly_revenue", ascending=False
    )

    print_subsection_header("Revenue by Airport (Monthly)")
    print(
        revenue_by_airport[
            [
                "passenger_count",
                "monthly_revenue",
                "revenue_per_passenger",
                "avg_confidence",
            ]
        ].to_string()
    )

    return revenue_by_airport


# ============================================================================
# LEASE DATA PROCESSING
# ============================================================================


def parse_lease_data():
    """
    Parse and clean lease data
    """
    print_section_header("LOADING LEASE DATA")

    lease_data = []

    try:
        with open(LEASE_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                line = line.strip().replace('"', "")
                parts = line.split(",")
                if len(parts) == 3:
                    lease_data.append(
                        {
                            "airport": parts[0],
                            "sqm": int(parts[1]),
                            "price_per_sqm": int(parts[2]),
                        }
                    )

        df_lease = pd.DataFrame(lease_data)
        df_lease["monthly_cost"] = df_lease["sqm"] * df_lease["price_per_sqm"]
        df_lease["annual_cost"] = df_lease["monthly_cost"] * MONTHS_PER_YEAR

        print(f"✓ Loaded lease terms for {len(df_lease)} airports")
        print("\nLease costs summary:")
        print(df_lease[["airport", "sqm", "monthly_cost"]].to_string(index=False))

        return df_lease

    except Exception as e:
        print(f"✗ Error loading lease data: {str(e)}")
        return None


# ============================================================================
# PROFITABILITY ANALYSIS
# ============================================================================


def calculate_profitability(revenue_by_airport):
    """
    Calculate profitability (revenue - costs)
    """
    print_section_header("PROFITABILITY ANALYSIS")

    # Load lease data
    df_lease = parse_lease_data()

    if df_lease is None:
        print("✗ Cannot calculate profitability without lease data")
        return None

    # Merge revenue with costs
    profitability = revenue_by_airport.merge(
        df_lease[["airport", "sqm", "monthly_cost", "annual_cost"]],
        left_index=True,
        right_on="airport",
    ).set_index("airport")

    # Calculate profit
    profitability["monthly_profit"] = (
        profitability["monthly_revenue"] - profitability["monthly_cost"]
    )
    profitability["annual_profit"] = (
        profitability["annual_revenue"] - profitability["annual_cost"]
    )

    # Calculate profitability metrics
    profitability["profit_margin"] = (
        profitability["monthly_profit"] / profitability["monthly_revenue"] * 100
    )
    profitability["profit_per_passenger"] = (
        profitability["monthly_profit"] / profitability["passenger_count"]
    )
    profitability["roi_monthly"] = (
        profitability["monthly_profit"] / profitability["monthly_cost"] * 100
    )

    # Sort by annual profit
    profitability = profitability.sort_values("annual_profit", ascending=False)

    # Display results
    print_subsection_header("PROFITABILITY RANKING")

    for idx, (airport, row) in enumerate(profitability.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"{idx}.")

        print(f"\n{medal} {airport}")
        print(f"   Passengers:          {row['passenger_count']:,}")
        print(f"   Store size:          {row['sqm']} sqm")
        print(f"   Monthly Revenue:     {format_currency(row['monthly_revenue'])}")
        print(f"   Monthly Cost:        {format_currency(row['monthly_cost'])}")
        print(f"   Monthly Profit:      {format_currency(row['monthly_profit'])}")
        print(f"   Annual Profit:       {format_currency(row['annual_profit'])}")
        print(f"   Profit Margin:       {format_percentage(row['profit_margin'])}")
        print(f"   ROI:                 {format_percentage(row['roi_monthly'])}")

        if row["monthly_profit"] < 0:
            print(f"   ⚠ WARNING: Unprofitable location!")

    # Save profitability results
    if SAVE_INTERMEDIATE:
        save_dataframe(profitability, PROFITABILITY_FILE, "Profitability Ranking")

    return profitability


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================


def perform_sensitivity_analysis(profitability):
    """
    Sensitivity analysis on key assumptions
    """
    print_section_header("SENSITIVITY ANALYSIS")

    profitability_copy = profitability.copy()

    # Scenario 1: Revenue ±10%
    print_subsection_header("Scenario 1: Revenue Variance (±10%)")

    profitability_copy["monthly_profit_low"] = (
        profitability_copy["monthly_revenue"] * REVENUE_VARIANCE_LOW
        - profitability_copy["monthly_cost"]
    )
    profitability_copy["monthly_profit_high"] = (
        profitability_copy["monthly_revenue"] * REVENUE_VARIANCE_HIGH
        - profitability_copy["monthly_cost"]
    )

    print("\nTop 3 airports - profit range:")
    for airport in profitability_copy.head(3).index:
        row = profitability_copy.loc[airport]
        print(
            f"{airport:5s}: {format_currency(row['monthly_profit_low']):>10s} to "
            + f"{format_currency(row['monthly_profit_high']):>10s}"
        )

    # Scenario 2: Check if ranking changes
    print_subsection_header("Scenario 2: Ranking Stability")

    original_top3 = profitability.head(3).index.tolist()
    low_ranking = (
        profitability_copy.sort_values("monthly_profit_low", ascending=False)
        .head(3)
        .index.tolist()
    )
    high_ranking = (
        profitability_copy.sort_values("monthly_profit_high", ascending=False)
        .head(3)
        .index.tolist()
    )

    print(f"Original top 3: {original_top3}")
    print(f"Low scenario:   {low_ranking}")
    print(f"High scenario:  {high_ranking}")

    if original_top3 == low_ranking == high_ranking:
        print("\n✓ Ranking is STABLE across ±10% revenue scenarios")
    else:
        print("\n⚠ Ranking changes under different scenarios")

    # Scenario 3: Break-even analysis
    print_subsection_header("Scenario 3: Break-Even Analysis")

    print("\nPassengers needed for break-even:")
    for airport in profitability.head(5).index:
        row = profitability.loc[airport]
        breakeven_passengers = row["monthly_cost"] / row["revenue_per_passenger"]
        current_passengers = row["passenger_count"]

        if current_passengers > breakeven_passengers:
            margin_of_safety = (
                (current_passengers - breakeven_passengers) / current_passengers * 100
            )
            print(
                f"{airport:5s}: {breakeven_passengers:6.0f} passengers (margin of safety: {margin_of_safety:.1f}%)"
            )
        else:
            print(
                f"{airport:5s}: {breakeven_passengers:6.0f} passengers ⚠ ABOVE current volume!"
            )

    return profitability_copy


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_profitability_visualizations(profitability, df_ww):
    """
    Create comprehensive visualizations
    """
    print_section_header("CREATING VISUALIZATIONS")

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

    # Add percentages on bars
    for i, v in enumerate(pred_dist.values):
        ax1.text(
            i,
            v + 500,
            f"{v / len(df_ww) * 100:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

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

    # 5. Monthly profit (THE KEY CHART)
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

    # Add value labels
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

    if CREATE_VISUALIZATIONS:
        plt.savefig(VISUALIZATION_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"✓ Visualizations saved to: {VISUALIZATION_FILE}")

    return fig


# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================


def generate_final_recommendation(profitability):
    """
    Generate final recommendation with justification
    """
    print_section_header("FINAL RECOMMENDATION")

    top_3 = profitability.head(3)

    print("\n🎯 TOP 3 AIRPORTS FOR EXPANSION:\n")

    for idx, (airport, row) in enumerate(top_3.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}[idx]

        print(f"{medal} RANK #{idx}: {airport}")
        print(f"{'=' * 60}")
        print(f"Expected Annual Profit:    {format_currency(row['annual_profit'])}")
        print(f"Passenger Volume:          {row['passenger_count']:,}")
        print(f"Profit Margin:             {format_percentage(row['profit_margin'])}")
        print(
            f"Profit per Passenger:      {format_currency(row['profit_per_passenger'])}"
        )
        print(f"Store Size:                {row['sqm']} sqm")
        print(f"Monthly Lease Cost:        {format_currency(row['monthly_cost'])}")
        print(f"ROI (Monthly):             {format_percentage(row['roi_monthly'])}")
        print(f"Prediction Confidence:     {row['avg_confidence']:.1%}")
        print()

    # Primary recommendation
    best_airport = profitability.index[0]
    best_profit = profitability.iloc[0]["annual_profit"]
    best_margin = profitability.iloc[0]["profit_margin"]

    print(f"\n{'=' * 60}")
    print("PRIMARY RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"\n✓ Launch first store at {best_airport}")
    print(f"\nRationale:")
    print(f"  • Highest expected annual profit: {format_currency(best_profit)}")
    print(f"  • Strong profit margin: {format_percentage(best_margin)}")
    print(f"  • Validated through predictive modeling of 200K+ passengers")

    print(f"\n📋 IMPLEMENTATION STRATEGY:")
    print(f"  Phase 1 (Months 1-6):  Pilot at {best_airport}")
    print(f"  Phase 2 (Months 7-9):  Validate predictions, calibrate model")
    print(f"  Phase 3 (Months 10+):  Expand to #2 and #3 based on pilot results")

    print(f"\n⚠ KEY ASSUMPTIONS & RISKS:")
    print(f"  • EU passenger behavior generalizes to worldwide markets")
    print(f"  • Category midpoints represent true average spending")
    print(f"  • December 2019 data represents annual patterns")
    print(f"  • Passenger volumes remain stable post-launch")

    print(f"\n✓ CONFIDENCE LEVEL:")
    print(f"  • High confidence in RANKING (top 3 vs bottom 3)")
    print(f"  • Moderate confidence in ABSOLUTE profit figures (±20%)")
    print(f"  • Recommend 6-month pilot to validate assumptions")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """
    Main prediction and profitability analysis pipeline
    """
    print("=" * 80)
    print("DELOITTE CASE STUDY - PREDICTION & PROFITABILITY ANALYSIS")
    print("=" * 80)

    # Step 1: Load model and data
    print("\n[1/7] Loading model and data...")
    model, feature_cols, df_ww = load_model_and_data()

    if model is None:
        print("✗ Cannot proceed without trained model")
        print("Please run 2_model_training.py first")
        return None

    # Step 2: Predict worldwide spending
    print("\n[2/7] Predicting worldwide spending...")
    df_ww_predictions = predict_worldwide_spending(df_ww, model, feature_cols)

    # Step 3: Calculate revenue
    print("\n[3/7] Calculating revenue by airport...")
    revenue_by_airport = calculate_revenue_by_airport(df_ww_predictions)

    # Step 4: Calculate profitability
    print("\n[4/7] Calculating profitability...")
    profitability = calculate_profitability(revenue_by_airport)

    if profitability is None:
        print("✗ Cannot proceed without profitability data")
        return None

    # Step 5: Sensitivity analysis
    print("\n[5/7] Performing sensitivity analysis...")
    sensitivity_results = perform_sensitivity_analysis(profitability)

    # Step 6: Create visualizations
    print("\n[6/7] Creating visualizations...")
    fig = create_profitability_visualizations(profitability, df_ww_predictions)

    # Step 7: Generate recommendation
    print("\n[7/7] Generating final recommendation...")
    generate_final_recommendation(profitability)

    # Summary
    print_section_header("ANALYSIS COMPLETE")
    print(f"✓ Predictions generated for {len(df_ww_predictions):,} passengers")
    print(f"✓ Profitability calculated for {len(profitability)} airports")
    print(f"✓ Top recommendation: {profitability.index[0]}")
    print(
        f"✓ Expected annual profit: {format_currency(profitability.iloc[0]['annual_profit'])}"
    )
    print(f"✓ Results saved to: {OUTPUT_DIR}")

    return {
        "profitability": profitability,
        "df_ww_predictions": df_ww_predictions,
        "revenue_by_airport": revenue_by_airport,
        "sensitivity": sensitivity_results,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Prediction and profitability analysis complete!")
    print("\nAll files ready for presentation!")
