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
    """Load trained model and preprocessed data"""
    print_section_header("LOADING MODEL AND DATA")

    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded: {MODEL_FILE}")
        print(f"  Trained on: {model_data['trained_date']}")
        print(f"  Test accuracy: {model_data['metrics']['test_accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None, None, None, None, None

    df_ww = load_csv_with_info(WW_CLEAN_FILE, "WW Passengers (Clean)")

    return (
        model_data["model"],
        model_data["feature_cols"],
        model_data.get("feature_selector"),
        model_data.get("candidate_features", model_data["feature_cols"]),
        df_ww,
    )


# ============================================================================
# PREDICTION
# ============================================================================


def predict_worldwide_spending(
    df_ww, model, feature_cols, selector=None, candidate_features=None
):
    """Predict spending categories for worldwide passengers"""
    print_section_header("PREDICTING WORLDWIDE SPENDING")

    print("Preparing features for prediction...")

    if candidate_features is None:
        candidate_features = feature_cols

    # Verify all features exist
    missing_features = [f for f in candidate_features if f not in df_ww.columns]
    if missing_features:
        print(f"\n⚠️  Warning: Missing {len(missing_features)} features in WW dataset:")
        for f in missing_features[:10]:
            print(f"    - {f}")
        if len(missing_features) > 10:
            print(f"    ... and {len(missing_features) - 10} more")
        print("\n✗ Cannot proceed without all required features.")
        return None

    # Apply feature selection
    if selector:
        X_ww = df_ww[candidate_features].values
        X_ww_selected, _ = selector.apply_feature_selection(X_ww, candidate_features)
    else:
        X_ww_selected = df_ww[feature_cols].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Passengers: {len(X_ww_selected):,}")

    # Predict
    print("\nGenerating predictions...")
    df_ww["predicted_category"] = model.predict(X_ww_selected)

    pred_proba = model.predict_proba(X_ww_selected)
    df_ww["pred_proba"] = list(pred_proba)
    df_ww["prediction_confidence"] = pred_proba.max(axis=1)

    print("✓ Predictions complete")

    # Distribution
    print_subsection_header("Predicted Category Distribution")
    pred_dist = print_category_distribution(
        df_ww, "predicted_category", normalize=True, name="Predicted Spending"
    )

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
    """Calculate expected revenue per airport"""
    print_section_header("REVENUE CALCULATION")

    # Map categories to revenue using midpoints
    print("Mapping predicted categories to revenue...")
    print("Category midpoints (EUR):")
    for cat, midpoint in CATEGORY_MIDPOINTS.items():
        count = len(df_ww[df_ww["predicted_category"] == cat])
        print(f"  Category {cat}: €{midpoint:3d} ({count:,} passengers)")

    # Create revenue column : category 2 -> €100, etc.
    df_ww["predicted_revenue"] = df_ww["predicted_category"].map(CATEGORY_MIDPOINTS)

    # Aggregate revenue by airport
    print("\nAggregating revenue by airport...")
    print("Note: Data represents full year 2019 passenger surveys")

    revenue_by_airport = (
        # Group all passengers by their shopping airport (e.g., "HND", "DXB", "JFK")
        df_ww.groupby("shopped_at")
        .agg(
            {
                "predicted_revenue": "sum",  # Sum all revenue for passengers at this airport
                "name": "count",
                "prediction_confidence": "mean",
            }
        )
        .rename(
            columns={
                "name": "passenger_count",
                "predicted_revenue": "annual_revenue",
                "prediction_confidence": "avg_confidence",
            }
        )
    )

    # Calculate monthly revenue (annual divided by 12)
    revenue_by_airport["monthly_revenue"] = (
        revenue_by_airport["annual_revenue"] / MONTHS_PER_YEAR
    )

    # Calculate revenue per passenger (based on annual)
    revenue_by_airport["revenue_per_passenger"] = (
        revenue_by_airport["annual_revenue"] / revenue_by_airport["passenger_count"]
    )

    # Sort by annual revenue
    revenue_by_airport = revenue_by_airport.sort_values(
        "annual_revenue", ascending=False
    )

    print_subsection_header("Revenue by Airport")
    print("(Data represents full year 2019)\n")
    print(
        revenue_by_airport[
            [
                "passenger_count",
                "annual_revenue",
                "monthly_revenue",
                "revenue_per_passenger",
                "avg_confidence",
            ]
        ].to_string()
    )

    return revenue_by_airport


# ============================================================================
# LEASE DATA PARSING
# ============================================================================


def parse_lease_data():
    """Parse and clean lease data"""
    print_subsection_header("Loading Lease Data")

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

        return df_lease

    except Exception as e:
        print(f"✗ Error loading lease data: {str(e)}")
        return None


# ============================================================================
# SIMPLE PROFITABILITY CALCULATION
# ============================================================================


def calculate_simple_profitability(revenue_by_airport):
    """
    Calculate profitability as instructed: Revenue - Leasing Costs

    This follows the case study instructions:
    "ranked by expected profit (i.e. revenue minus leasing costs)"
    """
    print_section_header("PROFITABILITY CALCULATION")
    print("(Revenue - Leasing Costs)")

    # Load lease data
    df_lease = parse_lease_data()
    if df_lease is None:
        print("✗ Cannot calculate profitability without lease data")
        return None

    # Merge revenue with lease costs
    profitability = revenue_by_airport.merge(
        df_lease[["airport", "sqm", "monthly_cost", "annual_cost"]],
        left_index=True,
        right_on="airport",
    ).set_index("airport")

    # Simple calculation: Revenue - Rent
    profitability["monthly_profit"] = (
        profitability["monthly_revenue"] - profitability["monthly_cost"]
    )
    profitability["annual_profit"] = (
        profitability["annual_revenue"] - profitability["annual_cost"]
    )

    # Calculate metrics
    profitability["profit_margin"] = (
        profitability["monthly_profit"] / profitability["monthly_revenue"] * 100
    )

    profitability["profit_per_passenger"] = (
        profitability["annual_profit"] / profitability["passenger_count"]
    )

    profitability["profit_per_sqm"] = (
        profitability["monthly_profit"] / profitability["sqm"]
    )

    # Sort by annual profit
    profitability = profitability.sort_values("annual_profit", ascending=False)

    # Display results
    print_subsection_header("PROFITABILITY RANKING")
    print("(Based on full year 2019 data)\n")

    print(
        f"{'Rank':<6} {'Airport':<8} {'Annual Revenue':>15} {'Annual Rent':>15} "
        f"{'Annual Profit':>15} {'Margin':>10}"
    )
    print("─" * 80)

    for idx, (airport, row) in enumerate(profitability.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"{idx}.")

        print(
            f"{medal:<6} {airport:<8} "
            f"{format_currency(row['annual_revenue']):>15} "
            f"{format_currency(row['annual_cost']):>15} "
            f"{format_currency(row['annual_profit']):>15} "
            f"{format_percentage(row['profit_margin']):>10}"
        )

    # Detailed breakdown for top 3
    print("\n" + "=" * 80)
    print("TOP 3 AIRPORTS - DETAILED BREAKDOWN")
    print("=" * 80)

    for idx, (airport, row) in enumerate(profitability.head(3).iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}[idx]

        print(f"\n{medal} RANK #{idx}: {airport}")
        print(f"{'─' * 60}")
        print(
            f"  Passenger Volume:        {row['passenger_count']:>10,} passengers/year"
        )
        print(f"  Store Size:              {row['sqm']:>10,.0f} sqm")
        print(
            f"  Revenue per Passenger:   {format_currency(row['revenue_per_passenger']):>10}"
        )
        print(f"  Prediction Confidence:   {row['avg_confidence']:>10.1%}")
        print(f"\n  ANNUAL FINANCIALS:")
        print(
            f"    Revenue:               {format_currency(row['annual_revenue']):>10}"
        )
        print(f"    Leasing Cost:          {format_currency(row['annual_cost']):>10}")
        print(f"    ────────────────────────────────")
        print(f"    Profit:                {format_currency(row['annual_profit']):>10}")
        print(f"\n  METRICS:")
        print(
            f"    Profit Margin:         {format_percentage(row['profit_margin']):>10}"
        )
        print(
            f"    Profit per Passenger:  {format_currency(row['profit_per_passenger']):>10}"
        )
        print(
            f"    Profit per sqm/month:  {format_currency(row['profit_per_sqm']):>10}"
        )

    # Save results
    if SAVE_INTERMEDIATE:
        save_dataframe(profitability, PROFITABILITY_FILE, "Profitability Ranking")

    return profitability


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualizations(profitability, df_ww):
    """Create profitability visualizations"""
    print_section_header("CREATING VISUALIZATIONS")

    fig = plt.figure(figsize=FIGURE_SIZE)

    # 1. Predicted spending distribution
    ax1 = plt.subplot(2, 2, 1)
    pred_dist = df_ww["predicted_category"].value_counts().sort_index()
    colors_spending = COLORS["spending_categories"]
    pred_dist.plot(kind="bar", ax=ax1, color=colors_spending)
    ax1.set_title("Predicted Spending Distribution\n(Worldwide)", fontweight="bold")
    ax1.set_xlabel("Spending Category")
    ax1.set_ylabel("Number of Passengers")
    ax1.tick_params(axis="x", rotation=0)

    for i, v in enumerate(pred_dist.values):
        ax1.text(
            i,
            v + 500,
            f"{v / len(df_ww) * 100:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    # 2. Annual profit by airport
    ax2 = plt.subplot(2, 2, 2)
    annual_profits = profitability["annual_profit"].sort_values(ascending=True)
    colors_profit = [
        COLORS["profit_positive"] if p > 0 else COLORS["profit_negative"]
        for p in annual_profits
    ]
    annual_profits.plot(kind="barh", ax=ax2, color=colors_profit)
    ax2.set_title("Annual Profit by Airport ⭐", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Annual Profit (€)")
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=2)

    for i, (airport, profit) in enumerate(annual_profits.items()):
        ax2.text(
            profit + (50000 if profit > 0 else -50000),
            i,
            f"€{profit:,.0f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # 3. Profit margin
    ax3 = plt.subplot(2, 2, 3)
    profit_margins = profitability["profit_margin"].sort_values(ascending=True)
    colors_margin = [
        COLORS["profit_positive"] if m > 0 else COLORS["profit_negative"]
        for m in profit_margins
    ]
    profit_margins.plot(kind="barh", ax=ax3, color=colors_margin)
    ax3.set_title("Profit Margin by Airport", fontweight="bold")
    ax3.set_xlabel("Profit Margin (%)")
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=2)

    # 4. Profit per passenger
    ax4 = plt.subplot(2, 2, 4)
    profit_per_pax = profitability["profit_per_passenger"].sort_values(ascending=True)
    colors_pax = [
        COLORS["profit_positive"] if p > 0 else COLORS["profit_negative"]
        for p in profit_per_pax
    ]
    profit_per_pax.plot(kind="barh", ax=ax4, color=colors_pax)
    ax4.set_title("Profit per Passenger", fontweight="bold")
    ax4.set_xlabel("Profit per Passenger (€)")
    ax4.axvline(x=0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()

    if CREATE_VISUALIZATIONS:
        plt.savefig(VISUALIZATION_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"✓ Visualizations saved to: {VISUALIZATION_FILE}")
        plt.close()

    return fig


# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================


def generate_final_recommendation(profitability):
    """Generate final recommendation"""
    print_section_header("FINAL RECOMMENDATION")

    top_3 = profitability.head(3)

    print("\n🎯 TOP 3 AIRPORTS FOR EXPANSION:\n")

    for idx, (airport, row) in enumerate(top_3.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}[idx]

        print(f"{medal} RANK #{idx}: {airport}")
        print(f"{'=' * 60}")
        print(f"Expected Annual Profit:    {format_currency(row['annual_profit'])}")
        print(f"Profit Margin:             {format_percentage(row['profit_margin'])}")
        print(f"Passenger Volume:          {row['passenger_count']:>10,}")
        print(
            f"Profit per Passenger:      {format_currency(row['profit_per_passenger'])}"
        )
        print(f"Store Size:                {row['sqm']:.0f} sqm")
        print()

    # Primary recommendation
    best_airport = profitability.index[0]
    best_profit = profitability.iloc[0]["annual_profit"]
    best_margin = profitability.iloc[0]["profit_margin"]
    best_passengers = profitability.iloc[0]["passenger_count"]

    print(f"\n{'=' * 60}")
    print("PRIMARY RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"\n✓ Launch first store at {best_airport}")
    print(f"\nKey Metrics:")
    print(f"  • Expected annual profit: {format_currency(best_profit)}")
    print(f"  • Profit margin: {format_percentage(best_margin)}")
    print(f"  • Passenger volume: {best_passengers:,} per year")
    print(f"  • Calculation: Revenue - Leasing Costs")

    print(f"\n📋 EXPANSION STRATEGY:")
    second_airport = profitability.index[1]
    third_airport = profitability.index[2]

    print(f"  Phase 1: Launch at {best_airport}")
    print(
        f"  Phase 2: Expand to {second_airport} (Profit: {format_currency(profitability.iloc[1]['annual_profit'])})"
    )
    print(
        f"  Phase 3: Expand to {third_airport} (Profit: {format_currency(profitability.iloc[2]['annual_profit'])})"
    )

    print(f"\n⚠️  KEY ASSUMPTIONS:")
    print(f"  • Data represents full year 2019 passenger surveys")
    print(f"  • Category midpoints represent average spending")
    print(f"  • EU spending patterns generalize to worldwide markets")
    print(f"  • Profit = Revenue - Leasing Costs (as instructed)")
    print(f"  • Does not include: COGS, staffing, or operational costs")

    print(f"\n✓ Model Performance:")
    avg_confidence = profitability.head(3)["avg_confidence"].mean()
    print(f"  • Average prediction confidence: {avg_confidence:.1%}")
    print(f"  • Test accuracy: Available in model metrics")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main prediction and profitability analysis pipeline"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - PREDICTION & PROFITABILITY ANALYSIS")
    print("=" * 80)

    # Step 1: Load model and data
    print("\n[1/5] Loading model and data...")
    model, feature_cols, selector, candidate_features, df_ww = load_model_and_data()

    if model is None:
        print("✗ Cannot proceed without trained model")
        return None

    # Step 2: Predict worldwide spending
    print("\n[2/5] Predicting worldwide spending...")
    df_ww_predictions = predict_worldwide_spending(
        df_ww, model, feature_cols, selector, candidate_features
    )

    if df_ww_predictions is None:
        print("✗ Prediction failed - missing features in WW dataset")
        return None

    # Step 3: Calculate revenue
    print("\n[3/5] Calculating revenue by airport...")
    revenue_by_airport = calculate_revenue_by_airport(df_ww_predictions)

    # Step 4: Calculate profitability
    print("\n[4/5] Calculating profitability...")
    profitability = calculate_simple_profitability(revenue_by_airport)

    if profitability is None:
        print("✗ Cannot proceed without profitability data")
        return None

    # Step 5: Create visualizations and recommendation
    print("\n[5/5] Generating outputs...")
    create_visualizations(profitability, df_ww_predictions)
    generate_final_recommendation(profitability)

    # Summary
    print_section_header("ANALYSIS COMPLETE")
    print(f"✓ Predictions generated for {len(df_ww_predictions):,} passengers")
    print(f"✓ Profitability calculated for {len(profitability)} airports")
    print(f"✓ Top recommendation: {profitability.index[0]}")
    print(
        f"✓ Expected annual profit: {format_currency(profitability.iloc[0]['annual_profit'])}"
    )
    print(
        f"✓ Profit margin: {format_percentage(profitability.iloc[0]['profit_margin'])}"
    )
    print(f"✓ Results saved to: {OUTPUT_DIR}")

    return {
        "profitability": profitability,
        "df_ww_predictions": df_ww_predictions,
        "revenue_by_airport": revenue_by_airport,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Analysis complete!")
