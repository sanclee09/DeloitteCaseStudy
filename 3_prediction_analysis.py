import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore")

from config import *
from utils import *
from realistic_profit_calculator import (
    calculate_profits_for_all_airports,
    print_profit_breakdown,
    COUNTRY_DATA,
)


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

    # Create revenue column
    df_ww["predicted_revenue"] = df_ww["predicted_category"].map(CATEGORY_MIDPOINTS)

    # Aggregate revenue by airport
    print("\nAggregating revenue by airport...")
    print("Note: Data represents full year 2019 passenger surveys")

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
                "predicted_revenue": "annual_revenue",
                "prediction_confidence": "avg_confidence",
            }
        )
    )

    # Calculate monthly revenue
    revenue_by_airport["monthly_revenue"] = (
        revenue_by_airport["annual_revenue"] / MONTHS_PER_YEAR
    )

    # Calculate revenue per passenger
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
# SIMPLE PROFITABILITY CALCULATION (BASELINE)
# ============================================================================


def calculate_simple_profitability(revenue_by_airport):
    """
    Calculate simple profitability as baseline: Revenue - Leasing Costs
    This is the original model for comparison purposes.
    """
    print_section_header("SIMPLE PROFITABILITY CALCULATION (BASELINE)")
    print("(Revenue - Leasing Costs)")

    # Load lease data
    df_lease = parse_lease_data()
    if df_lease is None:
        print("✗ Cannot calculate profitability without lease data")
        return None

    # Merge revenue with lease costs
    profitability_simple = revenue_by_airport.merge(
        df_lease[["airport", "sqm", "monthly_cost", "annual_cost"]],
        left_index=True,
        right_on="airport",
    ).set_index("airport")

    # Simple calculation: Revenue - Rent
    profitability_simple["monthly_profit"] = (
        profitability_simple["monthly_revenue"] - profitability_simple["monthly_cost"]
    )
    profitability_simple["annual_profit"] = (
        profitability_simple["annual_revenue"] - profitability_simple["annual_cost"]
    )

    # Calculate metrics
    profitability_simple["profit_margin"] = (
        profitability_simple["monthly_profit"]
        / profitability_simple["monthly_revenue"]
        * 100
    )

    # Sort by annual profit
    profitability_simple = profitability_simple.sort_values(
        "annual_profit", ascending=False
    )

    # Display results
    print_subsection_header("SIMPLE PROFITABILITY RANKING (for comparison)")
    print("(Based on full year 2019 data)\n")

    print(
        f"{'Rank':<6} {'Airport':<8} {'Annual Revenue':>15} {'Annual Rent':>15} "
        f"{'Annual Profit':>15} {'Margin':>10}"
    )
    print("─" * 80)

    for idx, (airport, row) in enumerate(profitability_simple.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"{idx}.")

        print(
            f"{medal:<6} {airport:<8} "
            f"{format_currency(row['annual_revenue']):>15} "
            f"{format_currency(row['annual_cost']):>15} "
            f"{format_currency(row['annual_profit']):>15} "
            f"{format_percentage(row['profit_margin']):>10}"
        )

    print("\n⚠️  Note: This simple model ignores staff costs, import duties, and taxes!")
    print("    See realistic profitability calculation below for accurate analysis.\n")

    return profitability_simple


def compare_profit_models(profitability_simple, profitability_realistic):
    """Compare simple vs realistic profit models"""
    print_section_header("PROFIT MODEL COMPARISON")

    print("\n📊 SIMPLE vs REALISTIC MODEL - TOP 5 AIRPORTS\n")

    print(
        f"{'Rank':<6} {'Airport':<8} {'Simple Profit':>15} {'Realistic Profit':>15} {'Difference':>15} {'% Change':>10}"
    )
    print("─" * 90)

    # Get top 5 from realistic model (our recommended ranking)
    top5_realistic = profitability_realistic.head(5).index

    for idx, airport in enumerate(top5_realistic, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}[idx]

        simple_profit = profitability_simple.loc[airport, "annual_profit"]
        realistic_profit = profitability_realistic.loc[airport, "net_profit"]
        difference = realistic_profit - simple_profit
        pct_change = (difference / simple_profit * 100) if simple_profit != 0 else 0

        print(
            f"{medal:<6} {airport:<8} "
            f"{format_currency(simple_profit):>15} "
            f"{format_currency(realistic_profit):>15} "
            f"{format_currency(difference):>15} "
            f"{pct_change:>9.1f}%"
        )

    print("\n" + "=" * 90)
    print("KEY INSIGHTS FROM MODEL COMPARISON")
    print("=" * 90)

    print("\n1. PROFIT MAGNITUDE:")
    avg_simple = profitability_simple["annual_profit"].mean()
    avg_realistic = profitability_realistic["net_profit"].mean()
    print(f"   Average Simple Profit:    {format_currency(avg_simple)}")
    print(f"   Average Realistic Profit: {format_currency(avg_realistic)}")
    print(
        f"   Reduction:                {format_percentage((avg_realistic - avg_simple) / avg_simple * 100)}"
    )

    print("\n2. RANKING CHANGES:")
    # Check if top 3 changed
    top3_simple = set(profitability_simple.head(3).index)
    top3_realistic = set(profitability_realistic.head(3).index)

    if top3_simple == top3_realistic:
        print("   ✓ Top 3 airports remain the same")
    else:
        moved_up = top3_realistic - top3_simple
        moved_down = top3_simple - top3_realistic
        if moved_up:
            print(f"   ↑ Moved INTO top 3: {', '.join(moved_up)}")
        if moved_down:
            print(f"   ↓ Moved OUT OF top 3: {', '.join(moved_down)}")

    print("\n3. WHY REALISTIC MODEL IS BETTER:")
    print("   ✗ Simple model assumes 90%+ profit margins (unrealistic!)")
    print("   ✓ Realistic model: 20-30% margins (industry standard)")
    print("   ✓ Accounts for country-specific staff costs")
    print("   ✓ Includes import duties (critical for global expansion)")
    print("   ✓ Considers corporate tax differences")
    print("   ✓ Provides true profitability picture for decision-making")

    print("\n4. BUSINESS IMPACT:")
    # Find biggest ranking change
    simple_ranks = {
        airport: idx + 1 for idx, airport in enumerate(profitability_simple.index)
    }
    realistic_ranks = {
        airport: idx + 1 for idx, airport in enumerate(profitability_realistic.index)
    }

    biggest_jump = None
    biggest_jump_value = 0
    biggest_drop = None
    biggest_drop_value = 0

    for airport in profitability_realistic.index:
        change = simple_ranks[airport] - realistic_ranks[airport]  # positive = moved up
        if change > biggest_jump_value:
            biggest_jump_value = change
            biggest_jump = airport
        if change < biggest_drop_value:
            biggest_drop_value = change
            biggest_drop = airport

    if biggest_jump:
        print(
            f"   ⬆️  Biggest Improvement: {biggest_jump} (moved up {biggest_jump_value} positions)"
        )
        reason_jump = profitability_realistic.loc[biggest_jump]
        if reason_jump["has_eu_fta"]:
            print(f"      Reason: EU FTA reduces import costs significantly")
        else:
            print(
                f"      Reason: Low tax rate ({reason_jump['effective_tax_rate']:.0f}%) or low wages"
            )

    if biggest_drop:
        print(
            f"   ⬇️  Biggest Decline: {biggest_drop} (dropped {abs(biggest_drop_value)} positions)"
        )
        reason_drop = profitability_realistic.loc[biggest_drop]
        print(
            f"      Reason: High wages (€{reason_drop['staff_cost']:,.0f}) and/or no FTA"
        )

    print("\n" + "=" * 90 + "\n")


# ============================================================================
# REALISTIC PROFITABILITY CALCULATION
# ============================================================================


def calculate_realistic_profitability(revenue_by_airport):
    """
    Calculate realistic profitability with country-specific factors:
    - Staff costs (local wage levels)
    - Import duties (FTA agreements)
    - Corporate taxes
    - Operating expenses
    """
    print_section_header("REALISTIC PROFITABILITY CALCULATION")
    print("Considering: Staff Costs, Import Duties, Taxes, Operating Expenses")

    # Load lease data
    df_lease = parse_lease_data()
    if df_lease is None:
        print("✗ Cannot calculate profitability without lease data")
        return None

    # Set airport as index for lease data
    df_lease.set_index("airport", inplace=True)

    # Calculate realistic profits for all airports
    print("\nCalculating country-specific profitability...")
    profitability = calculate_profits_for_all_airports(revenue_by_airport, df_lease)

    # Display top 5 in detail
    print_section_header("TOP 5 AIRPORTS - DETAILED BREAKDOWN")

    for idx, (airport, row) in enumerate(profitability.head(5).iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}[idx]

        print(f"\n{medal} RANK #{idx}: {airport} ({row['country']})")
        print(f"{'─' * 70}")
        print(f"  Revenue:               €{row['annual_revenue']:>10,.0f}")
        print(
            f"  COGS (with duties):    €{row['total_cogs']:>10,.0f} ({row['cogs_percent']:.1f}%)"
        )
        if row["import_duties"] > 0:
            print(f"    └─ Import Duties:    €{row['import_duties']:>10,.0f}")
            if row["has_eu_fta"]:
                print(f"       (80% reduced via EU FTA)")
        print(f"  Gross Profit:          €{row['gross_profit']:>10,.0f}")
        print(f"  Operating Expenses:    €{row['total_opex']:>10,.0f}")
        print(f"    ├─ Staff:            €{row['staff_cost']:>10,.0f}")
        print(f"    ├─ Lease:            €{row['lease_cost']:>10,.0f}")
        print(f"    └─ Other:            €{row['other_opex']:>10,.0f}")
        print(f"  EBIT:                  €{row['ebit']:>10,.0f}")
        print(
            f"  Corporate Tax ({row['effective_tax_rate']:.0f}%): €{row['corporate_tax']:>10,.0f}"
        )
        print(f"  {'═' * 70}")
        print(
            f"  NET PROFIT:            €{row['net_profit']:>10,.0f} ({row['profit_margin_percent']:.1f}%)"
        )

    # Summary comparison table
    print("\n" + "=" * 80)
    print("PROFITABILITY RANKING - SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Rank':<6} {'Airport':<8} {'Country':<12} {'Revenue':>12} {'Net Profit':>12} {'Margin':>8}"
    )
    print("─" * 80)

    for idx, (airport, row) in enumerate(profitability.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"{idx}.")

        print(
            f"{medal:<6} {airport:<8} {row['country']:<12} "
            f"{format_currency(row['annual_revenue']):>12} "
            f"{format_currency(row['net_profit']):>12} "
            f"{format_percentage(row['profit_margin_percent']):>8}"
        )

    # Save results
    if SAVE_INTERMEDIATE:
        # Save full breakdown
        profitability.to_csv(PROFITABILITY_FILE)
        print(f"\n✓ Profitability data saved to: {PROFITABILITY_FILE}")

    return profitability


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualizations(profitability, df_ww):
    """Create comprehensive profitability visualizations"""
    print_section_header("CREATING VISUALIZATIONS")

    fig = plt.figure(figsize=(18, 12))

    # 1. Predicted spending distribution
    ax1 = plt.subplot(3, 2, 1)
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

    # 2. Net profit by airport
    ax2 = plt.subplot(3, 2, 2)
    net_profits = profitability["net_profit"].sort_values(ascending=True)
    colors_profit = [
        COLORS["profit_positive"] if p > 0 else COLORS["profit_negative"]
        for p in net_profits
    ]
    net_profits.plot(kind="barh", ax=ax2, color=colors_profit)
    ax2.set_title("Net Profit by Airport ⭐", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Net Profit (€)")
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=2)

    for i, (airport, profit) in enumerate(net_profits.items()):
        ax2.text(
            profit + (30000 if profit > 0 else -30000),
            i,
            f"€{profit:,.0f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # 3. Profit margin comparison
    ax3 = plt.subplot(3, 2, 3)
    profit_margins = profitability["profit_margin_percent"].sort_values(ascending=True)
    colors_margin = [
        COLORS["profit_positive"] if m > 0 else COLORS["profit_negative"]
        for m in profit_margins
    ]
    profit_margins.plot(kind="barh", ax=ax3, color=colors_margin)
    ax3.set_title("Profit Margin by Airport", fontweight="bold")
    ax3.set_xlabel("Profit Margin (%)")
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=2)

    # 4. Revenue vs Profit scatter
    ax4 = plt.subplot(3, 2, 4)
    scatter = ax4.scatter(
        profitability["annual_revenue"] / 1000,
        profitability["net_profit"] / 1000,
        s=profitability["passenger_count"] / 100,
        c=profitability["profit_margin_percent"],
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )
    ax4.set_title("Revenue vs Profit (bubble = passenger volume)", fontweight="bold")
    ax4.set_xlabel("Annual Revenue (€ thousands)")
    ax4.set_ylabel("Net Profit (€ thousands)")
    ax4.grid(alpha=0.3)

    # Add airport labels
    for airport, row in profitability.iterrows():
        ax4.annotate(
            airport,
            (row["annual_revenue"] / 1000, row["net_profit"] / 1000),
            fontsize=8,
            ha="center",
        )

    plt.colorbar(scatter, ax=ax4, label="Profit Margin (%)")

    # 5. Cost breakdown for top 3
    ax5 = plt.subplot(3, 2, 5)
    top3 = profitability.head(3)

    cost_categories = [
        "total_cogs",
        "staff_cost",
        "lease_cost",
        "other_opex",
        "corporate_tax",
    ]
    cost_labels = ["COGS+Duties", "Staff", "Lease", "Other OpEx", "Tax"]

    x_pos = np.arange(len(top3))
    width = 0.15

    for i, (cost_cat, label) in enumerate(zip(cost_categories, cost_labels)):
        values = top3[cost_cat]
        ax5.bar(x_pos + i * width, values, width, label=label, alpha=0.8)

    ax5.set_title("Cost Breakdown - Top 3 Airports", fontweight="bold")
    ax5.set_xlabel("Airport")
    ax5.set_ylabel("Cost (€)")
    ax5.set_xticks(x_pos + width * 2)
    ax5.set_xticklabels(top3.index)
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(axis="y", alpha=0.3)

    # 6. Impact of FTA and taxes
    ax6 = plt.subplot(3, 2, 6)

    # Color by FTA status
    colors_fta = [
        "#2ecc71" if has_fta else "#e74c3c" for has_fta in profitability["has_eu_fta"]
    ]

    bars = ax6.barh(
        range(len(profitability)),
        profitability["effective_tax_rate"],
        color=colors_fta,
        alpha=0.7,
    )

    ax6.set_yticks(range(len(profitability)))
    ax6.set_yticklabels(profitability.index)
    ax6.set_title("Tax Rates & FTA Status", fontweight="bold")
    ax6.set_xlabel("Corporate Tax Rate (%)")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Has EU FTA"),
        Patch(facecolor="#e74c3c", label="No EU FTA"),
    ]
    ax6.legend(handles=legend_elements, loc="lower right")

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
    """Generate final recommendation with realistic profit analysis"""
    print_section_header("FINAL RECOMMENDATION")

    top_5 = profitability.head(5)

    print("\n🎯 TOP 5 AIRPORTS FOR EXPANSION:\n")

    for idx, (airport, row) in enumerate(top_5.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}[idx]

        print(f"{medal} RANK #{idx}: {airport} ({row['country']})")
        print(f"{'=' * 70}")
        print(f"Net Profit:              {format_currency(row['net_profit'])} annually")
        print(
            f"Profit Margin:           {format_percentage(row['profit_margin_percent'])}"
        )
        print(f"Revenue:                 {format_currency(row['annual_revenue'])}")
        print(f"Passenger Volume:        {row['passenger_count']:>10,}")

        # Key factors
        fta_status = "✓ Has EU FTA" if row["has_eu_fta"] else "✗ No EU FTA"
        print(f"FTA Status:              {fta_status}")
        print(f"Import Duty Rate:        {row['import_duty_rate_percent']:.1f}%")
        print(f"Corporate Tax Rate:      {row['effective_tax_rate']:.1f}%")
        print()

    # Primary recommendation
    best_airport = profitability.index[0]
    best = profitability.iloc[0]

    print(f"\n{'=' * 70}")
    print("PRIMARY RECOMMENDATION")
    print(f"{'=' * 70}")
    print(f"\n✓ Launch first store at {best_airport} ({best['country']})")

    print(f"\n💰 FINANCIAL HIGHLIGHTS:")
    print(f"  Expected Net Profit:     {format_currency(best['net_profit'])} annually")
    print(
        f"  Profit Margin:           {format_percentage(best['profit_margin_percent'])}"
    )
    print(f"  Revenue:                 {format_currency(best['annual_revenue'])}")
    print(f"  EBIT:                    {format_currency(best['ebit'])}")

    print(f"\n📊 COMPETITIVE ADVANTAGES:")
    advantages = []

    if best["has_eu_fta"]:
        advantages.append(f"  ✓ EU Free Trade Agreement (80% import duty reduction)")

    if best["effective_tax_rate"] < 25:
        advantages.append(f"  ✓ Favorable tax rate ({best['effective_tax_rate']:.0f}%)")

    if best["profit_margin_percent"] > profitability["profit_margin_percent"].median():
        advantages.append(f"  ✓ Above-average profit margin")

    if best["passenger_count"] > profitability["passenger_count"].median():
        advantages.append(f"  ✓ High passenger volume ({best['passenger_count']:,})")

    for adv in advantages:
        print(adv)

    print(f"\n📋 PHASED EXPANSION STRATEGY:")
    for idx, (airport, row) in enumerate(profitability.head(3).iterrows(), 1):
        profit = row["net_profit"]
        margin = row["profit_margin_percent"]
        print(
            f"  Phase {idx}: {airport} ({row['country']}) - "
            f"{format_currency(profit)}/year ({margin:.1f}% margin)"
        )

    print(f"\n⚠️  KEY CONSIDERATIONS:")
    print(f"  • Data represents full year 2019 passenger surveys")
    print(f"  • Realistic profit model includes:")
    print(f"    - Country-specific staff costs")
    print(f"    - Import duties (reduced 80% with EU FTA)")
    print(f"    - Corporate taxes")
    print(f"    - Lease and operational expenses")
    print(f"  • Category midpoints represent average spending per category")
    print(f"  • EU spending patterns generalized to worldwide markets")
    print(f"  • Does not include: CAPEX, inventory costs, currency fluctuations")

    print(f"\n🔍 SENSITIVITY FACTORS:")
    print(f"  High Impact: Local wage levels, import duties, tax rates")
    print(f"  Medium Impact: Passenger volume changes, category distribution shifts")
    print(f"  Low Impact: Lease cost variations, operational efficiency")

    print(f"\n✓ Model Performance:")
    avg_confidence = profitability.head(5)["avg_confidence"].mean()
    print(f"  • Average prediction confidence: {avg_confidence:.1%}")
    print(f"  • Test accuracy: 92% (from model training)")
    print(f"  • F1 Score: 0.92 (weighted)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main prediction and realistic profitability analysis pipeline"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - PREDICTION & REALISTIC PROFITABILITY ANALYSIS")
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

    # Step 4a: Calculate SIMPLE profitability (baseline)
    print("\n[4a/5] Calculating simple profitability (baseline)...")
    profitability_simple = calculate_simple_profitability(revenue_by_airport)

    # Step 4b: Calculate REALISTIC profitability
    print("\n[4b/5] Calculating realistic profitability...")
    profitability = calculate_realistic_profitability(revenue_by_airport)

    if profitability is None:
        print("✗ Cannot proceed without profitability data")
        return None

    # Step 4c: Compare both models
    if profitability_simple is not None:
        print("\n[4c/5] Comparing simple vs realistic models...")
        compare_profit_models(profitability_simple, profitability)

    # Step 5: Create visualizations and recommendation
    print("\n[5/5] Generating outputs...")
    create_visualizations(profitability, df_ww_predictions)
    generate_final_recommendation(profitability)

    # Summary
    print_section_header("ANALYSIS COMPLETE")
    print(f"✓ Predictions generated for {len(df_ww_predictions):,} passengers")
    print(f"✓ Simple profitability calculated (baseline)")
    print(f"✓ Realistic profitability calculated for {len(profitability)} airports")
    print(f"✓ Top recommendation: {profitability.index[0]}")
    print(
        f"✓ Expected net profit (realistic): {format_currency(profitability.iloc[0]['net_profit'])}"
    )
    print(
        f"✓ Profit margin: {format_percentage(profitability.iloc[0]['profit_margin_percent'])}"
    )
    print(f"✓ Results saved to: {OUTPUT_DIR}")

    return {
        "profitability_realistic": profitability,
        "profitability_simple": profitability_simple,
        "df_ww_predictions": df_ww_predictions,
        "revenue_by_airport": revenue_by_airport,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Realistic profitability analysis complete!")
