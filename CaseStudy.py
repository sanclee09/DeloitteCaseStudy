import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================


def load_data():
    """Load all datasets"""
    df_eu = pd.read_csv(
        r"/Users/SancLee/PycharmProjects/DeloitteCaseStudy/raw/passengersEU.csv"
    )
    df_ww = pd.read_csv(
        r"/Users/SancLee/PycharmProjects/DeloitteCaseStudy/raw/passengersWW.csv"
    )
    df_airports = pd.read_csv(
        r"/Users/SancLee/PycharmProjects/DeloitteCaseStudy/raw/airports.csv"
    )
    df_lease = pd.read_csv(
        r"/Users/SancLee/PycharmProjects/DeloitteCaseStudy/raw/airports_terms_of_lease.csv"
    )

    print(f"EU passengers: {len(df_eu):,} rows")
    print(f"WW passengers: {len(df_ww):,} rows")
    print(f"Airports: {len(df_airports)} rows")
    print(f"Lease terms: {len(df_lease)} rows")

    return df_eu, df_ww, df_airports, df_lease


# ============================================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================================================


def parse_birth_date(date_str):
    """Parse birth date with multiple format support"""
    if pd.isna(date_str):
        return None

    formats = ["%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except:
            continue
    return None


def extract_luggage_weight(luggage_str):
    """Extract weight from luggage string and convert to kg"""
    if pd.isna(luggage_str):
        return 0

    luggage_str = str(luggage_str).lower()

    # Extract number and unit
    match = re.search(r"(\d+\.?\d*)\s*(kg|l|lbs|lb)", luggage_str)
    if match:
        weight = float(match.group(1))
        unit = match.group(2)

        # Convert to kg
        if unit in ["lbs", "lb"]:
            weight = weight * 0.453592  # lbs to kg
        # L is already assumed as kg (1L ≙ 1kg per case description)

        return weight

    return 0


def calculate_age(birth_date, reference_date="2019-12-31"):
    """Calculate age from birth date"""
    if pd.isna(birth_date):
        return None

    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    age = ref_date.year - birth_date.year

    # Adjust if birthday hasn't occurred yet
    if (ref_date.month, ref_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


def clean_and_engineer_features(df):
    """Clean data and engineer features"""
    df = df.copy()

    # 1. Parse birth dates and calculate age
    df["birth_date_parsed"] = df["birth_date"].apply(parse_birth_date)
    df["age"] = df["birth_date_parsed"].apply(calculate_age)

    # 2. Extract luggage weight
    df["luggage_weight_kg"] = df["luggage"].apply(extract_luggage_weight)

    # 3. Create binary features
    df["is_business"] = (df["business_trip"] == "yes").astype(int)
    df["has_family"] = (df["traveled_with_family"] == "yes").astype(int)
    df["has_connection"] = (~df["flight_number_2"].isna()).astype(int)

    # 4. Handle missing values in numeric columns
    df["layover_time"] = df["layover_time"].fillna(0)
    df["age"] = df["age"].fillna(df["age"].median())

    # 5. Create layover time categories
    df["layover_category"] = pd.cut(
        df["layover_time"],
        bins=[-1, 0, 60, 180, 1000],
        labels=["no_layover", "short", "medium", "long"],
    )

    # 6. Encode sex
    df["is_male"] = (df["sex"] == "m").astype(int)

    return df


# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================


def perform_eda(df_eu):
    """Perform exploratory data analysis"""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Spending distribution
    print("\n1. Spending Category Distribution:")
    print(df_eu["amount_spent_cat"].value_counts().sort_index())
    print(f"\nPercentages:")
    print(
        (
            df_eu["amount_spent_cat"].value_counts(normalize=True).sort_index() * 100
        ).round(2)
    )

    # Spending by business trip
    print("\n2. Average Spending by Business Trip:")
    print(df_eu.groupby("is_business")["amount_spent_cat"].mean())

    # Spending by family travel
    print("\n3. Average Spending by Family Travel:")
    print(df_eu.groupby("has_family")["amount_spent_cat"].mean())

    # Spending by connection flight
    print("\n4. Average Spending by Connection Flight:")
    print(df_eu.groupby("has_connection")["amount_spent_cat"].mean())

    # Correlations
    print("\n5. Feature Correlations with Spending:")
    numeric_cols = [
        "age",
        "is_business",
        "has_family",
        "has_connection",
        "luggage_weight_kg",
        "total_flighttime",
        "layover_time",
    ]
    correlations = (
        df_eu[numeric_cols + ["amount_spent_cat"]]
        .corr()["amount_spent_cat"]
        .sort_values(ascending=False)
    )
    print(correlations)

    return correlations


# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================


def train_model(df_eu):
    """Train Random Forest classifier"""
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    # Select features
    feature_cols = [
        "age",
        "is_male",
        "is_business",
        "has_family",
        "has_connection",
        "luggage_weight_kg",
        "total_flighttime",
        "total_traveltime",
        "layover_time",
    ]

    # Prepare data
    X = df_eu[feature_cols].fillna(0)
    y = df_eu["amount_spent_cat"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    # Evaluate
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)

    print(f"\nTraining Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(
        f"Cross-validation F1 (weighted): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
    )

    # Predictions
    y_pred = rf_model.predict(X_test)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(feature_importance)

    return rf_model, feature_cols, feature_importance


# ============================================================================
# STEP 5: APPLY MODEL TO WORLDWIDE DATA
# ============================================================================


def predict_worldwide_spending(df_ww, model, feature_cols):
    """Predict spending categories for worldwide airports"""
    print("\n" + "=" * 80)
    print("PREDICTING WORLDWIDE SPENDING")
    print("=" * 80)

    # Prepare features
    X_ww = df_ww[feature_cols].fillna(0)

    # Predict categories
    df_ww["predicted_category"] = model.predict(X_ww)

    # Also get probability distributions for uncertainty analysis
    df_ww["pred_proba"] = list(model.predict_proba(X_ww))

    print(f"\nPredictions completed for {len(df_ww):,} passengers")
    print("\nPredicted Category Distribution:")
    print(df_ww["predicted_category"].value_counts().sort_index())

    return df_ww


# ============================================================================
# STEP 6: REVENUE CALCULATION
# ============================================================================


def calculate_revenue_per_airport(df_ww):
    """Calculate expected revenue per airport"""
    print("\n" + "=" * 80)
    print("REVENUE CALCULATION")
    print("=" * 80)

    # Define category midpoints (in EUR)
    category_midpoints = {0: 5, 1: 30, 2: 100, 3: 225, 4: 400}

    # Map predicted categories to revenue
    df_ww["predicted_revenue"] = df_ww["predicted_category"].map(category_midpoints)

    # Calculate total revenue per airport
    revenue_by_airport = (
        df_ww.groupby("shopped_at")
        .agg({"predicted_revenue": "sum", "name": "count"})  # passenger count
        .rename(
            columns={"name": "passenger_count", "predicted_revenue": "total_revenue"}
        )
    )

    # Calculate monthly revenue (assuming data represents 1 month)
    revenue_by_airport["monthly_revenue"] = revenue_by_airport["total_revenue"]
    revenue_by_airport["annual_revenue"] = revenue_by_airport["monthly_revenue"] * 12

    print("\nRevenue by Airport:")
    print(revenue_by_airport.sort_values("monthly_revenue", ascending=False))

    return revenue_by_airport


# ============================================================================
# STEP 7: PROFITABILITY ANALYSIS
# ============================================================================


def parse_lease_data():
    """Parse and clean lease data"""
    # The lease CSV has a composite header, need special handling
    df_lease = pd.read_csv(
        r"/Users/SancLee/PycharmProjects/DeloitteCaseStudy/raw/airports_terms_of_lease.csv"
    )

    # If the header is composite, parse manually
    # Expected format: "Airport,sqm of store,price per sqm/month"
    lease_data = []

    with open(
        "/Users/SancLee/PycharmProjects/DeloitteCaseStudy/raw/airports_terms_of_lease.csv",
        "r",
    ) as f:
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
    df_lease["annual_cost"] = df_lease["monthly_cost"] * 12

    return df_lease


def calculate_profitability(revenue_by_airport):
    """Calculate profitability (revenue - costs)"""
    print("\n" + "=" * 80)
    print("PROFITABILITY ANALYSIS")
    print("=" * 80)

    # Load lease data
    df_lease = parse_lease_data()

    # Merge revenue with costs
    profitability = revenue_by_airport.merge(
        df_lease[["airport", "monthly_cost", "annual_cost"]],
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

    # Calculate ROI
    profitability["profit_margin"] = (
        profitability["monthly_profit"] / profitability["monthly_revenue"] * 100
    )

    # Sort by annual profit
    profitability = profitability.sort_values("annual_profit", ascending=False)

    print("\nProfitability Ranking (Annual):")
    print("=" * 80)
    for idx, (airport, row) in enumerate(profitability.iterrows(), 1):
        print(f"\n{idx}. {airport}")
        print(f"   Passengers: {row['passenger_count']:,}")
        print(f"   Monthly Revenue: €{row['monthly_revenue']:,.0f}")
        print(f"   Monthly Cost: €{row['monthly_cost']:,.0f}")
        print(f"   Monthly Profit: €{row['monthly_profit']:,.0f}")
        print(f"   Annual Profit: €{row['annual_profit']:,.0f}")
        print(f"   Profit Margin: {row['profit_margin']:.1f}%")

    return profitability


# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================


def create_visualizations(df_eu, profitability, feature_importance):
    """Create key visualizations for presentation"""

    fig = plt.figure(figsize=(16, 10))

    # 1. Spending distribution in EU data
    ax1 = plt.subplot(2, 3, 1)
    df_eu["amount_spent_cat"].value_counts().sort_index().plot(kind="bar", ax=ax1)
    ax1.set_title("EU Spending Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Spending Category")
    ax1.set_ylabel("Number of Passengers")
    ax1.tick_params(rotation=0)

    # 2. Feature importance
    ax2 = plt.subplot(2, 3, 2)
    feature_importance.plot(
        x="feature", y="importance", kind="barh", ax=ax2, legend=False
    )
    ax2.set_title("Feature Importance", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Importance")
    ax2.set_ylabel("")

    # 3. Revenue by airport
    ax3 = plt.subplot(2, 3, 3)
    profitability["monthly_revenue"].sort_values(ascending=True).plot(
        kind="barh", ax=ax3
    )
    ax3.set_title("Monthly Revenue by Airport", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Revenue (€)")

    # 4. Cost by airport
    ax4 = plt.subplot(2, 3, 4)
    profitability["monthly_cost"].sort_values(ascending=True).plot(
        kind="barh", ax=ax4, color="orange"
    )
    ax4.set_title("Monthly Lease Cost by Airport", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Cost (€)")

    # 5. Profit by airport
    ax5 = plt.subplot(2, 3, 5)
    profitability["monthly_profit"].sort_values(ascending=True).plot(
        kind="barh", ax=ax5, color="green"
    )
    ax5.set_title("Monthly Profit by Airport", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Profit (€)")
    ax5.axvline(x=0, color="red", linestyle="--", linewidth=1)

    # 6. Profit margin by airport
    ax6 = plt.subplot(2, 3, 6)
    profitability["profit_margin"].sort_values(ascending=True).plot(
        kind="barh", ax=ax6, color="purple"
    )
    ax6.set_title("Profit Margin by Airport", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Profit Margin (%)")
    ax6.axvline(x=0, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig("case_study_analysis.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'case_study_analysis.png'")

    return fig


# ============================================================================
# STEP 9: SENSITIVITY ANALYSIS
# ============================================================================


def sensitivity_analysis(profitability, df_ww):
    """Perform sensitivity analysis on key assumptions"""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)

    print("\nScenario 1: Revenue +/- 10%")
    profitability_copy = profitability.copy()
    profitability_copy["monthly_profit_low"] = (
        profitability_copy["monthly_revenue"] * 0.9 - profitability_copy["monthly_cost"]
    )
    profitability_copy["monthly_profit_high"] = (
        profitability_copy["monthly_revenue"] * 1.1 - profitability_copy["monthly_cost"]
    )

    print("\nTop 3 airports - profit range:")
    for airport in profitability_copy.head(3).index:
        row = profitability_copy.loc[airport]
        print(
            f"{airport}: €{row['monthly_profit_low']:,.0f} to €{row['monthly_profit_high']:,.0f}"
        )

    print("\nScenario 2: What if we improve model accuracy by 5%?")
    print("(Higher spending categories increase)")
    # This is illustrative - actual implementation would require resampling

    return profitability_copy


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================


def main():
    """Main execution pipeline"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - AIRPORT EXPANSION ANALYSIS")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/9] Loading data...")
    df_eu, df_ww, df_airports, df_lease = load_data()

    # Step 2: Clean and engineer features
    print("\n[2/9] Cleaning and engineering features...")
    df_eu_clean = clean_and_engineer_features(df_eu)
    df_ww_clean = clean_and_engineer_features(df_ww)

    # Step 3: Exploratory data analysis
    print("\n[3/9] Performing exploratory data analysis...")
    correlations = perform_eda(df_eu_clean)

    # Step 4: Train model
    print("\n[4/9] Training predictive model...")
    model, feature_cols, feature_importance = train_model(df_eu_clean)

    # Step 5: Predict worldwide spending
    print("\n[5/9] Predicting worldwide spending...")
    df_ww_predictions = predict_worldwide_spending(df_ww_clean, model, feature_cols)

    # Step 6: Calculate revenue
    print("\n[6/9] Calculating revenue by airport...")
    revenue_by_airport = calculate_revenue_per_airport(df_ww_predictions)

    # Step 7: Calculate profitability
    print("\n[7/9] Calculating profitability...")
    profitability = calculate_profitability(revenue_by_airport)

    # Step 8: Create visualizations
    print("\n[8/9] Creating visualizations...")
    fig = create_visualizations(df_eu_clean, profitability, feature_importance)

    # Step 9: Sensitivity analysis
    print("\n[9/9] Performing sensitivity analysis...")
    sensitivity = sensitivity_analysis(profitability, df_ww_predictions)

    # Final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    top_3 = profitability.head(3)
    print("\nTop 3 Airports for Expansion:")
    for idx, (airport, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{idx}. {airport}")
        print(f"   Expected Annual Profit: €{row['annual_profit']:,.0f}")
        print(f"   Passenger Volume: {row['passenger_count']:,}")
        print(f"   Profit Margin: {row['profit_margin']:.1f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        "profitability": profitability,
        "model": model,
        "feature_importance": feature_importance,
        "df_ww_predictions": df_ww_predictions,
    }


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    results = main()
