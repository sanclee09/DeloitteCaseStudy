import pickle
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

from config import *
from utils import *


# ============================================================================
# DATA LOADING
# ============================================================================


def load_all_data():
    """Load all datasets"""
    print_section_header("DATA LOADING")

    df_eu = load_csv_with_info(EU_PASSENGERS_FILE, "EU Passengers")
    df_ww = load_csv_with_info(WW_PASSENGERS_FILE, "WW Passengers")
    df_airports = load_csv_with_info(AIRPORTS_FILE, "Airports")
    df_lease = load_csv_with_info(LEASE_FILE, "Lease Terms")

    return df_eu, df_ww, df_airports, df_lease


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================


def perform_data_quality_checks(df, name="Dataset"):
    """Comprehensive data quality assessment"""
    print_section_header(f"DATA QUALITY - {name}")

    check_missing_values(df, name)
    check_duplicates(df, name)

    return df


# ============================================================================
# OUTLIER HANDLING
# ============================================================================


def handle_outliers(df, method="cap", percentile=(1, 99)):
    """
    Handle outliers by capping at percentiles

    Args:
        df: DataFrame
        method: 'cap' or 'remove'
        percentile: Tuple of (lower, upper) percentiles
    """
    print_section_header(f"OUTLIER HANDLING: {method.upper()}")

    df = df.copy()
    numeric_cols = [
        "age",
        "luggage_weight_kg",
        "total_flighttime",
        "total_traveltime",
        "layover_time",
    ]

    for col in numeric_cols:
        if col in df.columns:
            lower = df[col].quantile(percentile[0] / 100)
            upper = df[col].quantile(percentile[1] / 100)
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"  ✓ {col}: Capped to [{lower:.2f}, {upper:.2f}]")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_basic_features(df):
    """Create basic engineered features"""
    print_subsection_header("Basic Feature Engineering")

    df = df.copy()

    # 1. Parse dates and calculate age
    df["birth_date_parsed"] = df["birth_date"].apply(parse_birth_date)
    df["age"] = df["birth_date_parsed"].apply(calculate_age)
    print("  ✓ Created: age")

    # 2. Extract luggage weight
    df["luggage_weight_kg"] = df["luggage"].apply(extract_luggage_weight)
    print("  ✓ Created: luggage_weight_kg")

    # 3. Binary features
    df["is_business"] = (df["business_trip"] == "yes").astype(int)
    df["has_family"] = (df["traveled_with_family"] == "yes").astype(int)
    df["has_connection"] = (~df["flight_number_2"].isna()).astype(int)
    df["is_male"] = (df["sex"] == "m").astype(int)
    print("  ✓ Created: is_business, has_family, has_connection, is_male")

    # 4. Handle missing values
    df["layover_time"] = df["layover_time"].fillna(0)
    if df["age"].isna().sum() > 0:
        df["age"] = df["age"].fillna(df["age"].median())

    return df


def engineer_advanced_features(df):
    """Create advanced derived features"""
    print_subsection_header("Advanced Feature Engineering")

    df = df.copy()

    # 1. Travel complexity score
    df["travel_complexity"] = (
        df["has_connection"] * 2
        + (df["layover_time"] > 0).astype(int)
        + (df["luggage_weight_kg"] > 20).astype(int)
    )
    print("  ✓ Created: travel_complexity")

    # 2. Long-haul flight indicator
    df["is_long_haul"] = (df["total_flighttime"] > 360).astype(int)
    print("  ✓ Created: is_long_haul")

    # 3. Layover ratio (safer calculation with epsilon)
    df["layover_ratio"] = df["layover_time"] / (df["total_traveltime"] + 1)
    df["layover_ratio_log"] = np.log1p(df["layover_ratio"])
    print("  ✓ Created: layover_ratio, layover_ratio_log")

    # 4. Layover category (ordinal)
    df["layover_category"] = pd.cut(
        df["layover_time"], bins=LAYOVER_BINS, labels=range(len(LAYOVER_LABELS))
    ).astype(int)
    print("  ✓ Created: layover_category (ordinal)")

    return df


def encode_categorical_features(df_eu, df_ww, df_airports):
    """
    Encode categorical features (airports)
    Fit encoder on combined EU+WW+airports data
    """
    print_subsection_header("Encoding Categorical Features")

    # Collect all unique airports
    all_airports = set()
    if "iata_code" in df_airports.columns:
        all_airports.update(df_airports["iata_code"].dropna().unique())

    for df in [df_eu, df_ww]:
        for col in [
            "shopped_at",
            "departure_IATA_1",
            "destination_IATA_1",
            "departure_IATA_2",
            "destination_IATA_2",
        ]:
            if col in df.columns:
                all_airports.update(df[col].dropna().unique())

    # Create encoder
    all_airports = sorted(list(all_airports))
    encoder = LabelEncoder()
    encoder.fit(all_airports)

    print(
        f"  ✓ Encoded {len(all_airports)} unique airports (0-{len(all_airports) - 1})"
    )

    # Apply encoding
    airport_cols = [
        "shopped_at",
        "departure_IATA_1",
        "destination_IATA_1",
        "departure_IATA_2",
        "destination_IATA_2",
    ]

    for df in [df_eu, df_ww]:
        for col in airport_cols:
            if col in df.columns:
                new_col = f"{col}_encoded"
                df[new_col] = df[col].apply(
                    lambda x: (
                        encoder.transform([x])[0]
                        if pd.notna(x) and x in encoder.classes_
                        else -1
                    )
                )

    print(f"  ✓ Created encoded columns for airport features")

    return df_eu, df_ww, encoder


def scale_numerical_features(df_eu, df_ww):
    """
    Standardize numerical features
    Fit on EU data, transform both datasets
    """
    print_subsection_header("Scaling Numerical Features")

    df_eu = df_eu.copy()
    df_ww = df_ww.copy()

    # Features to scale
    numerical_features = [
        "age",
        "luggage_weight_kg",
        "total_flighttime",
        "total_traveltime",
        "layover_time",
        "layover_ratio_log",
    ]

    features_to_scale = [f for f in numerical_features if f in df_eu.columns]

    # Fit scaler on EU data
    scaler = StandardScaler()
    scaler.fit(df_eu[features_to_scale])

    # Transform both datasets
    df_eu_scaled = scaler.transform(df_eu[features_to_scale])
    df_ww_scaled = scaler.transform(df_ww[features_to_scale])

    # Create scaled columns
    for i, feature in enumerate(features_to_scale):
        df_eu[f"{feature}_scaled"] = df_eu_scaled[:, i]
        df_ww[f"{feature}_scaled"] = df_ww_scaled[:, i]
        print(f"  ✓ Scaled: {feature}")

    return df_eu, df_ww, scaler


# ============================================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================


def complete_feature_engineering(df_eu, df_ww, df_airports):
    """Complete feature engineering pipeline"""
    print_section_header("FEATURE ENGINEERING PIPELINE")

    # 1. Basic features
    print("\n[1/4] Basic Features")
    df_eu = engineer_basic_features(df_eu)
    df_ww = engineer_basic_features(df_ww)

    # 2. Advanced features
    print("\n[2/4] Advanced Features")
    df_eu = engineer_advanced_features(df_eu)
    df_ww = engineer_advanced_features(df_ww)

    # 3. Encode categoricals
    print("\n[3/4] Categorical Encoding")
    df_eu, df_ww, encoder = encode_categorical_features(df_eu, df_ww, df_airports)

    # 4. Scale numericals
    print("\n[4/4] Numerical Scaling")
    df_eu, df_ww, scaler = scale_numerical_features(df_eu, df_ww)

    print(f"\n✓ Feature engineering complete")
    print(f"  EU: {len(df_eu.columns)} columns")
    print(f"  WW: {len(df_ww.columns)} columns")

    return df_eu, df_ww, encoder, scaler


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================


def perform_eda(df_eu):
    """Perform exploratory data analysis"""
    print_section_header("EXPLORATORY DATA ANALYSIS")

    # 1. Target distribution
    print_subsection_header("Target Distribution")
    print_category_distribution(df_eu, "amount_spent_cat", normalize=True)

    # 2. Key relationships
    print_subsection_header("Spending by Key Groups")

    for group in ["is_business", "has_family", "has_connection", "is_male"]:
        if group in df_eu.columns:
            print(f"\nBy {group}:")
            print(df_eu.groupby(group)["amount_spent_cat"].agg(["mean", "count"]))

    # 3. Correlation analysis
    print_subsection_header("Correlation with Target")

    numeric_cols = [c for c in df_eu.columns if c.endswith("_scaled")] + [
        "is_business",
        "has_family",
        "has_connection",
        "is_male",
        "travel_complexity",
        "is_long_haul",
        "layover_category",
    ]

    numeric_cols = [c for c in numeric_cols if c in df_eu.columns]

    if "amount_spent_cat" in df_eu.columns:
        correlations = df_eu[numeric_cols + ["amount_spent_cat"]].corr()[
            "amount_spent_cat"
        ]
        correlations = correlations.drop("amount_spent_cat").sort_values(
            ascending=False
        )

        print("\nTop correlations with spending:")
        print(correlations.head(10))

        # Identify strong predictors
        strong_predictors = correlations[abs(correlations) > 0.3]
        print(f"\n✓ Strong predictors (|r| > 0.3): {len(strong_predictors)}")

        return correlations

    return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main preprocessing pipeline"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - DATA PREPROCESSING (CLEANED)")
    print("=" * 80)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df_eu, df_ww, df_airports, df_lease = load_all_data()

    # 2. Data quality checks
    print("\n[2/5] Data quality checks...")
    df_eu = perform_data_quality_checks(df_eu, "EU Passengers")
    df_ww = perform_data_quality_checks(df_ww, "WW Passengers")

    # 3. Handle outliers
    print("\n[3/5] Handling outliers...")
    df_eu = handle_outliers(df_eu)
    df_ww = handle_outliers(df_ww)

    # 4. Feature engineering
    print("\n[4/5] Feature engineering...")
    df_eu, df_ww, encoder, scaler = complete_feature_engineering(
        df_eu, df_ww, df_airports
    )

    # 5. EDA
    print("\n[5/5] Exploratory data analysis...")
    correlations = perform_eda(df_eu)

    # Save results
    if SAVE_INTERMEDIATE:
        print_section_header("SAVING RESULTS")

        save_dataframe(df_eu, EU_CLEAN_FILE, "EU Passengers (Clean)")
        save_dataframe(df_ww, WW_CLEAN_FILE, "WW Passengers (Clean)")

        # Save encoders
        with open(os.path.join(PROCESSED_DATA_DIR, "airport_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
        with open(os.path.join(PROCESSED_DATA_DIR, "feature_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        print("✓ All preprocessing artifacts saved")

    # Summary
    print_section_header("PREPROCESSING COMPLETE")
    print(f"✓ EU dataset: {len(df_eu):,} rows, {len(df_eu.columns)} columns")
    print(f"✓ WW dataset: {len(df_ww):,} rows, {len(df_ww.columns)} columns")
    print(f"✓ Ready for model training")

    return {
        "df_eu": df_eu,
        "df_ww": df_ww,
        "df_airports": df_airports,
        "df_lease": df_lease,
        "encoder": encoder,
        "scaler": scaler,
        "correlations": correlations,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Preprocessing complete! Next: Run 2_model_training.py")
