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
    initial_ww_rows = len(df_ww)
    df_ww = df_ww.drop_duplicates().reset_index(drop=True)
    removed = initial_ww_rows - len(df_ww)
    if removed > 0:
        print(
            f"  Removed {removed} duplicate rows from WW data ({removed / initial_ww_rows * 100:.1f}%)"
        )
        print(f"  WW dataset: {initial_ww_rows:,} → {len(df_ww):,} rows")

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

    # 1. Long-haul flight indicator
    df["is_long_haul"] = (df["total_flighttime"] > 360).astype(int)
    print("  ✓ Created: is_long_haul")

    # 2. Layover ratio (safer calculation with epsilon)
    df["layover_ratio"] = df["layover_time"] / (df["total_traveltime"] + 1)
    df["layover_ratio_log"] = np.log1p(df["layover_ratio"])
    print("  ✓ Created: layover_ratio, layover_ratio_log")

    # 3. Layover category (ordinal) - FIXED: handle NaN
    df["layover_category"] = pd.cut(
        df["layover_time"],
        bins=LAYOVER_BINS,
        labels=range(len(LAYOVER_LABELS)),
        include_lowest=True,
    )
    # Fill any NaN values before converting to int
    df["layover_category"] = df["layover_category"].fillna(0).astype(int)
    print("  ✓ Created: layover_category (ordinal)")

    # 4. NEW: Critical interaction features
    # Business travelers with long flights spend more
    df["business_longhaul"] = df["is_business"] * df["is_long_haul"]

    # Age-business interaction (older business travelers spend more)
    df["age_business"] = df["age"] * df["is_business"]

    # Family travel with luggage (families with heavy luggage shop more)
    df["family_luggage"] = df["has_family"] * df["luggage_weight_kg"]

    # Layover shopping opportunity (long layovers = more shopping time)
    df["layover_shopping_time"] = df["layover_time"] * (df["layover_time"] > 60).astype(
        int
    )

    # Gender-business interaction
    df["male_business"] = df["is_male"] * df["is_business"]

    print("  ✓ Created: business_longhaul, age_business, family_luggage")
    print("  ✓ Created: layover_shopping_time, male_business")

    # 5. Flight time bins (categorical spending patterns) - FIXED: handle NaN
    df["flight_time_category"] = pd.cut(
        df["total_flighttime"],
        bins=[0, 180, 360, 600, 2000],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    )
    # Fill any NaN values before converting to int
    df["flight_time_category"] = df["flight_time_category"].fillna(1).astype(int)
    print("  ✓ Created: flight_time_category")

    # 6. Age groups (spending patterns vary by age) - FIXED: handle NaN
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    )
    # Fill any NaN values before converting to int (use median age group = 2)
    df["age_group"] = df["age_group"].fillna(2).astype(int)
    print("  ✓ Created: age_group")

    # 7. Polynomial features for key numerical variables
    df["age_squared"] = df["age"] ** 2
    df["luggage_squared"] = df["luggage_weight_kg"] ** 2
    df["flighttime_log"] = np.log1p(df["total_flighttime"])
    print("  ✓ Created: age_squared, luggage_squared, flighttime_log")

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
    print("\n[1/3] Basic Features")
    df_eu = engineer_basic_features(df_eu)
    df_ww = engineer_basic_features(df_ww)

    # 2. Advanced features
    print("\n[2/3] Advanced Features")
    df_eu = engineer_advanced_features(df_eu)
    df_ww = engineer_advanced_features(df_ww)

    # 3. Encode categoricals
    print("\n[3/3] Categorical Encoding")
    df_eu, df_ww, encoder = encode_categorical_features(df_eu, df_ww, df_airports)

    # REMOVED: scaling (will be done in pipeline)
    print(f"\n✓ Feature engineering complete (scaling will be done in model pipeline)")
    print(f"  EU: {len(df_eu.columns)} columns")
    print(f"  WW: {len(df_ww.columns)} columns")

    return df_eu, df_ww, encoder


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
    print("DELOITTE CASE STUDY - DATA PREPROCESSING (NO LEAKAGE)")
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

    # 4. Feature engineering (WITHOUT scaling)
    print("\n[4/5] Feature engineering...")
    df_eu, df_ww, encoder = complete_feature_engineering(df_eu, df_ww, df_airports)

    # 5. EDA
    print("\n[5/5] Exploratory data analysis...")
    correlations = perform_eda(df_eu)

    # Save results
    if SAVE_INTERMEDIATE:
        print_section_header("SAVING RESULTS")

        save_dataframe(df_eu, EU_CLEAN_FILE, "EU Passengers (Clean)")
        save_dataframe(df_ww, WW_CLEAN_FILE, "WW Passengers (Clean)")

        # Save encoder only (no scaler)
        with open(os.path.join(PROCESSED_DATA_DIR, "airport_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

        print("✓ All preprocessing artifacts saved")

    # Summary
    print_section_header("PREPROCESSING COMPLETE")
    print(f"✓ EU dataset: {len(df_eu):,} rows, {len(df_eu.columns)} columns")
    print(f"✓ WW dataset: {len(df_ww):,} rows, {len(df_ww.columns)} columns")
    print(f"✓ Ready for model training (scaling will happen in pipeline)")

    return {
        "df_eu": df_eu,
        "df_ww": df_ww,
        "df_airports": df_airports,
        "df_lease": df_lease,
        "encoder": encoder,
        "correlations": correlations,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Preprocessing complete! Next: Run 2_model_training.py")
