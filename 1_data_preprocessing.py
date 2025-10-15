import pickle

from scipy import stats
import warnings

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

from config import *
from utils import *


# ============================================================================
# MAIN DATA LOADING
# ============================================================================


def load_all_data():
    """
    Load all datasets
    """
    print_section_header("DATA LOADING")

    df_eu = load_csv_with_info(EU_PASSENGERS_FILE, "EU Passengers")
    df_ww = load_csv_with_info(WW_PASSENGERS_FILE, "WW Passengers")
    df_airports = load_csv_with_info(AIRPORTS_FILE, "Airports")
    df_lease = load_csv_with_info(LEASE_FILE, "Lease Terms")

    return df_eu, df_ww, df_airports, df_lease


# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================


def perform_data_quality_checks(df, name="Dataset"):
    """
    Comprehensive data quality assessment
    """
    print_section_header(f"DATA QUALITY ASSESSMENT - {name}")

    # 1. Missing values
    print_subsection_header("1. Missing Values")
    missing_report = check_missing_values(df, name)

    # 2. Duplicates
    print_subsection_header("2. Duplicate Records")
    dup_count = check_duplicates(df, name)

    # 3. Data types
    print_subsection_header("3. Data Types")
    print(df.dtypes)

    # 4. Basic statistics
    print_subsection_header("4. Numeric Column Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().T)

    return {"missing_report": missing_report, "duplicates": dup_count}


# ============================================================================
# OUTLIER DETECTION
# ============================================================================


def detect_outliers(df, columns=None):
    """
    Detect outliers using multiple methods
    """
    print_section_header("OUTLIER DETECTION")

    if columns is None:
        columns = [
            "age",
            "luggage_weight_kg",
            "total_flighttime",
            "total_traveltime",
            "layover_time",
        ]

    outlier_summary = []

    for col in columns:
        if col not in df.columns:
            continue

        print_subsection_header(f"{col.upper()}")

        col_data = df[col].dropna()

        # Basic statistics
        print(f"Count: {len(col_data):,}")
        print(f"Mean: {col_data.mean():.2f}")
        print(f"Median: {col_data.median():.2f}")
        print(f"Std: {col_data.std():.2f}")
        print(f"Range: [{col_data.min():.2f}, {col_data.max():.2f}]")

        # IQR Method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        iqr_outliers = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])

        print(f"\nIQR Method (1.5×IQR):")
        print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outliers: {iqr_outliers} ({iqr_outliers / len(df) * 100:.2f}%)")

        # Z-score method
        z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
        z_outliers = len(df[z_scores > 3])

        print(f"\nZ-Score Method (|z| > 3):")
        print(f"  Outliers: {z_outliers} ({z_outliers / len(df) * 100:.2f}%)")

        # Domain-specific checks
        if col in OUTLIER_THRESHOLDS:
            thresholds = OUTLIER_THRESHOLDS[col]
            if "min" in thresholds:
                domain_out_low = len(df[df[col] < thresholds["min"]])
                if domain_out_low > 0:
                    print(
                        f"\nDomain Check (< {thresholds['min']}): {domain_out_low} records"
                    )
            if "max" in thresholds:
                domain_out_high = len(df[df[col] > thresholds["max"]])
                if domain_out_high > 0:
                    print(
                        f"Domain Check (> {thresholds['max']}): {domain_out_high} records"
                    )

        outlier_summary.append(
            {
                "feature": col,
                "iqr_outliers": iqr_outliers,
                "iqr_pct": iqr_outliers / len(df) * 100,
                "z_outliers": z_outliers,
                "z_pct": z_outliers / len(df) * 100,
            }
        )

    print_subsection_header("OUTLIER SUMMARY")
    summary_df = pd.DataFrame(outlier_summary)
    print(summary_df.to_string(index=False))

    return summary_df


def handle_outliers(df, method="cap"):
    """
    Handle outliers - cap at 1st and 99th percentiles
    """
    print_section_header(f"OUTLIER HANDLING: {method.upper()} METHOD")

    df = df.copy()

    if method == "cap":
        numeric_cols = [
            "age",
            "luggage_weight_kg",
            "total_flighttime",
            "total_traveltime",
            "layover_time",
        ]

        for col in numeric_cols:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)

                df[col] = df[col].clip(lower=lower, upper=upper)

                print(f"✓ {col}: Capped to [{lower:.2f}, {upper:.2f}]")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_features(df):
    """
    Create all engineered features
    """
    print_section_header("FEATURE ENGINEERING")

    df = df.copy()

    # 1. Parse birth dates and calculate age
    print("1. Parsing birth dates and calculating age...")
    df["birth_date_parsed"] = df["birth_date"].apply(parse_birth_date)
    df["age"] = df["birth_date_parsed"].apply(calculate_age)
    print(f"   ✓ Created 'age' feature")

    # 2. Extract luggage weight
    print("2. Extracting luggage weight...")
    df["luggage_weight_kg"] = df["luggage"].apply(extract_luggage_weight)
    print(f"   ✓ Created 'luggage_weight_kg' feature")

    # 3. Binary features
    print("3. Creating binary features...")
    df["is_business"] = (df["business_trip"] == "yes").astype(int)
    df["has_family"] = (df["traveled_with_family"] == "yes").astype(int)
    df["has_connection"] = (~df["flight_number_2"].isna()).astype(int)
    df["is_male"] = (df["sex"] == "m").astype(int)
    print(f"   ✓ Created binary features")

    # 4. Handle missing values
    print("4. Handling missing values...")
    df["layover_time"] = df["layover_time"].fillna(0)
    if df["age"].isna().sum() > 0:
        df["age"] = df["age"].fillna(df["age"].median())
        print(f"   ✓ Filled {df['age'].isna().sum()} missing age values with median")

    # 5. Categorical features
    print("5. Creating categorical features...")
    df["layover_category"] = pd.cut(
        df["layover_time"], bins=LAYOVER_BINS, labels=LAYOVER_LABELS
    )
    print(f"   ✓ Created 'layover_category' feature")

    # 6. Advanced features (optional)
    print("6. Creating advanced features...")
    df["travel_complexity"] = (
        df["has_connection"] * 2
        + (df["layover_time"] > 0).astype(int)
        + (df["luggage_weight_kg"] > 20).astype(int)
    )
    df["is_long_haul"] = (df["total_flighttime"] > 360).astype(int)  # > 6 hours
    df["layover_ratio"] = df["layover_time"] / (df["total_traveltime"] + 1)
    print(f"   ✓ Created advanced features")

    print(f"\n✓ Feature engineering complete. Total columns: {len(df.columns)}")

    return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================


def perform_eda(df_eu):
    """
    Comprehensive exploratory data analysis
    """
    print_section_header("EXPLORATORY DATA ANALYSIS")

    # 1. Spending distribution
    print_subsection_header("1. Spending Category Distribution")
    spending_dist = print_category_distribution(
        df_eu, "amount_spent_cat", normalize=True, name="Spending Categories"
    )

    # 2. Spending by demographic groups
    print_subsection_header("2. Average Spending by Key Groups")

    print("\nBy Business Trip:")
    print(df_eu.groupby("is_business")["amount_spent_cat"].agg(["mean", "count"]))

    print("\nBy Family Travel:")
    print(df_eu.groupby("has_family")["amount_spent_cat"].agg(["mean", "count"]))

    print("\nBy Connection Flight:")
    connection_stats = df_eu.groupby("has_connection")["amount_spent_cat"].agg(
        ["mean", "count"]
    )
    print(connection_stats)
    print(
        f"\n⭐ KEY INSIGHT: Passengers with connections spend "
        + f"{connection_stats.loc[1, 'mean'] / connection_stats.loc[0, 'mean']:.2f}x more!"
    )

    print("\nBy Gender:")
    print(df_eu.groupby("is_male")["amount_spent_cat"].agg(["mean", "count"]))

    # 3. Correlations
    print_subsection_header("3. Feature Correlations with Spending")

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

    # Identify strong correlations
    strong_corr = correlations[abs(correlations) > 0.3]
    print(
        f"\n⭐ Strong predictors (|r| > 0.3): {len(strong_corr) - 1}"
    )  # -1 for itself
    print(strong_corr[1:])  # Exclude itself

    # 4. Multicollinearity check
    print_subsection_header("4. Multicollinearity Check")

    feature_corr = df_eu[numeric_cols].corr()
    high_corr_pairs = []

    for i in range(len(feature_corr.columns)):
        for j in range(i + 1, len(feature_corr.columns)):
            if abs(feature_corr.iloc[i, j]) > 0.8:
                high_corr_pairs.append(
                    {
                        "feature_1": feature_corr.columns[i],
                        "feature_2": feature_corr.columns[j],
                        "correlation": feature_corr.iloc[i, j],
                    }
                )

    if high_corr_pairs:
        print("⚠ High correlations detected (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(
                f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}"
            )
    else:
        print("✓ No severe multicollinearity detected (all |r| < 0.8)")

    return correlations, spending_dist


# ============================================================================
# AIRPORT ENCODING
# ============================================================================


def create_airport_encodings(df_eu, df_ww, df_airports):
    """
    Create consistent label encoding for all airport codes

    Returns:
        encoder: Fitted LabelEncoder
        airport_mapping: Dict mapping IATA codes to integers
    """
    print_section_header("CREATING AIRPORT ENCODINGS")

    # Get all unique airport codes from all sources
    all_airports = set()

    # From airports reference file
    if "iata_code" in df_airports.columns:
        all_airports.update(df_airports["iata_code"].dropna().unique())

    # From EU passengers
    for col in [
        "shopped_at",
        "departure_IATA_1",
        "destination_IATA_1",
        "departure_IATA_2",
        "destination_IATA_2",
    ]:
        if col in df_eu.columns:
            all_airports.update(df_eu[col].dropna().unique())

    # From WW passengers
    for col in [
        "shopped_at",
        "departure_IATA_1",
        "destination_IATA_1",
        "departure_IATA_2",
        "destination_IATA_2",
    ]:
        if col in df_ww.columns:
            all_airports.update(df_ww[col].dropna().unique())

    # Create encoder
    all_airports = sorted(list(all_airports))
    encoder = LabelEncoder()
    encoder.fit(all_airports)

    # Create mapping for reference
    airport_mapping = {code: idx for idx, code in enumerate(encoder.classes_)}

    print(f"✓ Encoded {len(all_airports)} unique airports")
    print(f"  Encoding range: 0 to {len(all_airports) - 1}")
    print(f"\nSample mappings:")
    for code in list(airport_mapping.keys())[:5]:
        print(f"  {code}: {airport_mapping[code]}")

    return encoder, airport_mapping


def encode_airport_columns(df, encoder, columns_to_encode):
    """
    Apply airport encoding to specified columns

    Args:
        df: DataFrame
        encoder: Fitted LabelEncoder
        columns_to_encode: List of column names to encode

    Returns:
        df: DataFrame with new encoded columns
    """
    df = df.copy()

    for col in columns_to_encode:
        if col in df.columns:
            # Create new encoded column
            new_col = f"{col}_encoded"

            # Handle missing values: assign -1 (unknown airport)
            df[new_col] = df[col].apply(
                lambda x: (
                    encoder.transform([x])[0]
                    if pd.notna(x) and x in encoder.classes_
                    else -1
                )
            )

            print(
                f"  ✓ Encoded {col} → {new_col} (unique values: {df[new_col].nunique()})"
            )

    return df


# ============================================================================
# LAYOVER CATEGORY ENCODING
# ============================================================================


def encode_layover_category(df):
    """
    Ordinal encoding for layover categories (ordered)
    no_layover < short < medium < long
    """
    df = df.copy()

    if "layover_category" in df.columns:
        # Ordinal mapping
        layover_mapping = {"no_layover": 0, "short": 1, "medium": 2, "long": 3}

        df["layover_category_encoded"] = df["layover_category"].map(layover_mapping)

        print("✓ Encoded layover_category (ordinal):")
        print("  no_layover=0, short=1, medium=2, long=3")
        print(
            f"  Encoded values: {sorted(df['layover_category_encoded'].dropna().unique())}"
        )

    return df


# ============================================================================
# LAYOVER RATIO TRANSFORMATION
# ============================================================================


def transform_layover_ratio(df):
    """
    Apply log transformation to layover_ratio to handle skewness
    """
    df = df.copy()

    if "layover_ratio" in df.columns:
        # Log(1 + x) transformation to handle zeros
        df["layover_ratio_log"] = np.log1p(df["layover_ratio"])

        print("✓ Applied log1p transform to layover_ratio")
        print(
            f"  Original range: [{df['layover_ratio'].min():.4f}, {df['layover_ratio'].max():.4f}]"
        )
        print(
            f"  Transformed range: [{df['layover_ratio_log'].min():.4f}, {df['layover_ratio_log'].max():.4f}]"
        )

        # Also clip extreme outliers if needed
        q99 = df["layover_ratio_log"].quantile(0.99)
        df["layover_ratio_log"] = df["layover_ratio_log"].clip(upper=q99)

    return df


# ============================================================================
# NUMERICAL SCALING
# ============================================================================


def scale_numerical_features(df_eu, df_ww, time_features):
    """
    Apply MinMax scaling to time-based features
    Fit on EU data, transform both EU and WW

    Args:
        df_eu: EU DataFrame (training data)
        df_ww: WW DataFrame (prediction data)
        time_features: List of columns to scale

    Returns:
        df_eu_scaled, df_ww_scaled: DataFrames with scaled features
        scaler: Fitted MinMaxScaler for reference
    """
    print_section_header("SCALING NUMERICAL FEATURES")

    df_eu = df_eu.copy()
    df_ww = df_ww.copy()

    # Features to scale
    features_to_scale = [f for f in time_features if f in df_eu.columns]

    print(f"Scaling {len(features_to_scale)} features: {features_to_scale}")

    # Fit scaler on EU data only
    scaler = MinMaxScaler()
    scaler.fit(df_eu[features_to_scale])

    # Transform both datasets
    df_eu_scaled_values = scaler.transform(df_eu[features_to_scale])
    df_ww_scaled_values = scaler.transform(df_ww[features_to_scale])

    # Create new scaled columns
    for i, feature in enumerate(features_to_scale):
        new_col = f"{feature}_scaled"
        df_eu[new_col] = df_eu_scaled_values[:, i]
        df_ww[new_col] = df_ww_scaled_values[:, i]

        print(f"  ✓ Scaled {feature}")
        print(
            f"    EU:  [{df_eu[feature].min():.1f}, {df_eu[feature].max():.1f}] "
            f"→ [{df_eu[new_col].min():.4f}, {df_eu[new_col].max():.4f}]"
        )
        print(
            f"    WW:  [{df_ww[feature].min():.1f}, {df_ww[feature].max():.1f}] "
            f"→ [{df_ww[new_col].min():.4f}, {df_ww[new_col].max():.4f}]"
        )

    return df_eu, df_ww, scaler


# ============================================================================
# INTEGRATED FEATURE ENGINEERING (ENHANCED)
# ============================================================================


def engineer_features_enhanced(df_eu, df_ww, df_airports):
    """
    Enhanced feature engineering with encoding and scaling

    This replaces or extends your existing engineer_features() function
    """
    print_section_header("ENHANCED FEATURE ENGINEERING")

    # 1. Original feature engineering (keep your existing code)
    print("\n[Step 1] Basic feature engineering...")
    print("  (age, luggage_weight, binary features, etc.)")
    # ... your existing feature engineering code here ...

    # 2. Airport encoding
    print("\n[Step 2] Encoding airport codes...")
    encoder, airport_mapping = create_airport_encodings(df_eu, df_ww, df_airports)

    airport_cols = [
        "shopped_at",
        "departure_IATA_1",
        "destination_IATA_1",
        "departure_IATA_2",
        "destination_IATA_2",
    ]

    df_eu = encode_airport_columns(df_eu, encoder, airport_cols)
    df_ww = encode_airport_columns(df_ww, encoder, airport_cols)

    # 3. Layover category encoding
    print("\n[Step 3] Encoding layover category...")
    df_eu = encode_layover_category(df_eu)
    df_ww = encode_layover_category(df_ww)

    # 4. Layover ratio transformation
    print("\n[Step 4] Transforming layover ratio...")
    df_eu = transform_layover_ratio(df_eu)
    df_ww = transform_layover_ratio(df_ww)

    # 5. Numerical scaling
    print("\n[Step 5] Scaling time features...")
    time_features = [
        "total_flighttime",
        "total_traveltime",
        "layover_time",
        "luggage_weight_kg",
    ]
    df_eu, df_ww, scaler = scale_numerical_features(df_eu, df_ww, time_features)

    print("\n✓ Enhanced feature engineering complete")
    print(f"  EU columns: {len(df_eu.columns)}")
    print(f"  WW columns: {len(df_ww.columns)}")

    # Return encoder and scaler for later use
    return df_eu, df_ww, encoder, scaler


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================


def main():
    """
    Main preprocessing pipeline with enhanced feature engineering
    """
    print("=" * 80)
    print("DELOITTE CASE STUDY - DATA PREPROCESSING")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    df_eu, df_ww, df_airports, df_lease = load_all_data()

    # Step 2: Data quality checks
    print("\n[2/7] Performing data quality checks...")
    eu_quality = perform_data_quality_checks(df_eu, "EU Passengers")
    ww_quality = perform_data_quality_checks(df_ww, "WW Passengers")

    # Step 3: Outlier detection
    print("\n[3/7] Detecting outliers...")
    eu_outliers = detect_outliers(df_eu)

    # Step 4: Handle outliers
    print("\n[4/7] Handling outliers...")
    df_eu_clean = handle_outliers(df_eu, method="cap")
    df_ww_clean = handle_outliers(df_ww, method="cap")

    # Step 5: Basic feature engineering (your existing function)
    print("\n[5/7] Basic feature engineering...")
    df_eu_basic = engineer_features(df_eu_clean)
    df_ww_basic = engineer_features(df_ww_clean)

    # Step 6: Enhanced feature engineering (NEW)
    print("\n[6/7] Enhanced feature engineering...")
    print("  (airport encoding, scaling, transformations)")

    # 6a. Airport encoding
    print("\n  Creating airport encodings...")
    encoder, airport_mapping = create_airport_encodings(
        df_eu_basic, df_ww_basic, df_airports
    )

    airport_cols = [
        "shopped_at",
        "departure_IATA_1",
        "destination_IATA_1",
        "departure_IATA_2",
        "destination_IATA_2",
    ]

    df_eu_encoded = encode_airport_columns(df_eu_basic, encoder, airport_cols)
    df_ww_encoded = encode_airport_columns(df_ww_basic, encoder, airport_cols)

    # 6b. Layover category encoding
    print("\n  Encoding layover category...")
    df_eu_encoded = encode_layover_category(df_eu_encoded)
    df_ww_encoded = encode_layover_category(df_ww_encoded)

    # 6c. Layover ratio transformation
    print("\n  Transforming layover ratio...")
    df_eu_encoded = transform_layover_ratio(df_eu_encoded)
    df_ww_encoded = transform_layover_ratio(df_ww_encoded)

    # 6d. Numerical scaling
    print("\n  Scaling numerical features...")
    time_features = [
        "total_flighttime",
        "total_traveltime",
        "layover_time",
        "luggage_weight_kg",
    ]
    df_eu_engineered, df_ww_engineered, scaler = scale_numerical_features(
        df_eu_encoded, df_ww_encoded, time_features
    )

    print(f"\n  ✓ Enhanced features added")
    print(f"    EU columns: {len(df_eu_engineered.columns)}")
    print(f"    WW columns: {len(df_ww_engineered.columns)}")

    # Step 7: EDA
    print("\n[7/7] Performing exploratory data analysis...")
    correlations, spending_dist = perform_eda(df_eu_engineered)

    # Save processed data and encoders
    if SAVE_INTERMEDIATE:
        print_section_header("SAVING PROCESSED DATA & ENCODERS")

        # Save dataframes
        save_dataframe(df_eu_engineered, EU_CLEAN_FILE, "EU Passengers (Clean)")
        save_dataframe(df_ww_engineered, WW_CLEAN_FILE, "WW Passengers (Clean)")

        # Save encoders and scalers for reproducibility
        encoder_file = os.path.join(PROCESSED_DATA_DIR, "airport_encoder.pkl")
        scaler_file = os.path.join(PROCESSED_DATA_DIR, "time_scaler.pkl")

        with open(encoder_file, "wb") as f:
            pickle.dump(encoder, f)
        print(f"✓ Saved airport encoder to: {encoder_file}")

        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved time scaler to: {scaler_file}")

    # Summary
    print_section_header("PREPROCESSING COMPLETE")
    print(
        f"✓ EU dataset: {len(df_eu_engineered):,} rows, {len(df_eu_engineered.columns)} features"
    )
    print(
        f"✓ WW dataset: {len(df_ww_engineered):,} rows, {len(df_ww_engineered.columns)} features"
    )
    print(f"✓ Data saved to: {PROCESSED_DATA_DIR}")

    # Show new features created
    basic_cols = set(df_eu_basic.columns)
    enhanced_cols = set(df_eu_engineered.columns)
    new_features = enhanced_cols - basic_cols

    print(f"\n✓ New features created: {len(new_features)}")
    if new_features:
        print("  Enhanced features:")
        for feat in sorted(new_features):
            print(f"    - {feat}")

    return {
        "df_eu": df_eu_engineered,
        "df_ww": df_ww_engineered,
        "df_airports": df_airports,
        "df_lease": df_lease,
        "correlations": correlations,
        "spending_dist": spending_dist,
        "airport_encoder": encoder,
        "time_scaler": scaler,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Data preprocessing complete!")
    print("Next step: Run 2_model_training.py")
