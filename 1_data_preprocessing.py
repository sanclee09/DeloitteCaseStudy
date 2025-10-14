from scipy import stats
import warnings

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
# MAIN PREPROCESSING PIPELINE
# ============================================================================


def main():
    """
    Main preprocessing pipeline
    """
    print("=" * 80)
    print("DELOITTE CASE STUDY - DATA PREPROCESSING")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df_eu, df_ww, df_airports, df_lease = load_all_data()

    # Step 2: Data quality checks
    print("\n[2/6] Performing data quality checks...")
    eu_quality = perform_data_quality_checks(df_eu, "EU Passengers")
    ww_quality = perform_data_quality_checks(df_ww, "WW Passengers")

    # Step 3: Outlier detection
    print("\n[3/6] Detecting outliers...")
    eu_outliers = detect_outliers(df_eu)

    # Step 4: Handle outliers
    print("\n[4/6] Handling outliers...")
    df_eu_clean = handle_outliers(df_eu, method="cap")
    df_ww_clean = handle_outliers(df_ww, method="cap")

    # Step 5: Feature engineering
    print("\n[5/6] Engineering features...")
    df_eu_engineered = engineer_features(df_eu_clean)
    df_ww_engineered = engineer_features(df_ww_clean)

    # Step 6: EDA
    print("\n[6/6] Performing exploratory data analysis...")
    correlations, spending_dist = perform_eda(df_eu_engineered)

    # Save processed data
    if SAVE_INTERMEDIATE:
        print_section_header("SAVING PROCESSED DATA")
        save_dataframe(df_eu_engineered, EU_CLEAN_FILE, "EU Passengers (Clean)")
        save_dataframe(df_ww_engineered, WW_CLEAN_FILE, "WW Passengers (Clean)")

    # Summary
    print_section_header("PREPROCESSING COMPLETE")
    print(
        f"✓ EU dataset: {len(df_eu_engineered):,} rows, {len(df_eu_engineered.columns)} features"
    )
    print(
        f"✓ WW dataset: {len(df_ww_engineered):,} rows, {len(df_ww_engineered.columns)} features"
    )
    print(f"✓ Data saved to: {PROCESSED_DATA_DIR}")

    return {
        "df_eu": df_eu_engineered,
        "df_ww": df_ww_engineered,
        "df_airports": df_airports,
        "df_lease": df_lease,
        "correlations": correlations,
        "spending_dist": spending_dist,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Data preprocessing complete!")
    print("Next step: Run 2_model_training.py")
