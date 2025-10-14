import pandas as pd
import numpy as np
from datetime import datetime
import re
from config import REFERENCE_DATE


# ============================================================================
# DATA LOADING
# ============================================================================


def load_csv_with_info(filepath, name="Dataset"):
    """
    Load CSV file and print basic information
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {name}: {len(df):,} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found - {filepath}")
        return None
    except Exception as e:
        print(f"✗ Error loading {name}: {str(e)}")
        return None


# ============================================================================
# DATE PARSING
# ============================================================================


def parse_birth_date(date_str):
    """
    Parse birth date with multiple format support
    Handles: YYYY/MM/DD, DD/MM/YYYY, MM/DD/YYYY
    """
    if pd.isna(date_str):
        return None

    formats = ["%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except:
            continue
    return None


def calculate_age(birth_date, reference_date=REFERENCE_DATE):
    """
    Calculate age from birth date at reference date
    """
    if pd.isna(birth_date):
        return None

    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    age = ref_date.year - birth_date.year

    # Adjust if birthday hasn't occurred yet
    if (ref_date.month, ref_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


# ============================================================================
# TEXT PARSING
# ============================================================================


def extract_luggage_weight(luggage_str):
    """
    Extract weight from luggage string and convert to kg
    Handles: kg, lbs, lb, L (assuming 1L = 1kg)

    Examples:
        "Sports Bag 19l" -> 19.0
        "Bag 19kg" -> 19.0
        "Duffel Bag 7lbs" -> 3.175
        "Brief Case (4L)" -> 4.0
    """
    if pd.isna(luggage_str):
        return 0

    luggage_str = str(luggage_str).lower()

    # Extract number and unit using regex
    match = re.search(r"(\d+\.?\d*)\s*(kg|l|lbs|lb)", luggage_str)
    if match:
        weight = float(match.group(1))
        unit = match.group(2)

        # Convert to kg
        if unit in ["lbs", "lb"]:
            weight = weight * 0.453592  # lbs to kg conversion
        # L is assumed as kg (1L ≙ 1kg per case description)

        return weight

    return 0


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================


def check_missing_values(df, name="Dataset"):
    """
    Analyze missing values in dataframe
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = (
        pd.DataFrame(
            {
                "Column": missing.index,
                "Missing_Count": missing.values,
                "Missing_Percentage": missing_pct.values,
            }
        )
        .query("Missing_Count > 0")
        .sort_values("Missing_Percentage", ascending=False)
    )

    if len(missing_df) > 0:
        print(f"\n{name} - Missing Values:")
        print(missing_df.to_string(index=False))
        return missing_df
    else:
        print(f"✓ {name}: No missing values detected")
        return None


def check_duplicates(df, name="Dataset"):
    """
    Check for duplicate rows
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(
            f"⚠ {name}: {duplicates} duplicate rows found ({duplicates / len(df) * 100:.2f}%)"
        )
    else:
        print(f"✓ {name}: No duplicate rows")
    return duplicates


def check_value_ranges(df, column, min_val=None, max_val=None, name=""):
    """
    Check if values fall within expected range
    """
    out_of_range = 0

    if min_val is not None:
        out_of_range += (df[column] < min_val).sum()

    if max_val is not None:
        out_of_range += (df[column] > max_val).sum()

    if out_of_range > 0:
        pct = out_of_range / len(df) * 100
        range_str = f"[{min_val if min_val else '-∞'}, {max_val if max_val else '∞'}]"
        print(
            f"⚠ {name or column}: {out_of_range} values outside range {range_str} ({pct:.2f}%)"
        )

    return out_of_range


# ============================================================================
# FEATURE CREATION
# ============================================================================


def create_binary_feature(df, column, condition_value, new_column_name=None):
    """
    Create binary (0/1) feature from categorical column
    """
    if new_column_name is None:
        new_column_name = f"is_{condition_value}"

    df[new_column_name] = (df[column] == condition_value).astype(int)
    return df


def create_age_groups(df, age_column="age", bins=None, labels=None):
    """
    Create age group categories
    """
    if bins is None:
        bins = [0, 25, 35, 50, 65, 100]
    if labels is None:
        labels = ["young", "young_adult", "middle_age", "senior", "elderly"]

    df["age_group"] = pd.cut(df[age_column], bins=bins, labels=labels)
    return df


# ============================================================================
# STATISTICAL SUMMARIES
# ============================================================================


def print_summary_statistics(df, columns=None, name="Dataset"):
    """
    Print summary statistics for numeric columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    print(f"\n{name} - Summary Statistics:")
    print(df[columns].describe().T.round(2))


def print_category_distribution(df, column, normalize=False, name=""):
    """
    Print distribution of categorical variable
    """
    counts = df[column].value_counts().sort_index()

    if normalize:
        pct = (counts / len(df) * 100).round(2)
        result = pd.DataFrame({"Count": counts, "Percentage": pct})
    else:
        result = counts

    print(f"\n{name or column} Distribution:")
    print(result)

    return result


# ============================================================================
# DATA EXPORT
# ============================================================================


def save_dataframe(df, filepath, name="DataFrame"):
    """
    Save dataframe to CSV with error handling
    """
    try:
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {name} to: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error saving {name}: {str(e)}")
        return False


# ============================================================================
# DISPLAY FORMATTING
# ============================================================================


def print_section_header(title):
    """
    Print formatted section header
    """
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_subsection_header(title):
    """
    Print formatted subsection header
    """
    print("\n" + title)
    print("-" * 80)


def format_currency(value, currency="€"):
    """
    Format number as currency
    """
    return f"{currency}{value:,.0f}"


def format_percentage(value, decimal_places=1):
    """
    Format number as percentage
    """
    return f"{value:.{decimal_places}f}%"


# ============================================================================
# VALIDATION
# ============================================================================


def validate_dataframe(df, required_columns=None, name="DataFrame"):
    """
    Validate that dataframe has required structure
    """
    if df is None or df.empty:
        print(f"✗ {name} is None or empty")
        return False

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            print(f"✗ {name} missing required columns: {missing_cols}")
            return False

    print(f"✓ {name} validation passed")
    return True


# ============================================================================
# CORRELATION HELPERS
# ============================================================================


def find_high_correlations(df, threshold=0.8, exclude_self=True):
    """
    Find pairs of features with high correlation

    Args:
        df: DataFrame with numeric columns
        threshold: Correlation threshold (default 0.8)
        exclude_self: Whether to exclude perfect self-correlations

    Returns:
        List of dictionaries with correlation pairs
    """
    corr_matrix = df.corr()
    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1 if exclude_self else i, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr.append(
                    {
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": corr_value,
                    }
                )

    return high_corr


# ============================================================================
# OUTLIER DETECTION HELPERS
# ============================================================================


def detect_outliers_iqr(series, multiplier=1.5):
    """
    Detect outliers using IQR method

    Args:
        series: pandas Series
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Boolean mask where True indicates outlier
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series, threshold=3):
    """
    Detect outliers using Z-score method

    Args:
        series: pandas Series
        threshold: Z-score threshold (default 3)

    Returns:
        Boolean mask where True indicates outlier
    """
    from scipy import stats

    z_scores = np.abs(stats.zscore(series.fillna(series.mean())))
    return z_scores > threshold


# ============================================================================
# TIME UTILITIES
# ============================================================================


def format_duration(seconds):
    """
    Format duration in seconds to readable string

    Examples:
        45 -> "45.0s"
        125 -> "2m 5s"
        3725 -> "1h 2m 5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


# ============================================================================
# FEATURE IMPORTANCE HELPERS
# ============================================================================


def plot_feature_importance(importance_df, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt

    top_features = importance_df.head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return plt.gcf()


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UTILS MODULE - UTILITY FUNCTIONS")
    print("=" * 80)
    print("\nAvailable functions:")
    print("  Data Loading:")
    print("    - load_csv_with_info()")
    print("  Date Processing:")
    print("    - parse_birth_date()")
    print("    - calculate_age()")
    print("  Text Processing:")
    print("    - extract_luggage_weight()")
    print("  Data Quality:")
    print("    - check_missing_values()")
    print("    - check_duplicates()")
    print("    - check_value_ranges()")
    print("  Feature Creation:")
    print("    - create_binary_feature()")
    print("    - create_age_groups()")
    print("  Display:")
    print("    - print_section_header()")
    print("    - print_subsection_header()")
    print("    - format_currency()")
    print("    - format_percentage()")
    print("  Validation:")
    print("    - validate_dataframe()")
    print("  Analysis:")
    print("    - find_high_correlations()")
    print("    - detect_outliers_iqr()")
    print("    - detect_outliers_zscore()")
    print("\n✓ Utility functions loaded successfully!")
else:
    print("✓ Utils module imported successfully!")
