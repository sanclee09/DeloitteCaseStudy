import os

# ============================================================================
# FILE PATHS
# ============================================================================

# Base directory
BASE_DIR = "/Users/SancLee/PycharmProjects/DeloitteCaseStudy"

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Input files
EU_PASSENGERS_FILE = os.path.join(RAW_DATA_DIR, "passengersEU.csv")
WW_PASSENGERS_FILE = os.path.join(RAW_DATA_DIR, "passengersWW.csv")
AIRPORTS_FILE = os.path.join(RAW_DATA_DIR, "airports.csv")
LEASE_FILE = os.path.join(RAW_DATA_DIR, "airports_terms_of_lease.csv")

# Output files
EU_CLEAN_FILE = os.path.join(PROCESSED_DATA_DIR, "df_eu_clean.csv")
WW_CLEAN_FILE = os.path.join(PROCESSED_DATA_DIR, "df_ww_clean.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.pkl")
PROFITABILITY_FILE = os.path.join(OUTPUT_DIR, "profitability_ranking.csv")
VISUALIZATION_FILE = os.path.join(OUTPUT_DIR, "case_study_analysis.png")

# ============================================================================
# BUSINESS CONSTANTS
# ============================================================================

# Spending category definitions (in EUR)
SPENDING_CATEGORIES = {
    0: {"min": 0, "max": 10, "midpoint": 5, "label": "Very Low"},
    1: {"min": 10, "max": 50, "midpoint": 30, "label": "Low"},
    2: {"min": 50, "max": 150, "midpoint": 100, "label": "Medium"},
    3: {"min": 150, "max": 300, "midpoint": 225, "label": "High"},
    4: {"min": 300, "max": 500, "midpoint": 400, "label": "Very High"},
}

# Category midpoints for revenue calculation
CATEGORY_MIDPOINTS = {
    cat: info["midpoint"] for cat, info in SPENDING_CATEGORIES.items()
}

# Reference date for age calculation
REFERENCE_DATE = "2019-12-31"

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Features to use in model
FEATURE_COLUMNS = [
    # core binaries & numerics
    "age",
    "is_male",
    "is_business",
    "has_family",
    "has_connection",
    "travel_complexity",
    "is_long_haul",
    "layover_ratio_log",
    # prefer scaled versions for times/weights (avoid mixing raw+scaled)
    "total_flighttime_scaled",
    "total_traveltime_scaled",
    "layover_time_scaled",
    "luggage_weight_kg_scaled",
    # encoded categoricals
    "shopped_at_encoded",
    "departure_IATA_1_encoded",
    "destination_IATA_1_encoded",
    "departure_IATA_2_encoded",
    "destination_IATA_2_encoded",
    "layover_category_encoded",
    # optionally keep raw luggage if you want both signals
    "luggage_weight_kg",
]

# Layover time bins (in minutes)
LAYOVER_BINS = [-1, 0, 60, 180, 1000]
LAYOVER_LABELS = ["no_layover", "short", "medium", "long"]

# Outlier detection thresholds
OUTLIER_THRESHOLDS = {
    "age": {"min": 18, "max": 90},
    "total_flighttime": {"max": 1200},  # 20 hours
    "luggage_weight_kg": {"max": 64},  # 2 bags × 32kg airline limit
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 50,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5
CV_SCORING = "f1_weighted"

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Sensitivity analysis
REVENUE_VARIANCE_LOW = 0.9  # -10%
REVENUE_VARIANCE_HIGH = 1.1  # +10%

# Months per year for annualization
MONTHS_PER_YEAR = 12

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Figure size
FIGURE_SIZE = (16, 10)

# DPI for saving figures
FIGURE_DPI = 300

# Color schemes
COLORS = {
    "spending_categories": ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1", "#5f27cd"],
    "profit_positive": "#2ecc71",
    "profit_negative": "#e74c3c",
    "revenue": "green",
    "cost": "orange",
    "profit": "green",
    "margin": "purple",
}

# ============================================================================
# DISPLAY OPTIONS
# ============================================================================

# Pandas display options
PANDAS_DISPLAY_MAX_ROWS = None
PANDAS_DISPLAY_MAX_COLUMNS = None
PANDAS_DISPLAY_WIDTH = None

# Decimal places for display
DECIMAL_PLACES = 2

# ============================================================================
# LOGGING
# ============================================================================

# Verbosity level (0=quiet, 1=normal, 2=verbose)
VERBOSITY = 1

# Print section separators
SEPARATOR = "=" * 80
SUBSEPARATOR = "-" * 80

# ============================================================================
# VALIDATION FLAGS
# ============================================================================

# Whether to save intermediate results
SAVE_INTERMEDIATE = True

# Whether to create visualizations
CREATE_VISUALIZATIONS = True

# Whether to perform sensitivity analysis
RUN_SENSITIVITY = True

print("Configuration loaded successfully!")
print(f"Base directory: {BASE_DIR}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
