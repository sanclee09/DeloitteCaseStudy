import os

# ============================================================================
# FILE PATHS
# ============================================================================

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
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_model.pkl")
PROFITABILITY_FILE = os.path.join(OUTPUT_DIR, "profitability_ranking.csv")
VISUALIZATION_FILE = os.path.join(OUTPUT_DIR, "case_study_analysis.png")
FEATURE_IMPORTANCE_FILE = os.path.join(OUTPUT_DIR, "feature_importance.png")

# ============================================================================
# BUSINESS CONSTANTS
# ============================================================================

# Spending category midpoints for revenue calculation (EUR)
CATEGORY_MIDPOINTS = {0: 5, 1: 30, 2: 100, 3: 225, 4: 400}

# Reference date for age calculation
REFERENCE_DATE = "2019-12-31"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Layover time bins (in minutes)
LAYOVER_BINS = [-1, 0, 60, 180, 1000]
LAYOVER_LABELS = ["no_layover", "short", "medium", "long"]

# Outlier detection thresholds
OUTLIER_THRESHOLDS = {
    "age": {"min": 18, "max": 90},
    "total_flighttime": {"max": 1200},  # 20 hours
    "luggage_weight_kg": {"max": 64},  # 2 bags × 32kg
}

# Core features to engineer
CORE_FEATURES = [
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

# Advanced engineered features
ADVANCED_FEATURES = [
    "is_long_haul",
    "layover_ratio",
    "layover_category",
]

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5
CV_SCORING = "f1_macro"

# ============================================================================
# OPTIMAL HYPERPARAMETERS (From Tuning on 2025-10-15)
# ============================================================================

# Set to False to use optimal params directly (faster, no tuning)
# Set to True to re-run hyperparameter search (slower, ~30 min)
ENABLE_HYPERPARAMETER_TUNING = False

# XGBoost optimal parameters (from best RandomizedSearchCV result)
OPTIMAL_XGBOOST_PARAMS = {
    # Tuned parameters
    "n_estimators": 426,
    "max_depth": 8,
    "learning_rate": 0.2040,
    "subsample": 0.9895,
    "colsample_bytree": 0.9743,
    "gamma": 0.3927,
    "reg_alpha": 0.3723,
    "reg_lambda": 1.8803,
    # Base parameters
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# XGBoost base parameters (for when tuning is enabled)
XGBOOST_BASE_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# XGBoost hyperparameter search space (for tuning)
XGBOOST_PARAM_GRID = {
    "xgb__n_estimators": [200, 300, 400, 500],
    "xgb__max_depth": [4, 6, 8, 10],
    "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "xgb__gamma": [0, 0.1, 0.3, 0.5],
    "xgb__reg_alpha": [0, 0.1, 0.5, 1.0],
    "xgb__reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

# Random Forest optimal parameters (from best RandomizedSearchCV result)
OPTIMAL_RF_PARAMS = {
    # Tuned parameters
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 50,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    # Base parameters
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Feature selection parameters
FEATURE_SELECTION_PARAMS = {
    "correlation_threshold": 0.95,  # For multicollinearity check
    "variance_threshold": 0.01,  # Minimum variance for feature
    "importance_threshold": 0.0005,  # Minimum feature importance
}

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Sensitivity analysis
REVENUE_VARIANCE_LOW = 0.9  # -10%
REVENUE_VARIANCE_HIGH = 1.1  # +10%

MONTHS_PER_YEAR = 12

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

FIGURE_SIZE = (16, 10)
FIGURE_DPI = 300

COLORS = {
    "spending_categories": ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1", "#5f27cd"],
    "profit_positive": "#2ecc71",
    "profit_negative": "#e74c3c",
    "revenue": "green",
    "cost": "orange",
}

# ============================================================================
# LOGGING
# ============================================================================

SAVE_INTERMEDIATE = True
CREATE_VISUALIZATIONS = True
RUN_SENSITIVITY = True
VERBOSITY = 1

print("✓ Configuration loaded successfully!")
