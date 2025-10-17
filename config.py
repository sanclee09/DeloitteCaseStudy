import os

import numpy as np

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
# BUSINESS CONSTANTS - REALISTIC ASSUMPTIONS
# ============================================================================

# Spending category midpoints for revenue calculation (EUR)
# Category 4: Use TRUE midpoint of 300-500 EUR = 375 EUR (not 400)
CATEGORY_MIDPOINTS = {0: 5, 1: 30, 2: 100, 3: 225, 4: 375}

# Reference date for age calculation
REFERENCE_DATE = "2019-12-31"

# ============================================================================
# REALISTIC P&L ASSUMPTIONS
# ============================================================================

# Gross Margin (Revenue - COGS) / Revenue
# Typical retail fashion: 55-65%
# We use 60% as baseline
GROSS_MARGIN = 0.60  # 60% gross margin → COGS = 40% of revenue

# Operating Expenses
# Staff costs per sqm per month (EUR)
# Assumes: ~1 FTE per 40-50 sqm, at ~€3,500/month salary + benefits
# So: €3,500 / 40 sqm ≈ €87.50/sqm, rounded to €90
STAFF_COST_PER_SQM = 90  # EUR/sqm/month

# Overhead (utilities, shrinkage, operations, marketing)
# As % of revenue
OVERHEAD_PCT = 0.05  # 5% of revenue

# Note: Rent/lease costs are loaded from the lease file
# Note: We're ignoring CAPEX (store fit-out) for simplicity,
#       but in reality would amortize ~€500-1000/sqm over 3-5 years

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Layover time bins (in minutes)
LAYOVER_BINS = [-np.inf, 30, 90, 180, np.inf]
LAYOVER_LABELS = ["no_layover", "short", "medium", "long"]

# Outlier detection thresholds
OUTLIER_THRESHOLDS = {
    "age": {"min": 18, "max": 90},
    "total_flighttime": {"max": 1200},  # 20 hours
    "luggage_weight_kg": {"max": 64},  # 2 bags × 32kg
}

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
# OPTIMAL HYPERPARAMETERS
# ============================================================================

# Set to False to use optimal params directly (faster, no tuning)
# Set to True to re-run hyperparameter search (slower, ~30 min)
ENABLE_HYPERPARAMETER_TUNING = False

# XGBoost optimal parameters
OPTIMAL_XGBOOST_PARAMS = {
    # Tree structure
    "n_estimators": 500,  # Increased from 426
    "max_depth": 7,  # Slightly reduced for less overfitting
    "learning_rate": 0.1,  # Reduced from 0.204 for better generalization
    # Sampling
    "subsample": 0.9,  # Slightly reduced
    "colsample_bytree": 0.9,  # Slightly reduced
    "colsample_bylevel": 0.9,  # Additional regularization
    # Regularization
    "gamma": 0.5,  # Increased from 0.3927
    "reg_alpha": 0.5,  # Increased from 0.3723
    "reg_lambda": 2.0,  # Increased from 1.8803
    "min_child_weight": 3,  # Prevent overfitting
    # Performance
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",  # Faster training
    "max_bin": 256,  # More granular splits
}

# XGBoost base parameters (for when tuning is enabled)
XGBOOST_BASE_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# Feature selection parameters
FEATURE_SELECTION_PARAMS = {
    "correlation_threshold": 0.98,  # For multicollinearity check
    "variance_threshold": 0.005,  # Minimum variance for feature
    "importance_threshold": 0.0001,  # Minimum feature importance (not used in new version)
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

# ============================================================================
# DOCUMENTATION OF ASSUMPTIONS
# ============================================================================

ASSUMPTIONS = """
KEY BUSINESS ASSUMPTIONS:

1. REVENUE:
   - Category midpoints represent average spending within each bracket
   - Category 4: €375 (true midpoint of €300-500 range)
   - December 2019 patterns are representative of annual behavior

2. COSTS:
   - Gross Margin: 60% (industry standard for fashion retail)
   - COGS: 40% of revenue (cost to acquire goods)
   - Staff: €90/sqm/month (~1 FTE per 40-50 sqm)
   - Overhead: 5% of revenue (utilities, shrinkage, ops)
   - Rent: Per lease terms (provided in data)
   - CAPEX: Not included (would add ~€50-100/sqm/month if amortized)

3. GENERALIZABILITY:
   - EU passenger behavior patterns apply to worldwide markets
   - Cultural differences in spending not explicitly modeled
   - Economic conditions assumed similar across regions

4. DATA QUALITY:
   - Single month of data (December 2019) limits seasonality analysis
   - Survey responses assumed representative of actual spending
   - Passenger volumes assumed stable over time

5. COMPETITIVE LANDSCAPE:
   - No competition factored into revenue projections
   - Brand recognition assumed consistent across regions
   - Market saturation not considered

CONFIDENCE LEVELS:
- High: Relative ranking of airports (top 3 vs bottom 3)
- Moderate: Absolute profit figures (±15-20% variance expected)
- Low: Year-over-year growth rates (insufficient temporal data)

RECOMMENDATIONS:
- Run 6-month pilot at top-ranked airport
- Validate cost assumptions with actual operations data
- Adjust gross margin assumptions based on local sourcing costs
- Monitor seasonality vs December baseline
- Conduct competitive analysis at target airports
"""

print("✓ Configuration loaded successfully!")
print("\n📊 Business Assumptions:")
print(f"  Gross Margin:    {GROSS_MARGIN:.0%}")
print(f"  Staff Cost:      €{STAFF_COST_PER_SQM}/sqm/month")
print(f"  Overhead:        {OVERHEAD_PCT:.0%} of revenue")
print(f"  Category 4:      €{CATEGORY_MIDPOINTS[4]} (true midpoint)")
