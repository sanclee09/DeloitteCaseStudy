import pickle
from scipy.stats import randint
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import warnings

warnings.filterwarnings("ignore")

# Import from our modules
from config import *
from utils import *
from xgboost import XGBClassifier
from scipy.stats import randint, uniform


from config import *
from utils import *


# ============================================================================
# XGBOOST HYPERPARAMETER TUNING
# ============================================================================


def tune_xgboost_hyperparameters(X_train, y_train, n_iter=30, cv=5):
    """
    Tune XGBoost hyperparameters using Randomized Search CV

    Key parameters:
    - n_estimators: Number of boosting rounds
    - max_depth: Maximum tree depth
    - learning_rate: Step size shrinkage (eta)
    - subsample: Fraction of samples for each tree
    - colsample_bytree: Fraction of features for each tree

    Args:
        X_train: Training features
        y_train: Training labels
        n_iter: Number of combinations to try
        cv: Number of CV folds

    Returns:
        best_model: Tuned XGBoost model
        best_params: Best parameters found
        cv_results: DataFrame with all results
    """
    print_section_header("XGBOOST HYPERPARAMETER TUNING")

    # Calculate scale_pos_weight for imbalanced classes
    from collections import Counter

    class_counts = Counter(y_train)
    n_samples = len(y_train)

    # For multiclass, we'll use balanced weights
    print("\nClass Distribution:")
    for cls in sorted(class_counts.keys()):
        pct = class_counts[cls] / n_samples * 100
        print(f"  Class {cls}: {class_counts[cls]:,} ({pct:.2f}%)")

    # Parameter distributions
    param_distributions = {
        "n_estimators": randint(100, 500),  # Number of boosting rounds
        "max_depth": randint(
            3, 10
        ),  # Tree depth (XGBoost works well with shallower trees)
        "learning_rate": uniform(0.01, 0.29),  # 0.01 to 0.3
        "subsample": uniform(0.6, 0.4),  # 0.6 to 1.0
        "colsample_bytree": uniform(0.6, 0.4),  # 0.6 to 1.0
        "gamma": uniform(0, 0.5),  # Minimum loss reduction (regularization)
        "reg_alpha": uniform(0, 1),  # L1 regularization
        "reg_lambda": uniform(0, 2),  # L2 regularization
    }

    print("\nParameter Search Space:")
    print("  n_estimators:       100-500")
    print("  max_depth:          3-10 (shallower than RF)")
    print("  learning_rate:      0.01-0.3")
    print("  subsample:          0.6-1.0")
    print("  colsample_bytree:   0.6-1.0")
    print("  gamma:              0-0.5 (regularization)")
    print("  reg_alpha:          0-1 (L1)")
    print("  reg_lambda:         0-2 (L2)")

    print(f"\nSearch Configuration:")
    print(f"  Method:         Randomized Search")
    print(f"  Iterations:     {n_iter}")
    print(f"  CV Folds:       {cv}")
    print(f"  Scoring:        F1 (weighted)")
    print(f"  Total Fits:     {n_iter * cv}")

    # Base XGBoost classifier
    base_xgb = XGBClassifier(
        objective="multi:softmax",  # Multiclass classification
        eval_metric="mlogloss",  # Metric for validation
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    print("\n" + "=" * 80)
    print("Starting hyperparameter search...")
    print("(This may take 10-15 minutes)")
    print("=" * 80)

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)

    # Results
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)

    print("\n✓ XGBoost hyperparameter tuning complete!")

    print(f"\nOptimal Parameters (determined by CV):")
    for param, value in sorted(random_search.best_params_.items()):
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.4f}")
        else:
            print(f"  {param:20s}: {value}")

    print(f"\nBest Cross-Validation Score:")
    print(f"  F1 (weighted):      {random_search.best_score_:.4f}")

    # Top 5 combinations
    print("\nTop 5 Parameter Combinations:")
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    top_5 = cv_results_df.nlargest(5, "mean_test_score")[
        ["mean_test_score", "std_test_score", "rank_test_score"]
    ]

    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(
            f"  Rank {int(row['rank_test_score'])}: "
            f"F1 = {row['mean_test_score']:.4f} "
            f"(+/- {row['std_test_score']:.4f})"
        )

    return random_search.best_estimator_, random_search.best_params_, cv_results_df


# ============================================================================
# TRAIN XGBOOST MODEL
# ============================================================================


def train_xgboost(df_eu, feature_cols=None, enable_tuning=True):
    """
    Train XGBoost classifier with optional hyperparameter tuning

    Args:
        df_eu: Training dataframe
        feature_cols: Features to use
        enable_tuning: Whether to tune hyperparameters

    Returns:
        model: Trained XGBoost model
        X_train, X_test, y_train, y_test: Split datasets
        feature_cols: List of features used
    """
    print_section_header("XGBOOST TRAINING")

    # Use provided features or default
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    feature_cols = [f for f in feature_cols if f in df_eu.columns]

    print(f"\nUsing {len(feature_cols)} features:")
    print(feature_cols)

    # Prepare data
    X = df_eu[feature_cols].fillna(0)
    y = df_eu["amount_spent_cat"]

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nDataset split (stratified):")
    print(
        f"  Training set: {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)"
    )
    print(
        f"  Test set:     {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)"
    )

    # Train model
    if enable_tuning:
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING ENABLED")
        print("=" * 80)

        xgb_model, best_params, cv_results = tune_xgboost_hyperparameters(
            X_train, y_train, n_iter=30, cv=5
        )

        print("\n✓ Using tuned XGBoost model")

    else:
        print("\n" + "=" * 80)
        print("TRAINING WITH DEFAULT XGBOOST PARAMETERS")
        print("=" * 80)

        # Default XGBoost parameters (reasonable baseline)
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        print("\nDefault parameters:")
        for param, value in default_params.items():
            print(f"  {param}: {value}")

        xgb_model = XGBClassifier(**default_params)
        xgb_model.fit(X_train, y_train)

    print("\n✓ XGBoost training complete")

    return xgb_model, X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# EVALUATE XGBOOST (Same as RF, but included for completeness)
# ============================================================================


def evaluate_xgboost(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive XGBoost model evaluation
    """
    print_section_header("XGBOOST MODEL EVALUATION")

    # 1. Accuracy scores
    print_subsection_header("1. Accuracy Scores")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training Accuracy:   {train_score:.4f} ({train_score * 100:.2f}%)")
    print(f"Test Accuracy:       {test_score:.4f} ({test_score * 100:.2f}%)")
    print(f"Overfitting Check:   {(train_score - test_score):.4f}")

    if (train_score - test_score) < 0.05:
        print("  ✓ Good generalization")
    elif (train_score - test_score) < 0.10:
        print("  ⚠ Moderate overfitting")
    else:
        print("  ✗ High overfitting")

    # 2. Cross-validation
    print_subsection_header("2. Cross-Validation")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"F1 Score (weighted):  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 3. Predictions
    y_pred = model.predict(X_test)

    # 4. Classification report
    print_subsection_header("3. Classification Report")
    print(
        classification_report(
            y_test, y_pred, target_names=[f"Category {i}" for i in range(5)]
        )
    )

    # 5. Confusion matrix
    print_subsection_header("4. Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    print("\n(Rows = Actual, Columns = Predicted)")
    print(cm)

    print("\nPer-Class Accuracy:")
    for i in range(len(cm)):
        class_acc = cm[i, i] / cm[i].sum()
        print(f"  Category {i}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})")

    # 6. Overall metrics
    print_subsection_header("5. Overall Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:         {accuracy:.4f}")
    print(f"F1 (macro):       {f1_macro:.4f}")
    print(f"F1 (weighted):    {f1_weighted:.4f}")

    metrics = {
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }

    return metrics, y_pred


# ============================================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================================


def analyze_xgboost_feature_importance(model, feature_cols):
    """
    Analyze XGBoost feature importance
    XGBoost provides multiple importance types: weight, gain, cover
    """
    print_section_header("XGBOOST FEATURE IMPORTANCE")

    # Get feature importance directly from the model
    # This returns importance as an array
    importance_gain = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance_gain}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance (by Gain):")
    print(importance_df.to_string(index=False))

    print("\n⭐ Top 5 Features:")
    for idx, row in importance_df.head(5).iterrows():
        pct = row["importance"] / importance_df["importance"].sum() * 100
        print(f"  {row['feature']:20s}: {row['importance']:8.4f} ({pct:.1f}%)")

    print("\nNote: XGBoost importance represents average gain when feature is used")

    return importance_df


# ============================================================================
# MODEL COMPARISON
# ============================================================================


def compare_models(rf_metrics, xgb_metrics):
    """
    Compare Random Forest and XGBoost performance
    """
    print_section_header("MODEL COMPARISON: Random Forest vs XGBoost")

    print("\n                      Random Forest    XGBoost      Difference")
    print("-" * 70)

    metrics_to_compare = [
        ("Test Accuracy", "test_accuracy"),
        ("F1 (weighted)", "f1_weighted"),
        ("F1 (macro)", "f1_macro"),
        ("CV F1 Mean", "cv_f1_mean"),
    ]

    for name, key in metrics_to_compare:
        rf_val = rf_metrics[key]
        xgb_val = xgb_metrics[key]
        diff = xgb_val - rf_val

        winner = "🏆 XGBoost" if diff > 0.001 else ("🏆 RF" if diff < -0.001 else "Tie")

        print(
            f"{name:20s}  {rf_val:.4f}        {xgb_val:.4f}       "
            f"{diff:+.4f}  {winner}"
        )

    # Overall recommendation
    print("\n" + "=" * 70)
    if xgb_metrics["test_accuracy"] > rf_metrics["test_accuracy"] + 0.01:
        print("✓ RECOMMENDATION: Use XGBoost (significantly better performance)")
    elif xgb_metrics["test_accuracy"] > rf_metrics["test_accuracy"]:
        print("✓ RECOMMENDATION: Use XGBoost (slightly better performance)")
    elif rf_metrics["test_accuracy"] > xgb_metrics["test_accuracy"] + 0.01:
        print("✓ RECOMMENDATION: Use Random Forest (significantly better performance)")
    else:
        print("✓ RECOMMENDATION: Similar performance - choose based on:")
        print("  - XGBoost: Faster inference, better for production")
        print("  - Random Forest: More interpretable, less tuning needed")


# ============================================================================
# DATA LOADING
# ============================================================================


def load_preprocessed_data():
    """
    Load preprocessed data from previous step
    """
    print_section_header("LOADING PREPROCESSED DATA")

    df_eu = load_csv_with_info(EU_CLEAN_FILE, "EU Passengers (Clean)")
    df_ww = load_csv_with_info(WW_CLEAN_FILE, "WW Passengers (Clean)")

    # Validate required columns
    if validate_dataframe(df_eu, FEATURE_COLUMNS + ["amount_spent_cat"], "EU Dataset"):
        print("✓ EU dataset ready for modeling")

    if validate_dataframe(df_ww, FEATURE_COLUMNS, "WW Dataset"):
        print("✓ WW dataset ready for prediction")

    return df_eu, df_ww


# ============================================================================
# FEATURE SELECTION WITH L1 REGULARIZATION
# ============================================================================


def perform_l1_feature_selection(X, y):
    """
    Use L1 (Lasso) regularization for feature selection
    """
    print_section_header("FEATURE SELECTION: L1 REGULARIZATION")

    # Standardize features (required for L1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nTesting multiple regularization strengths...")
    C_values = [0.001, 0.01, 0.1, 1, 10]

    for C in C_values:
        lr = LogisticRegression(
            penalty="l1",
            C=C,
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        lr.fit(X_scaled, y)

        # Get feature coefficients
        coef_avg = np.abs(lr.coef_).mean(axis=0)
        non_zero = (coef_avg > 0.001).sum()

        print(f"  C={C:6.3f}: {non_zero}/{len(X.columns)} features selected")

    # Use C=1 for balanced regularization
    print("\nFeature importance (L1, C=1):")
    lr = LogisticRegression(
        penalty="l1", C=1, solver="liblinear", max_iter=1000, random_state=RANDOM_STATE
    )
    lr.fit(X_scaled, y)

    coef_avg = np.abs(lr.coef_).mean(axis=0)
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": coef_avg}
    ).sort_values("importance", ascending=False)

    print(feature_importance.to_string(index=False))

    # Select important features
    important_features = feature_importance[feature_importance["importance"] > 0.01][
        "feature"
    ].tolist()
    print(f"\n✓ Selected {len(important_features)} important features:")
    print(important_features)

    return feature_importance, important_features


# ============================================================================
# HYPERPARAMETER TUNING (SIMPLIFIED)
# ============================================================================


def tune_hyperparameters_simplified(X_train, y_train, n_iter=30, cv=5):
    """
    Perform hyperparameter tuning on 3 key parameters using Randomized Search CV

    The optimal values are determined through cross-validation:
    - Each parameter combination is tested with 5-fold CV
    - The combination with the best mean CV score is selected

    Args:
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter combinations to try (default: 30)
        cv: Number of cross-validation folds (default: 5)

    Returns:
        best_model: Tuned Random Forest model with optimal parameters
        best_params: Dictionary of best parameters found
        cv_results: DataFrame with all tried combinations
    """
    print_section_header("HYPERPARAMETER TUNING: RANDOMIZED SEARCH CV")

    # Define parameter distributions (ONLY 3 parameters)
    param_distributions = {
        "n_estimators": randint(100, 500),  # Number of trees: 100-500
        "max_depth": [10, 15, 20, 25, 30, None],  # Tree depth: limited or unlimited
        "min_samples_split": randint(20, 100),  # Min samples to split: 20-100
    }

    print("\nTuning Strategy:")
    print("  Parameters to tune: 3 (n_estimators, max_depth, min_samples_split)")
    print("  Fixed parameters:   min_samples_leaf=20, class_weight='balanced'")
    print()
    print("  Parameter Search Space:")
    print("    n_estimators:       100-500 (random integers)")
    print("    max_depth:          [10, 15, 20, 25, 30, None]")
    print("    min_samples_split:  20-100 (random integers)")
    print()
    print(f"  Search Strategy:")
    print(f"    Method:         Randomized Search")
    print(f"    Iterations:     {n_iter} random combinations")
    print(f"    CV Folds:       {cv}-fold cross-validation")
    print(f"    Scoring:        F1 (weighted)")
    print(f"    Total Fits:     {n_iter * cv} model fits")
    print()
    print("  How optimal parameters are determined:")
    print("    1. Each parameter combination is evaluated with 5-fold CV")
    print("    2. Mean F1 score across 5 folds is calculated")
    print("    3. Combination with highest mean CV score is selected as optimal")

    # Base estimator with FIXED parameters
    base_rf = RandomForestClassifier(
        min_samples_leaf=20,  # FIXED
        class_weight="balanced",  # FIXED
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    # Randomized search with cross-validation
    print("\n" + "=" * 80)
    print("Starting hyperparameter search...")
    print("(This may take 8-12 minutes)")
    print("=" * 80)

    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)

    # Extract results
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)

    print("\n✓ Hyperparameter tuning complete!")

    print(f"\nOptimal Parameters (determined by cross-validation):")
    print(f"  n_estimators:       {random_search.best_params_['n_estimators']}")
    print(f"  max_depth:          {random_search.best_params_['max_depth']}")
    print(f"  min_samples_split:  {random_search.best_params_['min_samples_split']}")

    print(f"\nFixed Parameters:")
    print(f"  min_samples_leaf:   20")
    print(f"  class_weight:       'balanced'")

    print(f"\nBest Cross-Validation Score:")
    print(f"  F1 (weighted):      {random_search.best_score_:.4f}")

    # Show top 5 combinations
    print("\nTop 5 Parameter Combinations (ranked by CV score):")
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    top_5 = cv_results_df.nlargest(5, "mean_test_score")[
        [
            "mean_test_score",
            "std_test_score",
            "param_n_estimators",
            "param_max_depth",
            "param_min_samples_split",
        ]
    ]

    print("\n  Rank  |  CV F1  |  Std  | n_estimators | max_depth | min_samples_split")
    print("  " + "-" * 76)
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(
            f"    {idx}   | {row['mean_test_score']:.4f} | {row['std_test_score']:.4f} |"
            f"     {row['param_n_estimators']:3d}      |   {str(row['param_max_depth']):>4s}  |"
            f"        {row['param_min_samples_split']:3d}"
        )

    # Compare to baseline
    print(f"\nComparison to Manual Parameters:")
    print(f"  Manual params:     n_estimators=200, max_depth=15, min_samples_split=50")
    print(
        f"  Tuned params:      n_estimators={random_search.best_params_['n_estimators']}, "
        f"max_depth={random_search.best_params_['max_depth']}, "
        f"min_samples_split={random_search.best_params_['min_samples_split']}"
    )

    return random_search.best_estimator_, random_search.best_params_, cv_results_df


# ============================================================================
# MODIFIED TRAINING FUNCTION
# ============================================================================


def train_random_forest(df_eu, feature_cols=None, enable_tuning=True):
    """
    Train Random Forest classifier with optional hyperparameter tuning

    Args:
        df_eu: Training dataframe
        feature_cols: Features to use (default: from config)
        enable_tuning: Whether to tune hyperparameters (default: True)

    Returns:
        model: Trained Random Forest model
        X_train, X_test, y_train, y_test: Split datasets
        feature_cols: List of features used
    """
    print_section_header("RANDOM FOREST TRAINING")

    # Use provided features or default from config
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    feature_cols = [f for f in feature_cols if f in df_eu.columns]

    print(f"\nUsing {len(feature_cols)} features:")
    print(feature_cols)

    # Prepare data
    X = df_eu[feature_cols].fillna(0)
    y = df_eu["amount_spent_cat"]

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Maintains class distribution
    )

    print(f"\nDataset split (stratified):")
    print(
        f"  Training set: {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)"
    )
    print(
        f"  Test set:     {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)"
    )

    # Verify stratification worked
    print(f"\nClass distribution verification:")
    print(
        f"  Original:  {(pd.Series(y).value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}"
    )
    print(
        f"  Training:  {(pd.Series(y_train).value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}"
    )
    print(
        f"  Test:      {(pd.Series(y_test).value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}"
    )

    # Train model
    if enable_tuning:
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING ENABLED")
        print("=" * 80)

        rf_model, best_params, cv_results = tune_hyperparameters_simplified(
            X_train, y_train, n_iter=30, cv=5
        )

        print("\n✓ Using tuned model for final evaluation")

    else:
        print("\n" + "=" * 80)
        print("TRAINING WITH MANUAL PARAMETERS (NO TUNING)")
        print("=" * 80)
        print("\nNote: Set enable_tuning=True to perform hyperparameter search")

        print("\nTraining Random Forest Classifier...")
        print("Hyperparameters:")
        for param, value in RF_PARAMS.items():
            print(f"  {param}: {value}")

        rf_model = RandomForestClassifier(**RF_PARAMS)
        rf_model.fit(X_train, y_train)

    print("\n✓ Model training complete")

    return rf_model, X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model evaluation
    """
    print_section_header("MODEL EVALUATION")

    # 1. Training and test accuracy
    print_subsection_header("1. Accuracy Scores")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training Accuracy:   {train_score:.4f} ({train_score * 100:.2f}%)")
    print(f"Test Accuracy:       {test_score:.4f} ({test_score * 100:.2f}%)")
    print(f"Overfitting Check:   {(train_score - test_score):.4f}")

    if (train_score - test_score) < 0.05:
        print("  ✓ Good generalization (low overfitting)")
    elif (train_score - test_score) < 0.10:
        print("  ⚠ Moderate overfitting")
    else:
        print("  ✗ High overfitting - consider regularization")

    # 2. Cross-validation
    print_subsection_header("2. Cross-Validation (5-Fold)")
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring=CV_SCORING
    )

    print(f"F1 Score (weighted):  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # 3. Predictions
    y_pred = model.predict(X_test)

    # 4. Classification report
    print_subsection_header("3. Classification Report")
    print(
        classification_report(
            y_test, y_pred, target_names=[f"Category {i}" for i in range(5)]
        )
    )

    # 5. Confusion matrix
    print_subsection_header("4. Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("(Rows = Actual, Columns = Predicted)")
    print(cm)

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(len(cm)):
        class_acc = cm[i, i] / cm[i].sum()
        print(f"  Category {i}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})")

    # 6. Overall metrics
    print_subsection_header("5. Overall Performance Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:         {accuracy:.4f}")
    print(f"F1 (macro):       {f1_macro:.4f}")
    print(f"F1 (weighted):    {f1_weighted:.4f}")

    # Model performance summary
    metrics = {
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }

    return metrics, y_pred


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================


def analyze_feature_importance(model, feature_cols):
    """
    Analyze and visualize feature importance
    """
    print_section_header("FEATURE IMPORTANCE ANALYSIS")

    # Get feature importance from Random Forest
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance Ranking:")
    print(importance_df.to_string(index=False))

    # Identify top features
    top_5 = importance_df.head(5)
    print(f"\n⭐ Top 5 Most Important Features:")
    for idx, row in top_5.iterrows():
        print(
            f"  {row['feature']:20s}: {row['importance']:.4f} ({row['importance'] * 100:.2f}%)"
        )

    # Check cumulative importance
    importance_df["cumulative"] = importance_df["importance"].cumsum()
    n_for_80 = (importance_df["cumulative"] >= 0.80).idxmax() + 1
    print(f"\n✓ Top {n_for_80} features explain 80% of model decisions")

    return importance_df


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================


def save_model(model, feature_cols, metrics):
    """
    Save trained model and metadata
    """
    print_section_header("SAVING MODEL")

    model_data = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_params": RF_PARAMS,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to: {MODEL_FILE}")
        print(f"  Model type: Random Forest")
        print(f"  Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Features: {len(feature_cols)}")
        return True
    except Exception as e:
        print(f"✗ Error saving model: {str(e)}")
        return False


def load_model():
    """
    Load trained model
    """
    try:
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded from: {MODEL_FILE}")
        return model_data
    except FileNotFoundError:
        print(f"✗ Model file not found: {MODEL_FILE}")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================


def main():
    """
    Main model training pipeline
    """
    print("=" * 80)
    print("DELOITTE CASE STUDY - MODEL TRAINING")
    print("=" * 80)

    # Step 1: Load preprocessed data
    print("\n[1/6] Loading preprocessed data...")
    df_eu, df_ww = load_preprocessed_data()

    # Step 2: Feature selection (optional - for analysis)
    print("\n[2/6] Performing feature selection analysis...")
    X_for_selection = df_eu[FEATURE_COLUMNS].fillna(0)
    y_for_selection = df_eu["amount_spent_cat"]
    l1_importance, l1_features = perform_l1_feature_selection(
        X_for_selection, y_for_selection
    )

    # Step 3: Train Random Forest with hyperparameter tuning
    print("\n[3/6] Training predictive model...")

    # Set to True to enable tuning, False to use manual params
    ENABLE_TUNING = True

    model, X_train, X_test, y_train, y_test, feature_cols = train_random_forest(
        df_eu, enable_tuning=ENABLE_TUNING
    )

    # Step 4: Evaluate model
    print("\n[4/6] Evaluating model performance...")
    metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Step 5: Analyze feature importance
    print("\n[5/6] Analyzing feature importance...")
    importance_df = analyze_feature_importance(model, feature_cols)

    # Step 6: Save model
    print("\n[6/6] Saving trained model...")
    save_success = save_model(model, feature_cols, metrics)

    # Summary
    print_section_header("TRAINING COMPLETE")
    print(f"✓ Model: Random Forest Classifier")
    if ENABLE_TUNING:
        print(f"✓ Hyperparameters: Tuned via Randomized Search CV (3 parameters)")
    else:
        print(f"✓ Hyperparameters: Manual configuration")
    print(
        f"✓ Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
    )
    print(f"✓ F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    print(
        f"✓ CV F1 Score: {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})"
    )
    print(f"✓ Features used: {len(feature_cols)}")
    print(f"✓ Model saved: {save_success}")

    # Key insights
    print("\n⭐ KEY INSIGHTS:")
    top_3_features = importance_df.head(3)["feature"].tolist()
    print(f"  Top 3 predictive features: {', '.join(top_3_features)}")

    if metrics["test_accuracy"] > 0.75:
        print("  Model performance: GOOD (accuracy > 75%)")
    elif metrics["test_accuracy"] > 0.70:
        print("  Model performance: ACCEPTABLE (accuracy > 70%)")
    else:
        print("  Model performance: NEEDS IMPROVEMENT (accuracy < 70%)")

    if ENABLE_TUNING:
        print(f"  Hyperparameter tuning: COMPLETED")
        print(f"    Tuned: n_estimators, max_depth, min_samples_split")
        print(f"    Fixed: min_samples_leaf=20, class_weight='balanced'")
    else:
        print("  Hyperparameter tuning: SKIPPED")

    print("\nNext step: Run 3_prediction_analysis.py")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance_df,
        "l1_features": l1_features,
        "feature_cols": feature_cols,
        "tuning_enabled": ENABLE_TUNING,
    }


# ============================================================================
# MAIN FUNCTION FOR XGBOOST
# ============================================================================


def main_xgboost():
    """
    Main pipeline for XGBoost training
    """
    print("=" * 80)
    print("DELOITTE CASE STUDY - XGBOOST MODEL TRAINING")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading preprocessed data...")
    from utils import load_csv_with_info

    df_eu = load_csv_with_info(EU_CLEAN_FILE, "EU Passengers (Clean)")

    # Train XGBoost
    print("\n[2/5] Training XGBoost model...")
    ENABLE_TUNING = True

    model, X_train, X_test, y_train, y_test, feature_cols = train_xgboost(
        df_eu, enable_tuning=ENABLE_TUNING
    )

    # Evaluate
    print("\n[3/5] Evaluating XGBoost performance...")
    metrics, y_pred = evaluate_xgboost(model, X_train, X_test, y_train, y_test)

    # Feature importance
    print("\n[4/5] Analyzing feature importance...")
    importance_df = analyze_xgboost_feature_importance(model, feature_cols)

    # Save model
    print("\n[5/5] Saving model...")
    model_data = {
        "model": model,
        "model_type": "XGBoost",
        "feature_cols": feature_cols,
        "metrics": metrics,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    import pickle

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    print(f"✓ XGBoost model saved to: {MODEL_FILE}")

    # Summary
    print_section_header("XGBOOST TRAINING COMPLETE")
    print(f"✓ Model: XGBoost Classifier")
    print(
        f"✓ Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
    )
    print(f"✓ F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    print(
        f"✓ CV F1 Score: {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})"
    )

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance_df,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    results = main_xgboost()
    print("\n✓ XGBoost training complete!")
