"""
Model Training - Fixed SMOTE Implementation
Clean, simple, working version
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import randint, uniform
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

from config import *
from utils import *


# ============================================================================
# XGBOOST WITH SMOTE (PROPER IMPLEMENTATION)
# ============================================================================


def train_xgboost_with_smote(
    df_eu, feature_cols=None, enable_smote=True, enable_tuning=True
):
    """
    Train XGBoost with SMOTE properly integrated into CV pipeline

    Args:
        df_eu: Training dataframe
        feature_cols: Features to use
        enable_smote: Whether to use SMOTE (default: True)
        enable_tuning: Whether to tune hyperparameters (default: True)

    Returns:
        pipeline: Trained pipeline (SMOTE + XGBoost)
        X_test, y_test: Test data
        metrics: Performance metrics
    """
    print_section_header("XGBOOST TRAINING WITH SMOTE")

    # Features
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    feature_cols = [f for f in feature_cols if f in df_eu.columns]

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"SMOTE: {'Enabled' if enable_smote else 'Disabled'}")
    print(f"Tuning: {'Enabled' if enable_tuning else 'Disabled'}")

    # Prepare data
    X = df_eu[feature_cols].fillna(0).values
    y = df_eu["amount_spent_cat"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # Class distribution
    print("\nClass Distribution (Train):")
    for cls in sorted(np.unique(y_train)):
        count = np.sum(y_train == cls)
        pct = count / len(y_train) * 100
        print(f"  Class {cls}: {count:,} ({pct:.1f}%)")

    # ========================================================================
    # Build Pipeline (SMOTE + XGBoost)
    # ========================================================================

    if enable_smote:
        pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
                (
                    "xgb",
                    XGBClassifier(
                        objective="multi:softprob",  # Use probabilities
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                (
                    "xgb",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                )
            ]
        )

    # ========================================================================
    # Train (with or without tuning)
    # ========================================================================

    if enable_tuning:
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING (WITH SMOTE IN PIPELINE)")
        print("=" * 80)

        param_distributions = {
            "xgb__n_estimators": randint(200, 600),
            "xgb__max_depth": randint(3, 10),
            "xgb__learning_rate": uniform(0.01, 0.29),
            "xgb__subsample": uniform(0.6, 0.4),
            "xgb__colsample_bytree": uniform(0.6, 0.4),
            "xgb__gamma": uniform(0, 0.5),
            "xgb__reg_alpha": uniform(0, 1),
            "xgb__reg_lambda": uniform(0, 2),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=30,
            cv=cv,
            scoring="f1_macro",  # Use macro to see minority class gains
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )

        print("\nStarting search (this may take 10-15 minutes)...")
        search.fit(X_train, y_train)

        pipeline = search.best_estimator_
        best_score = search.best_score_

        print(f"\n✓ Best CV F1-macro: {best_score:.4f}")
        print("\nBest Parameters:")
        for param, value in search.best_params_.items():
            param_name = param.replace("xgb__", "")
            if isinstance(value, float):
                print(f"  {param_name:20s}: {value:.4f}")
            else:
                print(f"  {param_name:20s}: {value}")

    else:
        print("\n" + "=" * 80)
        print("TRAINING WITH DEFAULT PARAMETERS")
        print("=" * 80)

        # Set reasonable defaults
        pipeline.set_params(
            xgb__n_estimators=300,
            xgb__max_depth=6,
            xgb__learning_rate=0.1,
            xgb__subsample=0.8,
            xgb__colsample_bytree=0.8,
        )

        pipeline.fit(X_train, y_train)

    # ========================================================================
    # Evaluate
    # ========================================================================

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1_macro = f1_score(y_test, y_pred_test, average="macro")
    f1_weighted = f1_score(y_test, y_pred_test, average="weighted")

    print(f"\nTrain Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"F1 Macro:        {f1_macro:.4f}")
    print(f"F1 Weighted:     {f1_weighted:.4f}")
    print(f"Overfitting:     {train_acc - test_acc:.4f}")

    if train_acc - test_acc < 0.05:
        print("  ✓ Good generalization")
    elif train_acc - test_acc < 0.10:
        print("  ⚠ Moderate overfitting")
    else:
        print("  ✗ High overfitting")

    # Per-class metrics
    print("\n" + "-" * 80)
    print("PER-CLASS PERFORMANCE")
    print("-" * 80)

    report = classification_report(
        y_test, y_pred_test, output_dict=True, zero_division=0
    )

    print("\nClass  Precision  Recall    F1-Score  Support")
    print("-" * 50)
    for cls in sorted([k for k in report.keys() if k.isdigit()]):
        metrics_cls = report[cls]
        support = metrics_cls["support"]
        print(
            f"{cls:5s}  {metrics_cls['precision']:.4f}    {metrics_cls['recall']:.4f}    "
            f"{metrics_cls['f1-score']:.4f}    {int(support):,}"
        )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print("(Rows=Actual, Cols=Predicted)")
    print(cm)

    # Store metrics
    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return pipeline, X_test, y_test, feature_cols, metrics


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================


def analyze_feature_importance(pipeline, feature_cols):
    """Extract feature importance from XGBoost in pipeline"""
    print_section_header("FEATURE IMPORTANCE")

    # Get XGBoost model from pipeline
    xgb_model = pipeline.named_steps["xgb"]

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        pct = row["importance"] / importance_df["importance"].sum() * 100
        print(f"  {row['feature']:25s}: {row['importance']:.4f} ({pct:.1f}%)")

    return importance_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main_xgboost():
    """Main training pipeline"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - XGBOOST TRAINING")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    df_eu = load_csv_with_info(EU_CLEAN_FILE, "EU Passengers (Clean)")

    # Train
    print("\n[2/4] Training model...")
    pipeline, X_test, y_test, feature_cols, metrics = train_xgboost_with_smote(
        df_eu,
        enable_smote=True,  # SMOTE enabled
        enable_tuning=True,  # Hyperparameter tuning enabled
    )

    # Feature importance
    print("\n[3/4] Analyzing features...")
    importance_df = analyze_feature_importance(pipeline, feature_cols)

    # Save
    print("\n[4/4] Saving model...")
    model_data = {
        "model": pipeline,  # Changed from 'pipeline' to 'model' for compatibility
        "model_type": "XGBoost with SMOTE",
        "feature_cols": feature_cols,
        "metrics": metrics,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✓ Saved to: {MODEL_FILE}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✓ Model Type:        XGBoost with SMOTE Pipeline")
    print(f"✓ Test Accuracy:     {metrics['test_accuracy']:.4f}")
    print(f"✓ F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"✓ F1 Weighted:       {metrics['f1_weighted']:.4f}")

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importance": importance_df,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    results = main_xgboost()
    print("\n✓ Done!")
