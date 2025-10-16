import pickle
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import *
from utils import *


# ============================================================================
# FEATURE SELECTION - NO LEAKAGE
# ============================================================================


class FeatureSelector:
    """
    Feature selector that prevents data leakage by fitting only on training data
    """

    def __init__(self, variance_threshold=0.005, correlation_threshold=0.98):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_selector = None
        self.selected_features = None
        self.correlation_removals = []

    def fit(self, X, y, feature_names):
        """Fit feature selection on training data only"""
        print_subsection_header("Feature Selection (Training Data Only)")

        X_df = pd.DataFrame(X, columns=feature_names)

        # 1. Variance threshold
        print("\n1. Variance-Based Selection")
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        self.variance_selector.fit(X_df)

        variance_mask = self.variance_selector.get_support()
        features_after_variance = [f for f, m in zip(feature_names, variance_mask) if m]

        print(f"  Features kept: {len(features_after_variance)}/{len(feature_names)}")
        removed = [f for f, m in zip(feature_names, variance_mask) if not m]
        if removed:
            print(f"  Removed {len(removed)} low-variance features:")
            for f in removed[:5]:
                print(f"    - {f}")

        # 2. Correlation threshold
        print("\n2. Correlation-Based Selection")
        X_var_filtered = pd.DataFrame(
            self.variance_selector.transform(X_df), columns=features_after_variance
        )

        # Add target for correlation calculation
        X_with_target = X_var_filtered.copy()
        X_with_target["_target"] = y

        corr_matrix = X_var_filtered.corr().abs()
        target_corr = X_with_target.corr()["_target"].abs()

        # Find correlated pairs
        features_to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]

                    # Keep the one with higher target correlation
                    if target_corr[feat_i] >= target_corr[feat_j]:
                        features_to_remove.add(feat_j)
                        self.correlation_removals.append(
                            (feat_i, feat_j, corr_matrix.iloc[i, j])
                        )
                    else:
                        features_to_remove.add(feat_i)
                        self.correlation_removals.append(
                            (feat_j, feat_i, corr_matrix.iloc[i, j])
                        )

        self.selected_features = [
            f for f in features_after_variance if f not in features_to_remove
        ]

        print(
            f"  Features kept: {len(self.selected_features)}/{len(features_after_variance)}"
        )
        if features_to_remove:
            print(f"  Removed {len(features_to_remove)} correlated features:")
            for kept, removed, corr in self.correlation_removals[:5]:
                print(f"    - {removed} (corr with {kept}: {corr:.3f})")

        print(
            f"\n✓ Final feature count: {len(self.selected_features)}/{len(feature_names)}"
        )
        return self

    def transform(self, X, feature_names):
        """Transform features using fitted selector"""
        X_df = pd.DataFrame(X, columns=feature_names)
        X_var = self.variance_selector.transform(X_df)

        # Get features after variance filtering
        variance_mask = self.variance_selector.get_support()
        features_after_variance = [f for f, m in zip(feature_names, variance_mask) if m]

        # Select final features
        X_var_df = pd.DataFrame(X_var, columns=features_after_variance)
        return X_var_df[self.selected_features].values, self.selected_features

    def fit_transform(self, X, y, feature_names):
        """Fit and transform in one step"""
        self.fit(X, y, feature_names)
        return self.transform(X, feature_names)


# ============================================================================
# MODEL TRAINING - XGBOOST WITH PROPER PIPELINE
# ============================================================================


def train_xgboost_no_leakage(df, feature_cols, enable_tuning=None):
    """Train XGBoost with proper pipeline - NO leakage, NO fillna"""
    print_section_header("XGBOOST TRAINING (NO LEAKAGE)")

    if enable_tuning is None:
        enable_tuning = ENABLE_HYPERPARAMETER_TUNING

    # Prepare data - NO FILLNA FOR XGBOOST!
    X = df[feature_cols].values  # Keep NaNs
    y = df["amount_spent_cat"].values

    # CRITICAL: Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n✓ Data split BEFORE feature selection:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\n✓ Training set class distribution:")
    for cls, count in class_dist.items():
        print(f"  Class {cls}: {count:,} samples")

    # Feature selection on TRAINING data only
    print_subsection_header("Feature Selection on Training Data")
    selector = FeatureSelector(
        variance_threshold=FEATURE_SELECTION_PARAMS["variance_threshold"],
        correlation_threshold=FEATURE_SELECTION_PARAMS["correlation_threshold"],
    )

    X_train_selected, selected_features = selector.fit_transform(
        X_train, y_train, feature_cols
    )
    X_test_selected, _ = selector.transform(X_test, feature_cols)

    print(f"\n✓ Selected features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")

    # Identify numerical vs categorical features for scaling
    numerical_features = [
        f
        for f in selected_features
        if any(
            substring in f
            for substring in [
                "_scaled",
                "age",
                "luggage",
                "flighttime",
                "traveltime",
                "layover",
            ]
        )
        and "_encoded" not in f
        and "business_longhaul" not in f
        and "age_business" not in f
        and "family_luggage" not in f
        and "layover_shopping_time" not in f
        and "male_business" not in f
    ]

    categorical_features = [f for f in selected_features if f not in numerical_features]

    print(f"\n✓ Feature types:")
    print(f"  Numerical (will scale): {len(numerical_features)}")
    print(f"  Categorical/Binary (no scaling): {len(categorical_features)}")

    # Build proper preprocessing pipeline
    # Get indices for column transformer
    numerical_indices = [selected_features.index(f) for f in numerical_features]
    categorical_indices = [selected_features.index(f) for f in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_indices),
            ("cat", "passthrough", categorical_indices),
        ],
        remainder="drop",
    )

    # Calculate intelligent SMOTE strategy
    max_samples = max(class_dist.values())
    target_samples = int(max_samples * 0.8)

    sampling_strategy = {}
    for cls, count in class_dist.items():
        if count < target_samples:
            sampling_strategy[cls] = target_samples

    print(f"\n✓ SMOTE sampling strategy:")
    print(f"  Max class size: {max_samples:,}")
    print(f"  Target size: {target_samples:,}")
    for cls, target in sampling_strategy.items():
        original = class_dist[cls]
        print(f"  Class {cls}: {original:,} → {target:,} (+{target - original:,})")

    # Build complete pipeline: Preprocessing → SMOTE → XGBoost
    # XGBoost handles NaN natively, so we DON'T fillna!
    if enable_tuning:
        pipeline = ImbPipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        sampling_strategy=sampling_strategy,
                        k_neighbors=5,
                    ),
                ),
                ("xgb", XGBClassifier(**XGBOOST_BASE_PARAMS)),
            ]
        )
    else:
        print("\n✓ Using optimal hyperparameters")
        pipeline = ImbPipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        sampling_strategy=sampling_strategy,
                        k_neighbors=5,
                    ),
                ),
                ("xgb", XGBClassifier(**OPTIMAL_XGBOOST_PARAMS)),
            ]
        )

    # Training
    if enable_tuning:
        print_subsection_header("Hyperparameter Tuning")

        param_grid = {
            "xgb__n_estimators": randint(400, 700),
            "xgb__max_depth": randint(5, 9),
            "xgb__learning_rate": uniform(0.05, 0.15),
            "xgb__subsample": uniform(0.75, 0.2),
            "xgb__colsample_bytree": uniform(0.75, 0.2),
            "xgb__gamma": uniform(0, 1.0),
            "xgb__reg_alpha": uniform(0, 1.5),
            "xgb__reg_lambda": uniform(1.0, 2.5),
        }

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=30,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
        )

        print("\nStarting RandomizedSearchCV...")
        search.fit(X_train_selected, y_train)

        pipeline = search.best_estimator_
        best_cv_score = search.best_score_
        print(f"\n✓ Best CV {CV_SCORING}: {best_cv_score:.4f}")

        print("\n✓ Best parameters found:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
    else:
        print_subsection_header("Training with Optimal Parameters")
        pipeline.fit(X_train_selected, y_train)
        best_cv_score = None

    # Evaluation
    metrics = evaluate_model(
        pipeline,
        X_train_selected,
        X_test_selected,
        y_train,
        y_test,
        "XGBoost",
        selected_features,
    )
    metrics["best_cv_score"] = best_cv_score
    metrics["tuning_enabled"] = enable_tuning

    # Save preprocessing info for prediction
    preprocessing_info = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "numerical_indices": numerical_indices,
        "categorical_indices": categorical_indices,
    }

    return pipeline, metrics, selected_features, selector, preprocessing_info


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(
    pipeline, X_train, X_test, y_train, y_test, model_name, feature_names=None
):
    """Evaluate model performance"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    print_subsection_header(f"{model_name} Evaluation")

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1_macro = f1_score(y_test, y_pred_test, average="macro")
    f1_weighted = f1_score(y_test, y_pred_test, average="weighted")

    print(f"\nOverall Performance:")
    print(f"  Train Accuracy:  {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"  Test Accuracy:   {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  F1 Macro:        {f1_macro:.4f}")
    print(f"  F1 Weighted:     {f1_weighted:.4f}")
    print(f"  Overfitting:     {train_acc - test_acc:.4f}")

    if train_acc - test_acc < 0.05:
        print("  ✓ Excellent generalization")
    elif train_acc - test_acc < 0.10:
        print("  ⚠ Moderate overfitting")
    else:
        print("  ✗ High overfitting - consider regularization")

    # Cross-validation (on training data only!)
    print("\nCross-Validation Performance (Training Data):")
    # Note: We can't easily CV with our pre-selected features, so we skip this
    # or we'd need to wrap the selector in the pipeline too
    print("  (Skipped - feature selection done outside CV)")

    # Per-class performance
    print("\nPer-Class Performance:")
    report = classification_report(
        y_test, y_pred_test, output_dict=True, zero_division=0
    )

    print("\nClass  Precision  Recall    F1-Score  Support")
    print("-" * 50)
    for cls in sorted([k for k in report.keys() if k.isdigit()]):
        metrics_cls = report[cls]
        support = int(metrics_cls["support"])
        print(
            f"{cls:5s}  {metrics_cls['precision']:.4f}    {metrics_cls['recall']:.4f}    "
            f"{metrics_cls['f1-score']:.4f}    {support:,}"
        )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print("\nConfusion Matrix (Raw Counts):")
    print(cm)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        ax=ax1,
        xticklabels=range(5),
        yticklabels=range(5),
    )
    ax1.set_title(f"{model_name} - Raw Confusion Matrix", fontweight="bold")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax2,
        xticklabels=range(5),
        yticklabels=range(5),
    )
    ax2.set_title(f"{model_name} - Normalized Confusion Matrix", fontweight="bold")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()

    cm_filename = os.path.join(
        OUTPUT_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.savefig(cm_filename, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"\n✓ Confusion matrix saved to: {cm_filename}")
    plt.close()

    # Feature importance
    if feature_names and hasattr(pipeline.named_steps["xgb"], "feature_importances_"):
        print_subsection_header("Feature Importance")
        model = pipeline.named_steps["xgb"]
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "classification_report": report,
    }

    return metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main training pipeline with NO data leakage"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - MODEL TRAINING (NO LEAKAGE)")
    print("=" * 80)

    # 1. Load data
    print("\n[1/4] Loading preprocessed data...")
    df_eu = load_csv_with_info(EU_CLEAN_FILE, "EU Passengers (Clean)")

    # 2. Define candidate features (without _scaled suffix now)
    print("\n[2/4] Defining candidate features...")
    all_features = [
        # Raw numerical features (will be scaled in pipeline)
        "age",
        "luggage_weight_kg",
        "total_flighttime",
        "total_traveltime",
        "layover_time",
        "layover_ratio_log",
        # Binary features
        "is_male",
        "is_business",
        "has_family",
        "is_long_haul",
        # Categorical features
        "layover_category",
        "flight_time_category",
        "age_group",
        # Encoded airport features
        "shopped_at_encoded",
        "departure_IATA_1_encoded",
        "destination_IATA_1_encoded",
        "departure_IATA_2_encoded",
        "destination_IATA_2_encoded",
        # Interaction features
        "business_longhaul",
        "age_business",
        "family_luggage",
        "layover_shopping_time",
        "male_business",
    ]

    candidate_features = [f for f in all_features if f in df_eu.columns]
    print(f"\nCandidate features: {len(candidate_features)}")

    # 3. Train model
    print("\n[3/4] Training XGBoost with proper pipeline...")
    xgb_pipeline, xgb_metrics, selected_features, selector, preprocessing_info = (
        train_xgboost_no_leakage(df_eu, candidate_features)
    )

    # 4. Save model
    print("\n[4/4] Saving model...")
    print_section_header("SAVING MODEL")

    model_data = {
        "model": xgb_pipeline,
        "feature_selector": selector,
        "preprocessing_info": preprocessing_info,
        "model_type": "XGBoost with SMOTE (No Leakage)",
        "feature_cols": selected_features,
        "candidate_features": candidate_features,
        "metrics": xgb_metrics,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to: {MODEL_FILE}")

    # Summary
    print_section_header("TRAINING COMPLETE")
    print(f"\n✓ Model:               XGBoost with SMOTE (No Leakage)")
    print(f"✓ Features Used:       {len(selected_features)}/{len(candidate_features)}")
    print(
        f"✓ Test Accuracy:       {xgb_metrics['test_accuracy']:.4f} ({xgb_metrics['test_accuracy'] * 100:.2f}%)"
    )
    print(f"✓ F1 Weighted:         {xgb_metrics['f1_weighted']:.4f}")
    print(
        f"✓ Overfitting:         {xgb_metrics['train_accuracy'] - xgb_metrics['test_accuracy']:.4f}"
    )
    print(f"\n✅ NO DATA LEAKAGE:")
    print(f"   • Scaling done INSIDE pipeline on training data only")
    print(f"   • Feature selection done on training data only")
    print(f"   • XGBoost handles NaN natively (no fillna)")

    return {
        "pipeline": xgb_pipeline,
        "selected_features": selected_features,
        "candidate_features": candidate_features,
        "metrics": xgb_metrics,
        "selector": selector,
        "preprocessing_info": preprocessing_info,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Training complete! Next: Run 3_prediction_analysis.py")
