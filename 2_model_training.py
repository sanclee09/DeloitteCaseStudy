import pickle
import warnings
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

from config import *
from utils import *


# ============================================================================
# FEATURE SELECTION
# ============================================================================


def select_features_by_variance(df, feature_cols, threshold=0.01):
    """Remove low-variance features"""
    print_subsection_header("1. Variance-Based Selection")

    X = df[feature_cols].fillna(0).values
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    selected_features = [
        f for f, selected in zip(feature_cols, selector.get_support()) if selected
    ]
    removed_features = [
        f for f, selected in zip(feature_cols, selector.get_support()) if not selected
    ]

    print(f"  Threshold: {threshold}")
    print(f"  Features kept: {len(selected_features)}/{len(feature_cols)}")

    if removed_features:
        print(f"  Removed {len(removed_features)} low-variance features:")
        for f in removed_features:
            print(f"    - {f}")
    else:
        print("  ✓ All features passed variance threshold")

    return selected_features


def select_features_by_correlation(
    df, feature_cols, target_col="amount_spent_cat", threshold=0.8
):
    """Remove highly correlated features (multicollinearity)"""
    print_subsection_header("2. Correlation-Based Selection")

    feature_data = df[feature_cols + [target_col]].fillna(0)
    corr_matrix = feature_data[feature_cols].corr().abs()
    target_corr = feature_data.corr()[target_col].abs()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                feat_i = corr_matrix.columns[i]
                feat_j = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]

                if target_corr[feat_i] >= target_corr[feat_j]:
                    to_remove = feat_j
                    to_keep = feat_i
                else:
                    to_remove = feat_i
                    to_keep = feat_j

                high_corr_pairs.append(
                    {
                        "feature_1": feat_i,
                        "feature_2": feat_j,
                        "correlation": corr_val,
                        "removed": to_remove,
                        "kept": to_keep,
                    }
                )

    features_to_remove = list(set([pair["removed"] for pair in high_corr_pairs]))
    selected_features = [f for f in feature_cols if f not in features_to_remove]

    print(f"  Threshold: {threshold}")
    print(f"  Features kept: {len(selected_features)}/{len(feature_cols)}")

    if high_corr_pairs:
        print(f"  Removed {len(features_to_remove)} correlated features:")
        for pair in high_corr_pairs:
            print(
                f"    - {pair['removed']} (corr with {pair['kept']}: {pair['correlation']:.3f})"
            )
    else:
        print("  ✓ No highly correlated features found")

    return selected_features


def select_features_by_importance(model, feature_cols, threshold=0.001):
    """Select features based on model importance"""
    print_subsection_header("3. Importance-Based Selection")

    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    selected_features = importance_df[importance_df["importance"] > threshold][
        "feature"
    ].tolist()
    removed_features = importance_df[importance_df["importance"] <= threshold][
        "feature"
    ].tolist()

    print(f"  Threshold: {threshold}")
    print(f"  Features kept: {len(selected_features)}/{len(feature_cols)}")

    if removed_features:
        print(f"  Removed {len(removed_features)} low-importance features:")
        for f in removed_features[:5]:
            imp = importance_df[importance_df["feature"] == f]["importance"].values[0]
            print(f"    - {f} (importance: {imp:.4f})")
        if len(removed_features) > 5:
            print(f"    ... and {len(removed_features) - 5} more")
    else:
        print("  ✓ All features have sufficient importance")

    return selected_features, importance_df


def perform_feature_selection(df, feature_cols):
    """Complete feature selection pipeline"""
    print_section_header("FEATURE SELECTION PIPELINE")

    # Variance threshold
    features_after_variance = select_features_by_variance(
        df, feature_cols, threshold=FEATURE_SELECTION_PARAMS["variance_threshold"]
    )

    # Correlation threshold
    features_after_correlation = select_features_by_correlation(
        df,
        features_after_variance,
        threshold=FEATURE_SELECTION_PARAMS["correlation_threshold"],
    )

    # Train initial model for importance
    print_subsection_header("3. Training Initial Model for Importance")

    X = df[features_after_correlation].fillna(0).values
    y = df["amount_spent_cat"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    initial_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    initial_model.fit(X_train, y_train)

    print(
        f"  ✓ Initial model trained (accuracy: {initial_model.score(X_test, y_test):.4f})"
    )

    # Importance threshold
    final_features, importance_df = select_features_by_importance(
        initial_model,
        features_after_correlation,
        threshold=FEATURE_SELECTION_PARAMS["importance_threshold"],
    )

    # Summary
    print_subsection_header("Feature Selection Summary")
    print(f"  Original features:        {len(feature_cols)}")
    print(f"  After variance filter:    {len(features_after_variance)}")
    print(f"  After correlation filter: {len(features_after_correlation)}")
    print(f"  Final selected features:  {len(final_features)}")
    print(
        f"  Reduction:                {len(feature_cols) - len(final_features)} features ({(1 - len(final_features)/len(feature_cols))*100:.1f}%)"
    )

    print("\nFinal feature set:")
    for i, feat in enumerate(final_features, 1):
        imp = importance_df[importance_df["feature"] == feat]["importance"].values[0]
        print(f"  {i:2d}. {feat:30s} (importance: {imp:.4f})")

    return final_features, importance_df


# ============================================================================
# MODEL TRAINING - XGBOOST
# ============================================================================


def train_xgboost(df, feature_cols, enable_tuning=None):
    """Train XGBoost with SMOTE"""
    print_section_header("XGBOOST TRAINING")

    # Use config setting if not explicitly specified
    if enable_tuning is None:
        enable_tuning = ENABLE_HYPERPARAMETER_TUNING

    X = df[feature_cols].fillna(0).values
    y = df["amount_spent_cat"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")

    # Build pipeline
    if enable_tuning:
        # Use base params for tuning
        pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
                ("xgb", XGBClassifier(**XGBOOST_BASE_PARAMS)),
            ]
        )
    else:
        # Use optimal params directly
        print("\n✓ Using optimal hyperparameters from previous tuning")
        pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
                ("xgb", XGBClassifier(**OPTIMAL_XGBOOST_PARAMS)),
            ]
        )

    # Training
    if enable_tuning:
        print_subsection_header("Hyperparameter Tuning")

        param_grid = {
            "xgb__n_estimators": randint(200, 500),
            "xgb__max_depth": randint(4, 10),
            "xgb__learning_rate": uniform(0.05, 0.25),
            "xgb__subsample": uniform(0.7, 0.3),
            "xgb__colsample_bytree": uniform(0.7, 0.3),
            "xgb__gamma": uniform(0, 0.5),
            "xgb__reg_alpha": uniform(0, 1),
            "xgb__reg_lambda": uniform(0.5, 1.5),
        }

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )

        print("\nStarting RandomizedSearchCV...")
        search.fit(X_train, y_train)

        pipeline = search.best_estimator_
        best_cv_score = search.best_score_

        print(f"\n✓ Best CV {CV_SCORING}: {best_cv_score:.4f}")
    else:
        print_subsection_header("Training with Optimal Parameters")
        print("\nOptimal hyperparameters:")
        print(f"  n_estimators:      {OPTIMAL_XGBOOST_PARAMS['n_estimators']}")
        print(f"  max_depth:         {OPTIMAL_XGBOOST_PARAMS['max_depth']}")
        print(f"  learning_rate:     {OPTIMAL_XGBOOST_PARAMS['learning_rate']:.4f}")
        print(f"  subsample:         {OPTIMAL_XGBOOST_PARAMS['subsample']:.4f}")
        print(f"  colsample_bytree:  {OPTIMAL_XGBOOST_PARAMS['colsample_bytree']:.4f}")

        pipeline.fit(X_train, y_train)
        best_cv_score = None

    # Evaluation
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test, "XGBoost")
    metrics["best_cv_score"] = best_cv_score
    metrics["tuning_enabled"] = enable_tuning

    return pipeline, metrics


# ============================================================================
# MODEL TRAINING - RANDOM FOREST
# ============================================================================


def train_random_forest(df, feature_cols, enable_tuning=None):
    """Train Random Forest with SMOTE"""
    print_section_header("RANDOM FOREST TRAINING")

    # Use config setting if not explicitly specified
    if enable_tuning is None:
        enable_tuning = ENABLE_HYPERPARAMETER_TUNING

    X = df[feature_cols].fillna(0).values
    y = df["amount_spent_cat"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")

    # Build pipeline
    if enable_tuning:
        # Use default params for tuning
        pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
                ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
    else:
        # Use optimal params directly
        print("\n✓ Using optimal hyperparameters from previous tuning")
        pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
                ("rf", RandomForestClassifier(**OPTIMAL_RF_PARAMS)),
            ]
        )

    # Training
    if enable_tuning:
        print_subsection_header("Hyperparameter Tuning")

        param_grid = {
            "rf__n_estimators": randint(200, 500),
            "rf__max_depth": randint(10, 30),
            "rf__min_samples_split": randint(20, 100),
            "rf__min_samples_leaf": randint(10, 50),
            "rf__max_features": ["sqrt", "log2", None],
            "rf__class_weight": ["balanced", "balanced_subsample"],
        }

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )

        print("\nStarting RandomizedSearchCV...")
        search.fit(X_train, y_train)

        pipeline = search.best_estimator_
        best_cv_score = search.best_score_

        print(f"\n✓ Best CV {CV_SCORING}: {best_cv_score:.4f}")
    else:
        print_subsection_header("Training with Optimal Parameters")
        print("\nOptimal hyperparameters:")
        print(f"  n_estimators:      {OPTIMAL_RF_PARAMS['n_estimators']}")
        print(f"  max_depth:         {OPTIMAL_RF_PARAMS['max_depth']}")
        print(f"  min_samples_split: {OPTIMAL_RF_PARAMS['min_samples_split']}")
        print(f"  min_samples_leaf:  {OPTIMAL_RF_PARAMS['min_samples_leaf']}")

        pipeline.fit(X_train, y_train)
        best_cv_score = None

    # Evaluation
    metrics = evaluate_model(
        pipeline, X_train, X_test, y_train, y_test, "Random Forest"
    )
    metrics["best_cv_score"] = best_cv_score
    metrics["tuning_enabled"] = enable_tuning

    return pipeline, metrics


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name):
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
        print("  ✗ High overfitting")

    # Cross-validation
    print("\nCross-Validation Performance:")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1
    )
    print(f"  CV F1 (weighted): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

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

    # Confusion matrices (both raw and normalized)
    cm = confusion_matrix(y_test, y_pred_test)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print("\nConfusion Matrix (Raw Counts):")
    print(cm)

    # Create confusion matrix plots with viridis theme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        ax=ax1,
        xticklabels=range(5),
        yticklabels=range(5),
    )
    ax1.set_title(
        f"{model_name} - Raw Confusion Matrix", fontweight="bold", fontsize=12
    )
    ax1.set_ylabel("True Label", fontsize=11)
    ax1.set_xlabel("Predicted Label", fontsize=11)
    ax1.tick_params(axis="x", rotation=45)

    # Normalized confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax2,
        xticklabels=range(5),
        yticklabels=range(5),
    )
    ax2.set_title(
        f"{model_name} - Normalized Confusion Matrix", fontweight="bold", fontsize=12
    )
    ax2.set_ylabel("True Label", fontsize=11)
    ax2.set_xlabel("Predicted Label", fontsize=11)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save the plot
    cm_filename = os.path.join(
        OUTPUT_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.savefig(cm_filename, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"\n✓ Confusion matrix plot saved to: {cm_filename}")
    plt.close()

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "classification_report": report,
    }

    return metrics


# ============================================================================
# MODEL COMPARISON
# ============================================================================


def compare_models(xgb_metrics, rf_metrics):
    """Compare XGBoost vs Random Forest"""
    print_section_header("MODEL COMPARISON")

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Test Accuracy",
                "F1 Macro",
                "F1 Weighted",
                "CV F1 Mean",
                "CV F1 Std",
                "Overfitting",
            ],
            "XGBoost": [
                f"{xgb_metrics['test_accuracy']:.4f}",
                f"{xgb_metrics['f1_macro']:.4f}",
                f"{xgb_metrics['f1_weighted']:.4f}",
                f"{xgb_metrics['cv_f1_mean']:.4f}",
                f"{xgb_metrics['cv_f1_std']:.4f}",
                f"{xgb_metrics['train_accuracy'] - xgb_metrics['test_accuracy']:.4f}",
            ],
            "Random Forest": [
                f"{rf_metrics['test_accuracy']:.4f}",
                f"{rf_metrics['f1_macro']:.4f}",
                f"{rf_metrics['f1_weighted']:.4f}",
                f"{rf_metrics['cv_f1_mean']:.4f}",
                f"{rf_metrics['cv_f1_std']:.4f}",
                f"{rf_metrics['train_accuracy'] - rf_metrics['test_accuracy']:.4f}",
            ],
        }
    )

    print("\n" + comparison_df.to_string(index=False))

    # Determine winner
    print("\n" + "=" * 60)
    print("WINNER SELECTION")
    print("=" * 60)

    xgb_score = xgb_metrics["test_accuracy"]
    rf_score = rf_metrics["test_accuracy"]

    if xgb_score > rf_score:
        winner = "XGBoost"
        diff = xgb_score - rf_score
        print(f"\n🏆 Winner: XGBoost")
        print(f"   Test Accuracy: {xgb_score:.4f} vs {rf_score:.4f}")
        print(f"   Advantage: +{diff:.4f} ({diff*100:.2f}%)")
    elif rf_score > xgb_score:
        winner = "Random Forest"
        diff = rf_score - xgb_score
        print(f"\n🏆 Winner: Random Forest")
        print(f"   Test Accuracy: {rf_score:.4f} vs {xgb_score:.4f}")
        print(f"   Advantage: +{diff:.4f} ({diff*100:.2f}%)")
    else:
        winner = "Tie"
        print(f"\n🤝 Tie: Both models perform equally")
        print(f"   Defaulting to XGBoost (faster inference)")
        winner = "XGBoost"

    return winner, comparison_df


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================


def analyze_feature_importance(pipeline, feature_cols, model_name, save_plot=True):
    """Extract and visualize feature importance"""
    print_section_header(f"FEATURE IMPORTANCE - {model_name}")

    # Get model from pipeline
    if "xgb" in pipeline.named_steps:
        model = pipeline.named_steps["xgb"]
    else:
        model = pipeline.named_steps["rf"]

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nTop 15 Most Important Features ({model_name}):")
    for idx, row in importance_df.head(15).iterrows():
        pct = row["importance"] / importance_df["importance"].sum() * 100
        print(f"  {row['feature']:30s}: {row['importance']:.4f} ({pct:5.1f}%)")

    # Save plot if requested and this is the winning model
    if save_plot and model_name == "XGBoost":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = importance_df.head(15)
        ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 Feature Importance - {model_name}", fontweight="bold")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_FILE, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"\n✓ Feature importance plot saved to: {FEATURE_IMPORTANCE_FILE}")
        plt.close()

    return importance_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main training pipeline with model comparison"""
    print("=" * 80)
    print("DELOITTE CASE STUDY - MODEL COMPARISON (XGBoost vs Random Forest)")
    print("=" * 80)

    # 1. Load data
    print("\n[1/6] Loading preprocessed data...")
    df_eu = load_csv_with_info(EU_CLEAN_FILE, "EU Passengers (Clean)")

    # 2. Define features
    print("\n[2/6] Defining candidate features...")
    all_features = [
        "age_scaled",
        "luggage_weight_kg_scaled",
        "total_flighttime_scaled",
        "total_traveltime_scaled",
        "layover_time_scaled",
        "layover_ratio_log_scaled",
        "is_male",
        "is_business",
        "has_family",
        "has_connection",
        "travel_complexity",
        "is_long_haul",
        "layover_category",
        "shopped_at_encoded",
        "departure_IATA_1_encoded",
        "destination_IATA_1_encoded",
        "departure_IATA_2_encoded",
        "destination_IATA_2_encoded",
    ]
    candidate_features = [f for f in all_features if f in df_eu.columns]
    print(f"\nCandidate features: {len(candidate_features)}")

    # 3. Feature selection
    print("\n[3/6] Performing feature selection...")
    selected_features, _ = perform_feature_selection(df_eu, candidate_features)

    # Check tuning mode
    tuning_mode = (
        "ENABLED" if ENABLE_HYPERPARAMETER_TUNING else "DISABLED (using optimal params)"
    )
    print(f"\n⚙️  Hyperparameter tuning: {tuning_mode}")
    if not ENABLE_HYPERPARAMETER_TUNING:
        print("   (This will be much faster - ~2 minutes vs ~30 minutes)")

    # 4. Train XGBoost
    print("\n[4/6] Training XGBoost...")
    xgb_pipeline, xgb_metrics = train_xgboost(df_eu, selected_features)
    xgb_importance = analyze_feature_importance(
        xgb_pipeline, selected_features, "XGBoost", save_plot=True
    )

    # 5. Train Random Forest
    print("\n[5/6] Training Random Forest...")
    rf_pipeline, rf_metrics = train_random_forest(df_eu, selected_features)
    rf_importance = analyze_feature_importance(
        rf_pipeline, selected_features, "Random Forest", save_plot=False
    )

    # 6. Compare and select best model
    print("\n[6/6] Comparing models...")
    winner, comparison_df = compare_models(xgb_metrics, rf_metrics)

    # Select best model
    if winner == "XGBoost":
        best_pipeline = xgb_pipeline
        best_metrics = xgb_metrics
        best_importance = xgb_importance
    else:
        best_pipeline = rf_pipeline
        best_metrics = rf_metrics
        best_importance = rf_importance

    # Save best model
    print_section_header("SAVING BEST MODEL")

    model_data = {
        "model": best_pipeline,
        "model_type": f"{winner} with SMOTE",
        "feature_cols": selected_features,
        "feature_importance": best_importance,
        "metrics": best_metrics,
        "comparison": {
            "xgb_metrics": xgb_metrics,
            "rf_metrics": rf_metrics,
            "winner": winner,
        },
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✓ Best model ({winner}) saved to: {MODEL_FILE}")

    # Final summary
    print_section_header("TRAINING COMPLETE")
    print(f"\n✓ Winner:              {winner}")
    print(f"✓ Features Used:       {len(selected_features)}/{len(candidate_features)}")
    print(
        f"✓ Test Accuracy:       {best_metrics['test_accuracy']:.4f} ({best_metrics['test_accuracy']*100:.2f}%)"
    )
    print(f"✓ F1 Weighted:         {best_metrics['f1_weighted']:.4f}")
    print(
        f"✓ CV F1:               {best_metrics['cv_f1_mean']:.4f} (±{best_metrics['cv_f1_std']:.4f})"
    )

    if best_metrics["best_cv_score"]:
        print(f"✓ Best CV {CV_SCORING}:    {best_metrics['best_cv_score']:.4f}")

    print(f"\n✓ Model comparison saved for presentation!")

    return {
        "best_pipeline": best_pipeline,
        "best_model": winner,
        "selected_features": selected_features,
        "metrics": best_metrics,
        "comparison": comparison_df,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Training complete! Next: Run 3_prediction_analysis.py")
