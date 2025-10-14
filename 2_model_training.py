import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
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
# MODEL TRAINING
# ============================================================================


def train_random_forest(df_eu, feature_cols=None):
    """
    Train Random Forest classifier
    """
    print_section_header("RANDOM FOREST TRAINING")

    # Use provided features or default from config
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Ensure features exist
    feature_cols = [f for f in feature_cols if f in df_eu.columns]

    print(f"\nUsing {len(feature_cols)} features:")
    print(feature_cols)

    # Prepare data
    X = df_eu[feature_cols].fillna(0)
    y = df_eu["amount_spent_cat"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nDataset split:")
    print(
        f"  Training set: {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)"
    )
    print(
        f"  Test set:     {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)"
    )

    # Train Random Forest
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

    # Step 3: Train Random Forest (using all features from config)
    print("\n[3/6] Training Random Forest model...")
    model, X_train, X_test, y_train, y_test, feature_cols = train_random_forest(df_eu)

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
    print(
        f"✓ Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)"
    )
    print(f"✓ F1 Score (weighted): {metrics['f1_weighted']:.4f}")
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

    print("\nNext step: Run 3_prediction_analysis.py")

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance_df,
        "l1_features": l1_features,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    results = main()
    print("\n✓ Model training complete!")
