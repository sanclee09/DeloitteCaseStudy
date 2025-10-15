import sys
import time
from datetime import datetime

# Import pipeline modules
try:
    import importlib

    preprocessing = importlib.import_module("1_data_preprocessing")
    model_training = importlib.import_module("2_model_training")
    prediction_analysis = importlib.import_module("3_prediction_analysis")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

from config import *
from utils import print_section_header, format_currency


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================


def run_complete_pipeline():
    """
    Run the complete analysis pipeline
    """
    start_time = time.time()

    print("=" * 80)
    print("DELOITTE CASE STUDY - COMPLETE ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    try:
        # ====================================================================
        # STAGE 1: DATA PREPROCESSING
        # ====================================================================
        print_section_header("STAGE 1/3: DATA PREPROCESSING")
        print("Running: 1_data_preprocessing.py")
        print()

        stage1_start = time.time()
        preprocessing_results = preprocessing.main()
        stage1_time = time.time() - stage1_start

        results["preprocessing"] = preprocessing_results
        print(f"\n✓ Stage 1 complete in {stage1_time:.1f} seconds")

        # ====================================================================
        # STAGE 2: MODEL TRAINING
        # ====================================================================
        print_section_header("STAGE 2/3: MODEL TRAINING")
        print("Running: 2_model_training.py")
        print()

        stage2_start = time.time()
        training_results = model_training.main()
        stage2_time = time.time() - stage2_start

        results["training"] = training_results
        print(f"\n✓ Stage 2 complete in {stage2_time:.1f} seconds")

        # ====================================================================
        # STAGE 3: PREDICTION & PROFITABILITY
        # ====================================================================
        print_section_header("STAGE 3/3: PREDICTION & PROFITABILITY")
        print("Running: 3_prediction_analysis.py")
        print()

        stage3_start = time.time()
        prediction_results = prediction_analysis.main()
        stage3_time = time.time() - stage3_start

        results["prediction"] = prediction_results
        print(f"\n✓ Stage 3 complete in {stage3_time:.1f} seconds")

    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return None

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - start_time

    print_section_header("PIPELINE COMPLETE")

    print("\n📊 EXECUTION SUMMARY:")
    print(f"  Stage 1 (Preprocessing):     {stage1_time:6.1f}s")
    print(f"  Stage 2 (Model Training):    {stage2_time:6.1f}s")
    print(f"  Stage 3 (Prediction):        {stage3_time:6.1f}s")
    print(f"  {'─' * 40}")
    print(f"  Total execution time:        {total_time:6.1f}s")

    print("\n📁 OUTPUT FILES:")
    print(f"  Processed data:     {PROCESSED_DATA_DIR}")
    print(f"  Trained model:      {MODEL_FILE}")
    print(f"  Profitability:      {PROFITABILITY_FILE}")
    print(f"  Visualizations:     {VISUALIZATION_FILE}")

    # Extract key results
    if prediction_results and "profitability" in prediction_results:
        profitability = prediction_results["profitability"]
        top_airport = profitability.index[0]
        top_profit = profitability.iloc[0]["annual_profit"]

        print("\n🎯 FINAL RECOMMENDATION:")
        print(f"  Primary Target:     {top_airport}")
        print(f"  Expected Profit:    {format_currency(top_profit)} annually")
        print(f"  Confidence Level:   Moderate (±20% variance expected)")

    if training_results and "metrics" in training_results:
        metrics = training_results["metrics"]
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"  Test Accuracy:      {metrics['test_accuracy']:.2%}")
        print(f"  F1 Score:           {metrics['f1_weighted']:.4f}")
        print(
            f"  CV Score:           {metrics['cv_f1_mean']:.4f} (±{metrics['cv_f1_std']:.4f})"
        )

    print(f"\n✓ Analysis pipeline completed successfully!")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


# ============================================================================
# INDIVIDUAL STAGE RUNNERS
# ============================================================================


def run_preprocessing_only():
    """Run only preprocessing stage"""
    print_section_header("RUNNING PREPROCESSING ONLY")
    return preprocessing.main()


def run_training_only():
    """Run only model training stage"""
    print_section_header("RUNNING MODEL TRAINING ONLY")
    return model_training.main()


def run_prediction_only():
    """Run only prediction stage"""
    print_section_header("RUNNING PREDICTION ONLY")
    return prediction_analysis.main()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def print_usage():
    """Print usage instructions"""
    print("Usage: python main.py [option]")
    print("\nOptions:")
    print("  all          Run complete pipeline (default)")
    print("  preprocess   Run preprocessing only")
    print("  train        Run model training only")
    print("  predict      Run prediction analysis only")
    print("  help         Show this help message")
    print("\nExamples:")
    print("  python main.py")
    print("  python main.py all")
    print("  python main.py preprocess")


def main():
    """Main entry point with command line interface"""

    # Parse command line arguments
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
    else:
        option = "all"

    # Route to appropriate function
    if option in ["all", "complete", "full"]:
        results = run_complete_pipeline()
    elif option in ["preprocess", "preprocessing", "1"]:
        results = run_preprocessing_only()
    elif option in ["train", "training", "model", "2"]:
        results = run_training_only()
    elif option in ["predict", "prediction", "analysis", "3"]:
        results = run_prediction_only()
    elif option in ["help", "-h", "--help", "?"]:
        print_usage()
        return
    else:
        print(f"Unknown option: {option}")
        print()
        print_usage()
        return

    if results:
        print("\n" + "=" * 80)
        print("SUCCESS - Ready for presentation!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED - Please check errors above")
        print("=" * 80)


if __name__ == "__main__":
    main()
