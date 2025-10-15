"""
Standalone model training runner
Run this only when you need to train/retrain models
Requires preprocessed data to exist
"""

import sys
import time
from datetime import datetime

# Import training module
try:
    import importlib

    model_training = importlib.import_module("2_model_training")
except ImportError as e:
    print(f"Error importing training module: {e}")
    sys.exit(1)

from utils import print_section_header
from config import EU_CLEAN_FILE
import os


def main():
    """Run only the model training stage"""
    start_time = time.time()

    print("=" * 80)
    print("STANDALONE MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if preprocessed data exists
    if not os.path.exists(EU_CLEAN_FILE):
        print(f"✗ Error: Preprocessed data not found at {EU_CLEAN_FILE}")
        print("Please run preprocessing first: python run_preprocessing.py")
        return None

    try:
        results = model_training.main()

        elapsed = time.time() - start_time

        print_section_header("TRAINING COMPLETE")
        print(f"\n✓ Execution time: {elapsed:.1f} seconds")
        print(f"✓ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✓ Ready for predictions (run: python run_prediction.py)")

        return results

    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED")
        print("=" * 80)
        sys.exit(1)
