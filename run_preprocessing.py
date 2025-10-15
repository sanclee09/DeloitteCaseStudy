import sys
import time
from datetime import datetime

try:
    import importlib

    preprocessing = importlib.import_module("1_data_preprocessing")
except ImportError as e:
    print(f"Error importing preprocessing module: {e}")
    sys.exit(1)

from utils import print_section_header


def main():
    """Run only the preprocessing stage"""
    start_time = time.time()

    print("=" * 80)
    print("STANDALONE PREPROCESSING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        results = preprocessing.main()

        elapsed = time.time() - start_time

        print_section_header("PREPROCESSING COMPLETE")
        print(f"\n✓ Execution time: {elapsed:.1f} seconds")
        print(f"✓ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✓ Ready for model training (run: python run_training.py)")

        return results

    except Exception as e:
        print(f"\n✗ Preprocessing failed: {str(e)}")
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
