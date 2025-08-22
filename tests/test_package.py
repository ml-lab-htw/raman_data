import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from raman_data import raman_data

def run_test():
    """
    Tests the raman_data package.
    """
    print("--- Testing raman_data package ---")

    # Test listing all datasets
    print("\n1. Listing all available datasets:")
    all_datasets = raman_data()
    print(f"   - Found datasets: {all_datasets}")
    assert isinstance(all_datasets, list)
    assert "diabetes_kaggle" in all_datasets

    # Test listing datasets with a filter
    print("\n2. Listing classification datasets:")
    classification_datasets = raman_data(task_type='classification')
    print(f"   - Found classification datasets: {classification_datasets}")
    assert isinstance(classification_datasets, list)
    assert "diabetes_kaggle" in classification_datasets
    assert "SubstrateMixRaman_hf" not in classification_datasets

    # Test loading a dataset
    print("\n3. Loading a dataset (diabetes_kaggle):")
    dataset = raman_data(name="diabetes_kaggle")
    print(f"   - Dataset loaded successfully.")
    print(f"   - Data shape: {dataset.data.shape}")
    print(f"   - Target shape: {dataset.target.shape}")
    print(f"   - Metadata: {dataset.metadata}")
    assert dataset.data is not None
    assert dataset.target is not None
    assert dataset.metadata['name'] == 'diabetes_kaggle'

    print("\n--- Test finished successfully! ---")

if __name__ == "__main__":
    run_test()

