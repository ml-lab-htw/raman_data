from raman_data import raman_data

def test_list_all_datasets():
    """
    Tests listing all available datasets.
    """
    all_datasets = raman_data()
    assert isinstance(all_datasets, list)
    assert "diabetes_kaggle" in all_datasets

def test_list_classification_datasets():
    """
    Tests listing datasets with a filter.
    """
    classification_datasets = raman_data(task_type='classification')
    assert isinstance(classification_datasets, list)
    assert "diabetes_kaggle" in classification_datasets
    assert "SubstrateMixRaman_hf" not in classification_datasets

def test_load_dataset():
    """
    Tests loading a dataset.
    """
    dataset = raman_data(name="diabetes_kaggle")
    assert dataset.data is not None
    assert dataset.target is not None
    assert dataset.metadata['name'] == 'diabetes_kaggle'
