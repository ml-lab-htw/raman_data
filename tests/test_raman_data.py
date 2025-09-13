"""
Some general tests of package's functionality.
"""

from raman_data import raman_data
from raman_data.loaders.LoaderTools import TASK_TYPE

__DATASETS = {
    "codina/raman-spectroscopy-of-diabetes": TASK_TYPE.Classification,
    "sergioalejandrod/raman-spectroscopy": TASK_TYPE.Classification,
    "andriitrelin/cells-raman-spectra": TASK_TYPE.Classification
}

def test_list_all_datasets():
    """
    Tests listing all available datasets.
    """
    all_datasets = raman_data()
    assert isinstance(all_datasets, list)
    assert len(all_datasets) == len(__DATASETS)
    
    for dataset in __DATASETS:
        assert dataset in all_datasets

def test_list_classification_datasets():
    """
    Tests listing datasets with a filter.
    """
    classification_datasets = raman_data(task_type=TASK_TYPE.Classification)
    assert isinstance(classification_datasets, list)
    
    for dataset_name, task_type in __DATASETS.items():
        check = dataset_name in classification_datasets
        assert check if task_type == TASK_TYPE.Classification else not check

def test_load_dataset():
    """
    Tests loading a dataset.
    """
    dataset = raman_data(dataset_name="codina/raman-spectroscopy-of-diabetes")
    assert dataset.data is not None
    assert dataset.target is not None
    assert dataset.metadata["name"] == "codina/raman-spectroscopy-of-diabetes"
