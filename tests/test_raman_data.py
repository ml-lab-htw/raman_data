"""
Some general tests of package's functionality.
"""

from raman_data import raman_data
from raman_data.loaders.LoaderTools import TASK_TYPE

__DATASETS = {
    # "codina/raman-spectroscopy-of-diabetes": TASK_TYPE.Classification,
    # "sergioalejandrod/raman-spectroscopy": TASK_TYPE.Classification,
    # "andriitrelin/cells-raman-spectra": TASK_TYPE.Classification,
    "chlange/SubstrateMixRaman": TASK_TYPE.Regression,
    #"MIND-Lab_covid+pd_ad_bundle": TASK_TYPE.Classification,
    #"csho33_bacteria_id": TASK_TYPE.Classification,
    # "mendeley_surface-enhanced-raman": TASK_TYPE.Classification,
    #"dtu_raman-spectrum-matching": TASK_TYPE.Classification,
    "sugar mixtures": TASK_TYPE.Regression,
    "Wheat lines" : TASK_TYPE.Classification,
    "Adenine" : TASK_TYPE.Classification
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
    test_datasets = [
        # "codina/raman-spectroscopy-of-diabetes",    # hosted on Kaggle
        "chlange/SubstrateMixRaman",                # hosted on HuggingFace
        # "mendeley_surface-enhanced-raman",          # hosted on external website
        "Adenine"                                   # hosted on Zenodo
    ]
    for dataset_name in test_datasets:
        dataset = raman_data(dataset_name=dataset_name)
        assert dataset.data is not None
        assert dataset.target is not None
        assert dataset.metadata["name"] == dataset_name
