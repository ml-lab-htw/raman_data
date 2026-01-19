"""
Some general tests of package's functionality.
"""

from raman_data import raman_data, datasets
from raman_data.types import TASK_TYPE

__TODO_DATASETS = {
    #"MIND-Lab_covid+pd_ad_bundle": TASK_TYPE.Classification,
    #"csho33_bacteria_id": TASK_TYPE.Classification,
    # "mendeley_surface-enhanced-raman": TASK_TYPE.Classification,
    #"dtu_raman-spectrum-matching": TASK_TYPE.Classification,
}

__DATASETS = {
    'codina/diabetes/AGEs' : TASK_TYPE.Classification,
    'sergioalejandrod/AminoAcids/glycine' : TASK_TYPE.Classification,
    'andriitrelin/cells-raman-spectra/COOH' : TASK_TYPE.Classification,
    'chlange/SubstrateMixRaman' : TASK_TYPE.Regression,
    'sugar mixtures' : TASK_TYPE.Regression,
    'Wheat lines' : TASK_TYPE.Classification,
    'Adenine' : TASK_TYPE.Regression
}


def test_list_all_datasets():
    """
    Tests listing all available datasets.
    """
    all_datasets = raman_data()
    assert isinstance(all_datasets, list)
    expected = set(datasets.list_datasets())
    assert set(all_datasets) == expected

    for dataset in expected:
        assert dataset in all_datasets

def test_list_classification_datasets():
    """
    Tests listing datasets with a filter.
    """
    classification_datasets = raman_data(task_type=TASK_TYPE.Classification)
    assert isinstance(classification_datasets, list)
    expected = set(datasets.list_datasets(task_type=TASK_TYPE.Classification))
    assert set(classification_datasets) == expected

def test_load_dataset():
    """
    Tests loading a dataset.
    """
    test_datasets = [
        "codina/diabetes/earLobe",                  # hosted on Kaggle
        "chlange/SubstrateMixRaman",                # hosted on HuggingFace
        # "mendeley_surface-enhanced-raman",          # hosted on external website
        "Adenine"                                   # hosted on Zenodo
    ]
    for dataset_name in test_datasets:
        dataset = raman_data(dataset_name=dataset_name)
        assert dataset.spectra is not None
        assert dataset.targets is not None
        assert dataset.raman_shifts is not None
        assert dataset.metadata["full_name"] is not None
        assert dataset.metadata["source"] is not None
