"""
Some general tests of package's functionality.
"""

from raman_data import raman_data, datasets
from raman_data.types import TASK_TYPE

__TODO_DATASETS = {
    #"MIND_Lab_covid_and_pd_ad_bundle": TASK_TYPE.Classification,
    #"csho33_bacteria_id": TASK_TYPE.Classification,
    # "mendeley_surface-enhanced-raman": TASK_TYPE.Classification,
    #"dtu_raman-spectrum-matching": TASK_TYPE.Classification,
}

__DATASETS = {
    'codina_diabetes_AGEs' : TASK_TYPE.Classification,
    'sergioalejandrod_AminoAcids_glycine' : TASK_TYPE.Classification,
    'andriitrelin_cells-raman-spectra_COOH' : TASK_TYPE.Classification,
    'chlange_SubstrateMixRaman' : TASK_TYPE.Regression,
    'sugar_mixtures' : TASK_TYPE.Regression,
    'wheat_lines' : TASK_TYPE.Classification,
    'adenine' : TASK_TYPE.Regression
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
        "codina_diabetes_earLobe",                  # hosted on Kaggle
        "chlange_SubstrateMixRaman",                # hosted on HuggingFace
        # "mendeley_surface-enhanced-raman",          # hosted on external website
        "adenine"                                   # hosted on Zenodo
    ]
    for dataset_name in test_datasets:
        print(f"Loading {dataset_name} dataset...")
        dataset = raman_data(dataset_name=dataset_name)
        assert dataset.spectra is not None
        assert dataset.targets is not None
        assert dataset.raman_shifts is not None
        assert dataset.metadata["full_name"] is not None
        assert dataset.metadata["source"] is not None
        print(f"Dataset {dataset_name} loaded successfully.")
