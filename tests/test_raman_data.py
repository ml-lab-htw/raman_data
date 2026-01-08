"""
Some general tests of package's functionality.
"""

from raman_data import raman_data
from raman_data.types import TASK_TYPE

__TODO_DATASETS = {
    # "andriitrelin/cells-raman-spectra": TASK_TYPE.Classification,
    # "mendeley_surface-enhanced-raman": TASK_TYPE.Classification,
    #"dtu_raman-spectrum-matching": TASK_TYPE.Classification,
}

__DATASETS = {
    'codina/diabetes/AGEs' : TASK_TYPE.Classification,
    'codina/diabetes/earLobe' : TASK_TYPE.Classification,
    'codina/diabetes/innerArm' : TASK_TYPE.Classification,
    'codina/diabetes/thumbNail' : TASK_TYPE.Classification,
    'codina/diabetes/vein' : TASK_TYPE.Classification,
    'sergioalejandrod/AminoAcids/glycine' : TASK_TYPE.Classification,
    'sergioalejandrod/AminoAcids/leucine' : TASK_TYPE.Classification,
    'sergioalejandrod/AminoAcids/phenylalanine' : TASK_TYPE.Classification,
    'sergioalejandrod/AminoAcids/tryptophan' : TASK_TYPE.Classification,
    'chlange/SubstrateMixRaman' : TASK_TYPE.Regression,
    'chlange/RamanSpectraEcoliFermentation' : TASK_TYPE.Classification,
    "chlange/FuelRamanSpectraBenchtop" : TASK_TYPE.Regression,
    'sugar mixtures' : TASK_TYPE.Regression,
    'Wheat lines' : TASK_TYPE.Classification,
    'Adenine' : TASK_TYPE.Classification,
    'MIND-Lab_Raman-Spectra-Data_covid-dataset': TASK_TYPE.Classification,
    'MIND-Lab_Raman-Spectra-Data_pd-ad-dataset': TASK_TYPE.Classification,
    'csho33_bacteria_2018clinical': TASK_TYPE.Classification,
    'csho33_bacteria_2019clinical': TASK_TYPE.Classification,
    'csho33_bacteria_finetune': TASK_TYPE.Classification,
    'csho33_bacteria_reference': TASK_TYPE.Classification,
    'csho33_bacteria_test': TASK_TYPE.Classification
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
        "codina/diabetes/earLobe",                  # hosted on Kaggle
        "chlange/SubstrateMixRaman",                # hosted on HuggingFace
        # "mendeley_surface-enhanced-raman",          # hosted on external website
        "Adenine",                                  # hosted on Zenodo
        "csho33_bacteria_test"                      # hosted on GitHub
    ]
    for dataset_name in test_datasets:
        dataset = raman_data(dataset_name=dataset_name)
        assert dataset.data is not None
        assert dataset.target is not None
        assert dataset.spectra is not None
        assert dataset.metadata["full_name"] is not None
        assert dataset.metadata["source"] is not None
