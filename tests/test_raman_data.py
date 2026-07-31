"""
Some general tests of package's functionality.
"""

from raman_data import raman_data, datasets
from raman_data.types import TASK_TYPE


__DATASETS = {
    'codina_diabetes_AGEs' : TASK_TYPE.Classification,
    'sergioalejandrod_AminoAcids_glycine' : TASK_TYPE.Classification,
    'andriitrelin_cells_COOH' : TASK_TYPE.Classification,
    'bioprocess_substrates' : TASK_TYPE.Regression,
    'sugar_mixtures_low_snr' : TASK_TYPE.Regression,
    'wheat_lines' : TASK_TYPE.Classification,
    'adenine_cAg' : TASK_TYPE.Regression
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


def test_filter_by_is_grouped():
    """
    Grouped and ungrouped regression datasets must be disjoint, and known
    examples of each (checked against real data via
    raman_bench.splitting.infer_group_ids_from_targets) must land in the
    right bucket.
    """
    grouped = set(raman_data(task_type=TASK_TYPE.Regression, is_grouped=True))
    ungrouped = set(raman_data(task_type=TASK_TYPE.Regression, is_grouped=False))
    assert grouped.isdisjoint(ungrouped)

    assert "bioprocess_analytes_metrohm" in grouped  # real replicate structure
    assert "adenine_colloidal_gold" in grouped
    assert "amino_acids_glycine" in ungrouped  # no replicate structure
    assert "bioprocess_analytes_kaiser" in ungrouped

    # A dataset never checked for grouping is excluded by either filter value.
    unfiltered = set(raman_data(task_type=TASK_TYPE.Regression))
    assert "wheat_lines" not in unfiltered  # sanity: that one's classification
    never_checked = unfiltered - grouped - ungrouped
    assert "synthetic_organic_pigments_baseline_corrected" in never_checked


def test_filter_by_has_missing_labels():
    """
    Datasets with confirmed missing (NaN) target values and datasets confirmed
    fully labeled must be disjoint, and known examples of each (checked
    against real data) must land in the right bucket.
    """
    missing = set(raman_data(has_missing_labels=True))
    complete = set(raman_data(has_missing_labels=False))
    assert missing.isdisjoint(complete)

    assert "fuel_benchtop" in missing  # 11 of 12 targets have NaN
    assert "bioprocess_substrates" in missing
    assert "chlorinated_samples" in complete
    assert "amino_acids_glycine" in complete

    # A dataset never checked is excluded by either filter value.
    unfiltered = set(raman_data())
    never_checked = unfiltered - missing - complete
    assert "acetic_acid_species" in never_checked


def test_load_dataset():
    """
    Tests loading a dataset.
    """
    test_datasets = [
        "cancer_cell_cooh",            # hosted on Kaggle
        "ecoli_fermentation",          # hosted on HuggingFace
        "adenine_solid_gold"           # hosted on Zenodo
    ]
    for dataset_name in test_datasets:
        print(f"Loading {dataset_name} dataset...")
        dataset = raman_data(dataset_name=dataset_name)
        print(f"Dimensions of spectra: {dataset.spectra.shape}")
        assert dataset.spectra is not None
        assert dataset.targets is not None
        assert dataset.raman_shifts is not None
        assert dataset.info is not None
        assert dataset.task_type is not None
        assert dataset.application_type is not None
        print(f"Dataset {dataset_name} loaded successfully.")
