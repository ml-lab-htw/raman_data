"""Structured provider preprocessing (DatasetInfo.provider_preprocessing)."""

import pytest

from raman_data import raman_data
from raman_data.datasets import get_dataset_info, list_datasets
from raman_data.provider_preprocessing import PROVIDER_PREPROCESSING
from raman_data.types import PREPROCESSING_KINDS, PreprocessingStep


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown preprocessing kind"):
        PreprocessingStep("smoothing_ish", "Savitzky-Golay")


def test_every_dataset_with_a_prose_description_is_in_the_table():
    """embedded_preprocessing text without a structured entry would be reported as 'unknown'."""
    missing = [
        k for k in list_datasets()
        if get_dataset_info(k).embedded_preprocessing and get_dataset_info(k).provider_preprocessing is None
    ]
    # rod_minerals: the per-entry preprocessing varies by source and is not documented, so it stays unknown.
    assert missing == ["rod_minerals"]


def test_table_steps_use_known_kinds_and_are_applied():
    for key, (steps, _) in PROVIDER_PREPROCESSING.items():
        info = get_dataset_info(key)
        assert info.provider_preprocessing == steps, key
        assert all(s.kind in PREPROCESSING_KINDS for s in steps), key


def test_provider_preprocessed_is_three_valued():
    assert get_dataset_info("marine_pathogens").provider_preprocessed is True
    assert get_dataset_info("cwa_tep").provider_preprocessed is False
    assert get_dataset_info("rod_minerals").provider_preprocessed is None


def test_filter_excludes_unknown_for_both_values():
    yes = set(list_datasets(provider_preprocessed=True))
    no = set(list_datasets(provider_preprocessed=False))
    everything = set(list_datasets())
    assert yes and no and not (yes & no)
    assert everything - yes - no  # unchecked datasets are in neither
    assert set(raman_data(provider_preprocessed=True)) == yes


def test_microgel_variants():
    """lf = linear-fit and rb = rubber-band baseline; mm = min-max and snv = SNV; fingerprint = 850-1800 cm-1."""
    def steps(key):
        return [(s.kind, s.method) for s in get_dataset_info(key).provider_preprocessing]

    assert steps("microgel_size_raw_global") == []
    assert steps("microgel_size_lf_global") == [("baseline", "linear fit")]
    assert steps("microgel_size_rb_global") == [("baseline", "rubber band")]
    assert steps("microgel_size_snv_rb_global") == [("baseline", "rubber band"), ("normalization", "standard normal variate")]
    fp = get_dataset_info("microgel_size_mm_lf_fingerprint").provider_preprocessing
    assert [s.kind for s in fp] == ["baseline", "normalization", "crop"]
    assert fp[-1].params == {"min_cm-1": 850, "max_cm-1": 1800}
    assert get_dataset_info("microgel_size_raw_fingerprint").provider_preprocessed is False
    assert sum(k.startswith("microgel_size_") for k in PROVIDER_PREPROCESSING) == 14
    # Neither 'low-frequency' nor 'bias-corrected' (the earlier, wrong readings of lf and rb).
    for k in list_datasets():
        if k.startswith("microgel_size_"):
            text = get_dataset_info(k).embedded_preprocessing
            assert "low-frequency" not in text and "bias" not in text, k
