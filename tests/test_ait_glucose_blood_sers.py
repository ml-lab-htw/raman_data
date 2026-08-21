"""
Test for the ait_glucose_blood_sers loader (GitHubLoader).

See raman_data/loaders/GitHubLoader.py's DatasetInfo entry and
_load_ait_glucose_blood_sers docstring for the full provenance/consent
caveat associated with this dataset -- it is a deliberate, flagged
exception to raman_data's usual inclusion criteria, not a template.
"""
import numpy as np
import pytest

from raman_data.loaders.GitHubLoader import GitHubLoader


@pytest.mark.skip(reason="GitHub archive download (~56MB repo); run manually.")
def test_ait_glucose_blood_sers():
    dataset = GitHubLoader.load_dataset("ait_glucose_blood_sers")

    assert dataset.spectra.shape == (35, 1999)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    assert dataset.targets.shape[0] == dataset.spectra.shape[0]
    assert dataset.target_names == ["glucose_concentration"]

    assert not np.isnan(dataset.spectra).any()
    assert not np.isnan(dataset.targets).any()

    assert set(np.unique(dataset.targets).tolist()) == {88.0, 95.0, 98.0, 121.0, 166.0, 168.0}

    assert dataset.group_ids is not None
    assert len(set(dataset.group_ids.tolist())) == 6
