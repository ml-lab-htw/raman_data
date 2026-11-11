"""Source publications recorded for datasets whose citation was checked by hand (issue #84)."""

import re

import pytest

from raman_data.datasets import get_dataset_info

XUE_2025 = "10.1038/s42256-025-01027-5"
BI_2024 = "10.1016/j.xcrm.2024.101579"
LEGNER_2019 = "10.1021/acs.energyfuels.9b02944"
VOIGT_2019 = "10.1016/j.fuel.2018.09.006"
KORONAKI_2024 = "10.1002/aic.18494"  # AIChE J. 70(10), e18494; not the Small paper the loader cited before


@pytest.mark.parametrize("dataset, dois", [
    ("serum_prostate_cancer", {XUE_2025, BI_2024}),
    ("serum_alzheimer_disease", {XUE_2025}),
    ("comfile_stroke", {XUE_2025}),
    ("fuel_benchtop", {LEGNER_2019, VOIGT_2019}),
    ("fuel_handheld", {LEGNER_2019, VOIGT_2019}),
    ("microgel_size_raw_global", {KORONAKI_2024}),
    ("microgel_size_snv_rb_fingerprint", {KORONAKI_2024}),
])
def test_paper_and_bibtex_cite_the_source_publications(dataset, dois):
    meta = get_dataset_info(dataset).metadata
    papers = meta["paper"] if isinstance(meta["paper"], list) else [meta["paper"]]
    assert all(any(d in p for p in papers) for d in dois), papers
    bib_dois = {d.lower() for d in re.findall(r"DOI=\{([^}]+)\}", meta["bibtex"])}
    assert bib_dois == dois
    assert meta["bibtex"].count("@article{") == len(dois)
