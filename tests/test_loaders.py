"""
A general checkup of loader's implementation.
"""

import collections
import re
from raman_data import raman_data
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.KaggleLoader import KaggleLoader
from raman_data.loaders.HuggingFaceLoader import HuggingFaceLoader
from raman_data.loaders.ZenodoLoader import ZenodoLoader
from raman_data.loaders.MiscLoader import MiscLoader
from raman_data.loaders.MendeleyLoader import MendeleyLoader
from raman_data.loaders.RWTHLoader import RWTHLoader
from raman_data.loaders.GoogleDriveLoader import GoogleDriveLoader
from raman_data.loaders.FigshareLoader import FigshareLoader
from raman_data.loaders.GitHubLoader import GitHubLoader
import numpy as np
import pytest
import os

__LOADERS = [
    KaggleLoader,
    HuggingFaceLoader,
    ZenodoLoader,
    MiscLoader,
    MendeleyLoader,
    RWTHLoader,
    GoogleDriveLoader,
    FigshareLoader,
    GitHubLoader,
]


def _groups_are_label_pure(dataset):
    """A physical sample cannot carry two labels."""
    y = np.asarray(dataset.targets).ravel()
    g = dataset.group_ids
    return all(len(np.unique(y[g == i])) == 1 for i in np.unique(g))


def test_interfacing():
    for loader in __LOADERS:
        # This includes BaseLoader's __subclasshook__ method
        assert issubclass(loader, BaseLoader)
        assert hasattr(loader, 'DATASETS')

@pytest.mark.skip(reason="Zenodo dataset is huge and slow to download for local/dev testing.")
@pytest.mark.skipif(os.environ.get('CI') is not None, reason="Zenodo dataset is huge for CI")
def test_zen_loader_download():
    # Using a known dataset ID from ZenodoLoader.DATASETS
    test_dataset_name = list(ZenodoLoader.DATASETS.keys())[0]
    download_dir = ZenodoLoader.download_dataset(dataset_name=test_dataset_name)
    assert download_dir is not None
    assert os.path.isdir(download_dir)
    assert len(os.listdir(download_dir)) > 0

@pytest.mark.skip(reason="Zenodo dataset is huge and slow to download for local/dev testing.")
@pytest.mark.skipif(os.environ.get('CI') is not None, reason="Zenodo dataset is huge for CI")
def test_zen_loader_load():
    # Using a known dataset ID from ZenodoLoader.DATASETS
    test_dataset_name = list(ZenodoLoader.DATASETS.keys())[0]
    dataset = ZenodoLoader.load_dataset(dataset_name=test_dataset_name)
    assert dataset.spectra is not None
    assert dataset.targets is not None
    assert dataset.raman_shifts is not None
    assert dataset.metadata["full_name"] is not None
    assert dataset.metadata["source"] is not None

@pytest.mark.skip(reason="MiscLoader datasets require manual download from OneDrive.")
def test_misc_loader_load_missing():
    # Should print missing file warning and return None if files are missing
    dataset = MiscLoader.load_dataset("deepr_denoising", cache_path="/tmp/nonexistent_misc")
    assert dataset is None


def test_misc_loader_load_missing():
    # the timegate dataset is small and can be included in the repo for testing
    dataset = HuggingFaceLoader.load_dataset("bioprocess_analytes_timegate")
    assert dataset.target_names == ["Glucose", "Acetate", "MagnesiumSulfate"]
    assert dataset.spectra.shape == (133, 486)


def test_streptococcus_thermophilus_fermentation_timegate():
    # the timegate dataset is small and can be included in the repo for testing
    dataset = HuggingFaceLoader.load_dataset("streptococcus_thermophilus_fermentation_timegate")
    assert dataset.target_names == ["Lactose", "Galactose", "Lactate", "OD600"]
    assert dataset.spectra.shape == (14, 117)


@pytest.mark.skip(reason="Zenodo sugar dataset requires download; run manually.")
@pytest.mark.skipif(os.environ.get('CI') is not None, reason="Too large for CI")
def test_sugar_mixtures_low_snr():
    dataset = ZenodoLoader.load_dataset("sugar_mixtures_low_snr")
    assert dataset.target_names == ['Sucrose', 'Fructose', 'Maltose', 'Glucose']
    assert dataset.spectra.shape[0] == dataset.targets.shape[0]  # samples match
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)  # spectral axis matches
    assert dataset.targets.shape[1] == 4  # 4 analyte concentrations


@pytest.mark.skip(reason="~97MB Zenodo download; run manually.")
def test_locust_phase_hemolymph():
    dataset = ZenodoLoader.load_dataset("locust_phase_hemolymph")
    assert dataset.target_names == ["crowded", "isolated"]
    assert dataset.spectra.shape == (5000, 855)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 2500, 1: 2500}
    assert dataset.group_ids is not None
    assert dataset.group_ids.shape == (5000,)
    assert len(set(dataset.group_ids.tolist())) == 200

def test_cspp_serum_metabolites():
    # scoped to a single ~3MB CSV via file_typ, small enough to run in CI
    dataset = ZenodoLoader.load_dataset("cspp_serum_metabolites")
    assert dataset.target_names == [
        "serum (unspiked)",
        "serum + ergothioneine (25 uM)",
        "serum + hypoxanthine (50 uM)",
    ]
    assert dataset.spectra.shape == (150, 2048)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 50, 1: 50, 2: 50}
    assert dataset.group_ids is not None
    assert dataset.group_ids.shape == (150,)
    assert len(set(dataset.group_ids.tolist())) == 15
    assert _groups_are_label_pure(dataset)
    assert not np.isnan(dataset.spectra).any()


def test_marine_pathogens():
    # small (~4MB) raw-GitHub CSV download, fast and reliable enough for CI
    dataset = MiscLoader.load_dataset("marine_pathogens")
    assert dataset.target_names == [f"SX-{i}" for i in range(1, 9)]
    assert dataset.spectra.shape == (1138, 1200)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    assert dataset.raman_shifts.min() == 600.0
    assert dataset.raman_shifts.max() == 1800.0
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 149, 1: 150, 2: 150, 3: 102, 4: 137, 5: 150, 6: 150, 7: 150}
    assert not np.isnan(dataset.spectra).any()
    assert dataset.spectra.min() >= 0.0 and dataset.spectra.max() <= 1.0
    # No source metadata ties spectra to physical isolates/replicates.
    assert dataset.group_ids is None


def test_marine_pathogens_binary():
    # small (<1MB) raw-GitHub CSV download, fast and reliable enough for CI
    dataset = MiscLoader.load_dataset("marine_pathogens_binary")
    assert dataset.target_names == ["Acinetobacter baumannii", "Pseudomonas nitritireducens"]
    assert dataset.spectra.shape == (261, 1200)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    assert dataset.raman_shifts.min() == 600.0
    assert dataset.raman_shifts.max() == 1800.0
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 150, 1: 111}
    assert not np.isnan(dataset.spectra).any()
    assert dataset.spectra.min() >= 0.0 and dataset.spectra.max() <= 1.0
    # No source metadata ties spectra to physical isolates/replicates.
    assert dataset.group_ids is None


@pytest.mark.skip(reason="Google Drive dataset download is slow.")
def test_load_organic_compounds_raw(tmp_path):
    # given

    # when
    dataset = GoogleDriveLoader._load_onewarmheart(
        tmp_path,
        split=f"organic_preprocess",
    )


def test_metadata_only_load():
    """load_data=False must not build the arrays, and must not raise."""
    for name in ["wheat_lines", "comfile_stroke", "parkinson"]:
        ds = raman_data(name, load_data=False)
        assert ds.spectra is None, name
        assert ds.targets is None, name
        assert ds.info is not None, name
        assert ds.task_type is not None, name


@pytest.mark.skip(reason="Figshare download; run manually.")
def test_comfile_stroke_group_ids():
    dataset = FigshareLoader.load_dataset("comfile_stroke")
    assert dataset.spectra.shape == (4020, 724)
    assert dataset.group_ids is not None
    assert len(set(dataset.group_ids.tolist())) == 20
    assert _groups_are_label_pure(dataset)


@pytest.mark.skip(reason="Figshare download; run manually.")
def test_serum_alzheimer_disease_group_ids():
    dataset = FigshareLoader.load_dataset("serum_alzheimer_disease")
    assert dataset.spectra.shape == (3417, 724)
    assert len(set(dataset.group_ids.tolist())) == 17
    assert _groups_are_label_pure(dataset)


@pytest.mark.skip(reason="Figshare download (~37MB); run manually.")
def test_serum_prostate_cancer_group_ids():
    # 63 files but 62 distinct samples: 1Ag125 ships twice, baseline-corrected
    # and not, so a per-file counter would place one patient in two groups.
    dataset = FigshareLoader.load_dataset("serum_prostate_cancer")
    assert dataset.spectra.shape == (12601, 725)
    assert len(set(dataset.group_ids.tolist())) == 62
    sizes = np.bincount(dataset.group_ids)
    assert sizes.max() == 400  # the duplicated sample
    assert _groups_are_label_pure(dataset)


def test_glioma_slide_table():
    from raman_data.loaders.GoogleDriveLoader import _GLIOMA_SLIDES
    import re
    assert len(_GLIOMA_SLIDES) == 59
    patients = {}
    for subtype, name, _ in _GLIOMA_SLIDES:
        patients.setdefault(re.match(r"HF-(\d+)_", name).group(1), set()).add(subtype)
    assert len(patients) == 46
    assert all(len(s) == 1 for s in patients.values())  # one subtype per patient


def test_glioma_unpickler_refuses_code():
    import io
    import pickle
    from raman_data.loaders.GoogleDriveLoader import _NumpyOnlyUnpickler
    with pytest.raises(pickle.UnpicklingError):
        _NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(os.getcwd))).load()
    arrays = np.empty(2, dtype=object)
    arrays[0], arrays[1] = np.ones((2, 2), dtype=np.int64), np.zeros((3, 1), dtype=np.int64)
    loaded = _NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(arrays))).load()
    assert loaded[1].shape == (3, 1)


@pytest.mark.skip(reason="Google Drive download (~3 GB); run manually.")
def test_glioma_group_ids():
    for name, n_classes in [("glioma_idh", 2), ("glioma_methylation", 6)]:
        dataset = GoogleDriveLoader.load_dataset(name)
        assert dataset.spectra.shape == (257904, 1738), name
        assert len(dataset.target_names) == n_classes, name
        assert len(set(dataset.group_ids.tolist())) == 46, name
        assert _groups_are_label_pure(dataset), name


@pytest.mark.skip(reason="MIND-Lab download; run manually.")
def test_mind_dataset_group_ids():
    expected = {"parkinson": (1476, 60), "alzheimer": (1151, 46),
                "covid19_salvia": (2501, 101)}
    for name, (n_spectra, n_groups) in expected.items():
        dataset = GitHubLoader.load_dataset(name)
        assert dataset.spectra.shape[0] == n_spectra, name
        assert len(set(dataset.group_ids.tolist())) == n_groups, name
        assert _groups_are_label_pure(dataset), name


def test_chlorinated_samples():
    dataset = GitHubLoader.load_dataset("chlorinated_samples")
    assert dataset.target_names == ["no_chloroform", "chloroform"]
    assert dataset.spectra.shape == (230, 2473)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    assert dataset.targets.shape[0] == dataset.spectra.shape[0]
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 76, 1: 154}


def test_raman_spi_sample_stem():
    """Trailing-letter spectrum ids map to their physical sample."""
    from raman_data.loaders.MiscLoader import MiscLoader
    stem = MiscLoader._raspi_sample_stem
    assert stem("PE1a") == "PE1"
    assert stem("PE1b") == "PE1"
    assert stem("PET4c") == "PET4"
    assert stem("PET5d") == "PET5"
    assert stem("PU15") == "PU15"
    assert stem("PVC17") == "PVC17"
    assert stem("a") == "a"  # never strips an id down to empty


@pytest.mark.skip(reason="Borealis download (~10 MB); run manually.")
def test_raman_spi():
    dataset = MiscLoader.load_dataset("raman_spi")
    # 396 fully-labeled spectra (of 402 in the source file), uniform
    # 225-3949 cm^-1 / 1 cm^-1 grid, 14 plastic types, 272 physical samples.
    assert dataset.spectra.shape == (396, 3725)
    assert len(dataset.target_names) == 14
    assert len(set(dataset.group_ids.tolist())) == 272
    assert _groups_are_label_pure(dataset)


@pytest.mark.skip(reason="Borealis download (~530 MB); run manually.")
def test_rampi():
    dataset = MiscLoader.load_dataset("raman_mpi")
    # 16,598 of the 33,119 deposited pixel spectra carry one of the 14
    # plastic labels ('blank' pixels excluded); resampled onto the common
    # 710-1,827 cm^-1 / 1 cm^-1 grid; 34 maps as groups. NOTE: groups are
    # NOT label-pure -- mixed-material maps contain several classes.
    assert dataset.spectra.shape == (16598, 1118)
    assert len(dataset.target_names) == 14
    assert len(set(dataset.group_ids.tolist())) == 34


@pytest.mark.skip(reason="Borealis downloads (~110 MB across 4 variants); run manually.")
def test_lei_microplastics():
    for res, n_points in [("1cm", 1081), ("2cm", 541), ("4cm", 271), ("8cm", 136)]:
        dataset = MiscLoader.load_dataset(f"microplastics_training_{res}")
        # 3,720 of the 4,520 deposited rows (800 synthetic blanks excluded);
        # 14 plastic types; 108 supplier sample ids, each label-pure.
        assert dataset.spectra.shape == (3720, n_points), res
        assert len(dataset.target_names) == 14, res
        assert len(set(dataset.group_ids.tolist())) == 108, res
        assert _groups_are_label_pure(dataset), res


def test_lei_measurement_blocks():
    # The Lei training CSV rows are laid out in 20-row blocks, one per
    # source measurement; a sample measured twice gives a 40-row run of
    # the same sampleid, which must split into two blocks.
    sid = np.array(["A"] * 40 + ["B"] * 20)
    labels = np.array(["polyethylene"] * 40 + ["polypropylene"] * 20)
    blocks = MiscLoader._lei_measurement_blocks(sid, labels)
    assert [(k, s, l) for k, s, l in blocks] == [
        (0, "A", "polyethylene"), (20, "A", "polyethylene"),
        (40, "B", "polypropylene"),
    ]
    # A run not a multiple of the 20-row block is corrupt.
    with pytest.raises(RuntimeError):
        MiscLoader._lei_measurement_blocks(
            np.array(["A"] * 21), np.array(["polyethylene"] * 21)
        )
    # A sampleid run carrying mixed plastic labels is corrupt.
    with pytest.raises(RuntimeError):
        MiscLoader._lei_measurement_blocks(
            np.array(["A"] * 20),
            np.array(["polyethylene"] * 10 + ["polypropylene"] * 10),
        )


@pytest.mark.skip(reason="Borealis download (~59.6 MB, 1cm variant); run manually.")
def test_lei_microplastics_clean():
    dataset = MiscLoader.load_dataset("microplastics_training_clean")
    # One representative per source measurement: 186 measurement blocks
    # (the deposited rows are 20-row blocks per measurement) from 108
    # supplier samples, on the paper's native 1 cm^-1 grid.
    assert dataset.spectra.shape == (186, 1081)
    assert len(dataset.target_names) == 14
    assert len(set(dataset.group_ids.tolist())) == 108
    assert _groups_are_label_pure(dataset)
    assert not np.isnan(dataset.spectra).any()


def test_stroke_urine_sers_static():
    # The embedded patient table must match the paper's cohort: 101
    # participants, 55 healthy controls, 46 ischemic-stroke patients.
    labels = ZenodoLoader._STROKE_URI_PATIENT_LABELS
    assert set(labels) == {str(i) for i in range(1, 102)}
    vals = list(labels.values())
    assert vals.count("HC") == 55
    assert vals.count("IS") == 46
    # Common resampling grid: 485-1,638 cm^-1 at 1 cm^-1.
    grid = ZenodoLoader._STROKE_URI_GRID
    assert grid.shape == (1154,)
    assert grid[0] == 485 and grid[-1] == 1638
    assert (np.diff(grid) == 1.0).all()


@pytest.mark.skip(reason="Zenodo download (128 MB RAR) and a RAR extraction tool (bsdtar/unrar/unar/7z) required; run manually.")
def test_stroke_urine_sers():
    for nm in (532, 633):
        dataset = ZenodoLoader.load_dataset(f"stroke_urine_sers_{nm}")
        # 105 map files (101 participants + 4 repeat collections 31.1-34.1)
        # x 100 spectra per 10x10 map, resampled onto the common grid.
        assert dataset.spectra.shape == (10500, 1154), nm
        assert dataset.target_names == ["HC", "IS"], nm
        assert len(set(dataset.group_ids.tolist())) == 101, nm
        assert _groups_are_label_pure(dataset), nm


def test_breast_cancer_subtypes():
    # Scoped to a single ~11 MB xlsx via file_typ, small enough to run in CI.
    dataset = ZenodoLoader.load_dataset("breast_cancer_subtypes")
    # LabelEncoder class order: dcis, idc, ilc, normal
    assert dataset.target_names == ["dcis", "idc", "ilc", "normal"]
    assert dataset.spectra.shape == (641, 1250)
    assert dataset.spectra.shape[1] == len(dataset.raman_shifts)
    assert dataset.raman_shifts[0] == 500 and dataset.raman_shifts[-1] == 2998
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 221, 1: 114, 2: 129, 3: 177}
    assert dataset.group_ids is not None
    assert dataset.group_ids.shape == (641,)
    # 80 tissue samples (the deposit's finest grouping; see dataset
    # description for the 71-individual note).
    assert len(set(dataset.group_ids.tolist())) == 80
    assert _groups_are_label_pure(dataset)
    assert not np.isnan(dataset.spectra).any()


@pytest.mark.skip(reason="Mendeley Data returns HTTP 403 for GitHub-hosted CI runners; run manually (~10 MB, ~9 s).")
def test_water_quality_ions():
    # Single ~10 MB Mendeley download (4 files).
    dataset = MendeleyLoader.load_dataset("water_quality_ions")
    assert dataset.spectra.shape == (694, 1019)
    # Multi-target regression: 3 concentration columns in mg/L.
    assert dataset.targets.shape == (694, 3)
    assert dataset.target_names == [
        "Nitrate [mg/L]",
        "Nitrite [mg/L]",
        "Sulphate [mg/L]",
    ]
    assert dataset.raman_shifts[0] == 401.026
    assert dataset.raman_shifts[-1] == 3898.75
    # 70 prepared solutions, each measured 9-10 times.
    assert dataset.group_ids is not None
    assert dataset.group_ids.shape == (694,)
    assert len(set(dataset.group_ids.tolist())) == 70
    # 0 mg/L means the analyte is absent; each column has zeros and
    # non-negative values, no NaNs anywhere.
    for i in range(3):
        assert (dataset.targets[:, i] >= 0).all()
        assert (dataset.targets[:, i] == 0).any()
    assert not np.isnan(dataset.spectra).any()
    assert not np.isnan(dataset.targets).any()


def test_yeast_batch_effects():
    # Single 19.7 MB Zenodo download (GitHub repo snapshot), small enough
    # to run in CI (~25 s fresh).
    dataset = ZenodoLoader.load_dataset("yeast_batch_effects")
    assert dataset.spectra.shape == (675, 2048)
    assert dataset.targets.shape == (675,)
    # 9 strains -- 3 S. cerevisiae (BY4741, YNL141, YGL058) and 6
    # S. pombe (SP286, ED666, PDF1, RAD6, DEA2, ARSG) -- 75 spectra each.
    assert dataset.target_names == [
        "ARSG", "BY4741", "DEA2", "ED666", "PDF1", "RAD6",
        "SP286", "YGL058", "YNL141",
    ]
    assert np.all(np.bincount(dataset.targets) == 75)
    assert dataset.raman_shifts[0] == 235.71758360761123
    assert dataset.raman_shifts[-1] == 3653.5482095528914
    # 3 plate groups (one per desiccation date), 225 spectra each -- the
    # paper's batch unit; grouped 3-fold is leave-one-plate-out CV, its
    # protocol. Within a plate: 9 strain wells x 25 co-located site
    # spectra (the 10th well on each plate is an empty control, excluded).
    assert dataset.group_ids is not None
    assert dataset.group_ids.shape == (675,)
    per_group = collections.Counter(dataset.group_ids.tolist())
    assert len(per_group) == 3
    assert sorted(per_group.values()) == [225] * 3
    # Every plate carries all 9 strains, so leave-one-plate-out keeps all
    # classes in training (groups are intentionally NOT label-pure).
    for g in per_group:
        assert len(np.unique(dataset.targets[dataset.group_ids == g])) == 9
    assert not np.isnan(dataset.spectra).any()


@pytest.mark.skip(reason="Mendeley Data returns HTTP 403 for GitHub-hosted CI runners; run manually (~10 MB, ~10 s).")
def test_single_cell_prokaryotes():
    dataset = MendeleyLoader.load_dataset("single_cell_prokaryotes")
    assert dataset.spectra.shape == (782, 2333)
    assert dataset.targets.shape == (782,)
    # 6 species (3 bacteria + 3 archaea); per-species counts include the
    # mixed-culture cells (21 Bacillus, 21 Nitrososphaera, 20 Thermus).
    assert dataset.target_names == [
        "Bacillus_subtilis", "Escherichia_coli", "Nitrososphaera_viennensis",
        "Sulfolobus_acidocaldarius", "Thermococcus_kodakarensis",
        "Thermus_thermophilus",
    ]
    counts = collections.Counter(dataset.targets.tolist())
    assert counts == {
        dataset.target_names.index("Bacillus_subtilis"): 141,
        dataset.target_names.index("Escherichia_coli"): 120,
        dataset.target_names.index("Nitrososphaera_viennensis"): 141,
        dataset.target_names.index("Sulfolobus_acidocaldarius"): 120,
        dataset.target_names.index("Thermococcus_kodakarensis"): 120,
        dataset.target_names.index("Thermus_thermophilus"): 140,
    }
    # Common grid: 710-3,042 cm^-1 at 1 cm^-1 (per-session axes are
    # interpolated onto it).
    assert dataset.raman_shifts[0] == 710.0
    assert dataset.raman_shifts[-1] == 3042.0
    # 19 acquisition sessions: 18 species x date directories of 40 cells
    # (label-pure) + 1 mixed-culture session of 62 (multi-label by nature:
    # one culture, three species).
    g = dataset.group_ids
    assert g is not None and g.shape == (782,)
    per_group = collections.Counter(g.tolist())
    assert len(per_group) == 19
    assert sorted(per_group.values()) == [40] * 18 + [62]
    for name, n in per_group.items():
        if n == 40:
            assert len(np.unique(dataset.targets[g == name])) == 1
    mix = [name for name, n in per_group.items() if n == 62][0]
    assert "mixture" in mix
    assert len(np.unique(dataset.targets[g == mix])) == 3
    assert not np.isnan(dataset.spectra).any()


def test_yeast_bioreactor_fermentation():
    # Single 15.5 MB 4TU.ResearchData download, small enough to run in CI
    # (~10 s fresh).
    dataset = MiscLoader.load_dataset("yeast_bioreactor_fermentation")
    assert dataset.spectra.shape == (65, 3326)
    assert dataset.targets.shape == (65, 3)
    assert dataset.target_names == ["Glucose [mM]", "Ethanol [mM]", "Biomass [g/L]"]
    assert dataset.raman_shifts[0] == 100.0
    assert dataset.raman_shifts[-1] == 3425.0
    # 5 fermentations: 3 calibration batches (B1-B3), 1 validation batch
    # (B4), 1 fed-batch transfer test (FB).
    g = dataset.group_ids
    assert g is not None and g.shape == (65,)
    assert collections.Counter(g.tolist()) == {
        "B1": 12, "B2": 13, "B3": 13, "B4": 13, "FB": 14,
    }
    # 5 failed HPLC glucose references (NaN); the other targets complete.
    assert int(np.isnan(dataset.targets[:, 0]).sum()) == 5
    assert not np.isnan(dataset.targets[:, 1:]).any()
    assert not np.isnan(dataset.spectra).any()


def test_nanoplastics_individual():
    # Single 2 MB figshare download, small enough to run in CI.
    dataset = FigshareLoader.load_dataset("nanoplastics_individual")
    assert dataset.spectra.shape == (1000, 467)
    assert dataset.targets.shape == (1000,)
    assert dataset.target_names == ["PE", "PMMA", "PS", "PTFE", "PVC"]
    assert len(dataset.raman_shifts) == 467
    assert dataset.raman_shifts[0] == 500.0
    assert dataset.raman_shifts[-1] == 1799.0
    # 200 individual particles per polymer.
    for i, name in enumerate(dataset.target_names):
        assert int((dataset.targets == i).sum()) == 200, name
    # One spectrum per particle -- no grouping structure.
    assert dataset.group_ids is None
    assert not np.isnan(dataset.spectra).any()
    assert (dataset.spectra > 0).all()


def test_nist_ecsers_filename_parser():
    # No network: the filename parser must handle all 131 deposited
    # names, including the three irregular ones (a space-separated
    # 'Spectrum 293', a 'nontargeT' typo, and a file without a target
    # marker).
    parse = MiscLoader._nist_ecsers_parse_name
    files = [f for f, _, _ in MiscLoader._NIST_ECSERS_FILES]
    assert len(files) == 131

    val = parse("Valeryl_fentanyl_25ppm_nontarget_Spectrum 293.csv")
    assert val == {
        "substance": "Valeryl_fentanyl", "conc": 25, "form": None,
        "target": "nontarget", "rep": None, "spectrum": 293,
    }
    fent = parse("Fentanyl_1ppm_nontargeT_rep3_Spectrum_5.csv")
    assert fent["substance"] == "Fentanyl"
    assert fent["target"] == "nontarget" and fent["rep"] == 3
    nalt = parse("Naltrexone_1000ppm_Spectrum_7.csv")
    assert nalt["substance"] == "Naltrexone"
    assert nalt["conc"] == 1000 and nalt["target"] is None
    proc = parse("Procaine_form_solid_nontarget_rep1_Spectrum_7.csv")
    assert proc["substance"] == "Procaine"
    assert proc["form"] == "solid" and proc["conc"] is None and proc["rep"] == 1

    meta = [parse(f) for f in files]
    # 32 substances; 126/131 carry a concentration label.
    substances = {m["substance"] for m in meta}
    assert len(substances) == 32
    assert sum(m["conc"] is not None for m in meta) == 126
    assert {m["conc"] for m in meta if m["conc"] is not None} == {
        1, 5, 10, 25, 50, 100, 300, 333, 1000, 1001, 1013,
    }
    # 118 (substance, concentration, replicate) groups; the 24
    # measurements without a replicate marker share one group per
    # substance+concentration (possibly the same acquisition session).
    groups = {
        (m["substance"],
         m["conc"] if m["conc"] is not None else (m["form"] or "na"),
         m["rep"] if m["rep"] is not None else "unmarked")
        for m in meta
    }
    assert len(groups) == 118
    assert sum(m["rep"] is None for m in meta) == 24


@pytest.mark.skip(reason="data.nist.gov is flaky (504s/timeouts around 2026-10-19/20); run manually (131 files, ~4.1 MB, ~10 min).")
def test_nist_ecsers_drugs():
    dataset = MiscLoader.load_dataset("nist_ecsers_drugs")
    assert dataset.spectra.shape == (131, 1013)
    assert dataset.targets.shape == (131,)
    assert len(dataset.target_names) == 32
    # Common grid: the dominant 1,014-point batch's axis minus its
    # single point below the other batch's range (33.8 cm^-1).
    assert dataset.raman_shifts[0] == pytest.approx(36.594629)
    assert dataset.raman_shifts[-1] == pytest.approx(2859.07796)
    # Per-class spectrum counts (32 substances).
    counts = collections.Counter(dataset.targets.tolist())
    assert counts == {
        dataset.target_names.index(name): n
        for name, n in {
            "Alprazolam": 13, "Fentanyl": 13, "Cocaine": 12,
            "Heroin": 10, "Codeine": 9, "Levamisole": 7,
            "Methamphetamine": 5, "Morphine": 5, "Sufentanil": 5,
            "Dextromethorphan": 4, "Diphenhydramine": 4,
            "Etizolam": 3, "Etonitazene": 3, "Lidocaine": 3,
            "Methadone": 3, "Nicotine": 3,
            "Npiperidinylmetonitazene": 3, "Npyrrolidinoetonitazene": 3,
            "Procaine": 3, "Quinine": 3, "Temazepam": 3,
            "Xylazine": 3, "Diltiazem": 2, "4-ANPP": 1,
            "Acetyl_fentanyl": 1, "Acryl_fentanyl": 1,
            "Buprenorphine": 1, "Furanyl_fentanyl": 1,
            "Hydroxyzine": 1, "Methoxyacetyl_fentanyl": 1,
            "Naltrexone": 1, "Valeryl_fentanyl": 1,
        }.items()
    }
    # 118 (substance, concentration, replicate) groups: 106
    # single-spectrum, 11 two-spectrum, one three-spectrum.
    g = dataset.group_ids
    assert g is not None and g.shape == (131,)
    per_group = collections.Counter(g.tolist())
    assert len(per_group) == 118
    assert sorted(per_group.values()) == [1] * 106 + [2] * 11 + [3]
    # Raw dark-subtracted counts (NOT normalized): every file has
    # slightly negative points, maxima reach ~2e5.
    assert dataset.spectra.min() < 0
    assert 2e5 < dataset.spectra.max() < 2.1e5
    assert not np.isnan(dataset.spectra).any()


def test_rod_parse_cif():
    # No network: the minimal CIF parser must handle the value forms
    # that occur in the .rod entry files -- same-line plain and
    # quoted values, same-line and next-line ';'-blocks (with and
    # without inline content after the opening ';'), next-line quoted
    # values, and loop blocks.
    cif = """# comment line
data_1000000
_publ_author_name 'El Mendili, Y.'
_chemical_name_mineral Magnetite
_chemical_name_systematic 'Iron diiron(III) oxide'
_chemical_formula_sum
'C2 H2.64 Ba0.01 Ca0.02 Ce0.98 F0.14 O7.74 Sr0.98'
_publ_section_title
;
Multi-line
title text
;
_tag_block_inline ;first line of block
second line of block
;
loop_
_raman_spectrum.raman_shift
_raman_spectrum.intensity
91.5 1155
96.3 1139
101.1 1208
loop_
_raman_theoretical_spectrum.raman_shift
_raman_theoretical_spectrum.intensity
169.8 0.0
200.1 12.5
"""
    values, loops = MiscLoader._rod_parse_cif(cif)
    assert values["_publ_author_name"] == "El Mendili, Y."
    assert values["_chemical_name_mineral"] == "Magnetite"
    assert values["_chemical_name_systematic"] == "Iron diiron(III) oxide"
    assert values["_chemical_formula_sum"] == (
        "C2 H2.64 Ba0.01 Ca0.02 Ce0.98 F0.14 O7.74 Sr0.98"
    )
    assert values["_publ_section_title"] == "Multi-line\ntitle text"
    assert values["_tag_block_inline"] == (
        "first line of block\nsecond line of block"
    )
    assert len(loops) == 2
    cols, rows = loops[0]
    assert cols == [
        "_raman_spectrum.raman_shift",
        "_raman_spectrum.intensity",
    ]
    assert rows == ["91.5 1155", "96.3 1139", "101.1 1208"]
    cols2, rows2 = loops[1]
    assert cols2 == [
        "_raman_theoretical_spectrum.raman_shift",
        "_raman_theoretical_spectrum.intensity",
    ]
    assert rows2 == ["169.8 0.0", "200.1 12.5"]


def test_rod_minerals():
    dataset = MiscLoader.load_dataset("rod_minerals")
    # 1,098 experimental spectra on the common 250-1,200 cm^-1 @1
    # grid (951 points): the snapshot's 1,133 entries minus 18
    # theoretical-only entries and the 17 experimental spectra that
    # do not fully cover the common window; every remaining entry
    # resolves a label, so no nameless spectra are excluded.
    assert dataset.spectra.shape == (1098, 951)
    assert dataset.raman_shifts[0] == pytest.approx(250.0)
    assert dataset.raman_shifts[-1] == pytest.approx(1200.0)
    assert len(dataset.target_names) == 432
    # group = label: one group per mineral/phase (432 groups),
    # label-pure.
    g = dataset.group_ids
    assert g is not None and g.shape == (1098,)
    assert len(set(g.tolist())) == 432
    assert _groups_are_label_pure(dataset)
    # Label counts: 108 single-spectrum, 213 two-spectrum, 111
    # three-or-more.
    names = list(dataset.target_names)
    counts = collections.Counter(
        names[i] for i in np.asarray(dataset.targets).ravel().tolist()
    )
    assert counts["Andradite"] == 22
    assert counts["Corundum"] == 17
    assert counts["Anorthite"] == 15
    assert sum(1 for c in counts.values() if c == 1) == 108
    assert sum(1 for c in counts.values() if c == 2) == 213
    # Raw as-deposited intensities: arbitrary units, per-entry
    # normalization unknown; some entries have negative baseline
    # points, maxima reach ~3.2e5.
    assert dataset.spectra.min() < 0
    assert 3.2e5 < dataset.spectra.max() < 3.22e5
    assert not np.isnan(dataset.spectra).any()


def test_glaucoma_enigma_baseline():
    dataset = MendeleyLoader.load_dataset("glaucoma_enigma_baseline")
    # 101 baseline sessions of the 21 ENIGMA patients on the common
    # 401-3,999 cm^-1 @1 grid (3,599 points); 11 targets per patient.
    assert dataset.spectra.shape == (101, 3599)
    assert dataset.raman_shifts[0] == pytest.approx(401.0)
    assert dataset.raman_shifts[-1] == pytest.approx(3999.0)
    assert dataset.targets.shape == (101, 11)
    assert "GCC_6_glauc_severity" in dataset.target_names
    # group = patient: 21 groups, 4-9 sessions each (ET015 has 9
    # across the two baseline folders).
    g = dataset.group_ids
    assert g is not None and g.shape == (101,)
    per_group = collections.Counter(g.tolist())
    assert len(per_group) == 21
    assert min(per_group.values()) == 4
    assert max(per_group.values()) == 9
    # Targets are per patient: all sessions of one patient share the
    # same target row.
    g = np.asarray(g)
    for p in np.unique(g):
        assert np.allclose(dataset.targets[g == p], dataset.targets[g == p][0])
    # Raw LabSpec counts as deposited.
    assert dataset.spectra.min() > 0
    assert not np.isnan(dataset.spectra).any()
    assert not np.isnan(dataset.targets).any()


def test_glaucoma_enigma_18m():
    dataset = MendeleyLoader.load_dataset("glaucoma_enigma_18m")
    # 100 18-month sessions of the 23 ENIGMA patients (4 each, except
    # ET043/ET048/ET049 with 6/8/6 incl. the '_1' re-taken exports);
    # 11 targets per patient.
    assert dataset.spectra.shape == (100, 3599)
    assert dataset.raman_shifts[0] == pytest.approx(401.0)
    assert dataset.raman_shifts[-1] == pytest.approx(3999.0)
    assert dataset.targets.shape == (100, 11)
    assert "GCC_6calculate_18M" in dataset.target_names
    # group = patient: 23 groups, 4-8 sessions each.
    g = dataset.group_ids
    assert g is not None and g.shape == (100,)
    per_group = collections.Counter(g.tolist())
    assert len(per_group) == 23
    assert min(per_group.values()) == 4
    assert max(per_group.values()) == 8
    g = np.asarray(g)
    for p in np.unique(g):
        assert np.allclose(dataset.targets[g == p], dataset.targets[g == p][0])
    assert dataset.spectra.min() > 0
    assert not np.isnan(dataset.spectra).any()
    assert not np.isnan(dataset.targets).any()


def test_liver_cancer_raman():
    dataset = GitHubLoader.load_dataset("liver_cancer_raman")
    # The 6,000-spectrum demo of Huang et al. 2023 (4,000 train +
    # 1,000 validate + 1,000 test files of the authors' repo), binary
    # carcinoma vs paracarcinoma, 500-2,000 cm^-1 on the paper's
    # canonical 889-point grid (the 892-column demo files are
    # resampled onto it).
    assert dataset.spectra.shape == (6000, 889)
    assert dataset.raman_shifts[0] == pytest.approx(500.0)
    assert dataset.raman_shifts[-1] == pytest.approx(2000.0)
    assert dataset.target_names == ["paracarcinoma", "carcinoma"]
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert counts == {0: 3000, 1: 3000}
    # 100 sample groups of 50 (the train/validate files are ordered in
    # label-pure 50-spectrum blocks) + 1,000 singletons from the
    # authors' shuffled test file.
    g = dataset.group_ids
    assert g is not None and g.shape == (6000,)
    per_group = collections.Counter(g.tolist())
    assert len(per_group) == 1100
    assert sorted(per_group.values()) == [1] * 1000 + [50] * 100
    assert _groups_are_label_pure(dataset)
    # Raw as-deposited intensities: arbitrary units, negative
    # dark-subtracted points down to ~-1.1e3, maxima ~6.2e3.
    assert dataset.spectra.min() < -1000
    assert 6100 < dataset.spectra.max() < 6300
    assert not np.isnan(dataset.spectra).any()


def test_infant_formula_raman():
    dataset = MendeleyLoader.load_dataset("infant_formula_raman")
    # 373 spectra of powdered infant formula products: 24 product
    # types = 4 de-identified brands x 3 formula stages x
    # online/offline purchase channel (Zhang 2020, Mendeley
    # 4j8bj5vv52 v2). Native 1,738-point axis (CSV column headers),
    # no resampling.
    assert dataset.spectra.shape == (373, 1738)
    assert dataset.raman_shifts[0] == pytest.approx(50.15215)
    assert dataset.raman_shifts[-1] == pytest.approx(3399.899)
    assert len(dataset.target_names) == 24
    # Class names encode brand_stage-channel, e.g. A_stage1_offline.
    pattern = re.compile(r"^[ABNQ]_stage[123]_(online|offline)$")
    assert all(pattern.match(n) for n in dataset.target_names)
    assert dataset.target_names[0] == "A_stage1_offline"
    assert dataset.target_names[1] == "A_stage1_online"
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert set(counts) == set(range(24))
    assert sum(counts.values()) == 373
    # The two Q stage-1 types have the fewest spectra; B stage-3
    # offline the most.
    assert min(counts.values()) == 5
    assert max(counts.values()) == 24
    # No sample/bottle identifiers in the deposit: ungrouped.
    assert dataset.group_ids is None
    # Raw as-deposited intensities: arbitrary units, negative points
    # down to ~-1.8e3, maxima ~1.1e4.
    assert dataset.spectra.min() < -1800
    assert dataset.spectra.max() > 11000
    assert not np.isnan(dataset.spectra).any()
    assert not np.isnan(dataset.targets).any()


def test_sers_gna_bacteria():
    dataset = GitHubLoader.load_dataset("sers_gna_bacteria")
    # 117 raw SERS spectra (9 strains x 13) on a gold nanorod array
    # (Gulmus 2026), 785 nm, native 1,024-point axis (identical in
    # all 117 WITec instrument files, non-uniform 1.44-2.04 cm^-1
    # spacing).
    assert dataset.spectra.shape == (117, 1024)
    assert dataset.raman_shifts[0] == pytest.approx(247.990908, abs=1e-3)
    assert dataset.raman_shifts[-1] == pytest.approx(1997.905099, abs=1e-3)
    d = np.diff(dataset.raman_shifts)
    assert (d > 1.0).all() and (d < 2.1).all()
    expected = {
        "E_coli_29922", "E_coli_BAA-2523", "E_coli_K4",
        "P_aeruginosa_10145", "P_aeruginosa_27853", "P_aeruginosa_9027",
        "S_aureus_25923", "S_aureus_43300", "S_aureus_6538",
    }
    assert set(dataset.target_names) == expected
    counts = dict(zip(*np.unique(dataset.targets, return_counts=True)))
    assert len(counts) == 9
    assert all(c == 13 for c in counts.values())
    # All 13 spectra of a strain come from one droplet on one GNA
    # substrate of one culture: group = strain (the paper's
    # leave-strain-out validation).
    g = dataset.group_ids
    per_group = collections.Counter(g.tolist())
    assert set(per_group) == expected
    assert all(c == 13 for c in per_group.values())
    assert _groups_are_label_pure(dataset)
    # Raw instrument counts: strictly positive, ~4.2e3-1.45e4.
    assert dataset.spectra.min() > 0
    assert dataset.spectra.max() < 15000
    assert not np.isnan(dataset.spectra).any()


def test_sers_pesticide_produce():
    dataset = GitHubLoader.load_dataset("sers_pesticide_produce")
    # 33 SERS spectra of pesticide-spiked fresh produce (Hegde 2025,
    # SERSFormer2): a spinach control, a spinach carbophenothion+oxamyl
    # 50:50 mixture and a strawberry carbophenothion+phosmet 50:50
    # mixture, 3 replicate spectra per condition. Common 501-1,999
    # cm^-1 @1 grid (1,499 points).
    assert dataset.spectra.shape == (33, 1499)
    assert dataset.raman_shifts[0] == pytest.approx(501.0)
    assert dataset.raman_shifts[-1] == pytest.approx(1999.0)
    assert np.allclose(np.diff(dataset.raman_shifts), 1.0)
    # Multi-output regression: per-pesticide concentration (ppm) for
    # the three pesticides present in the public deposit.
    assert dataset.targets.shape == (33, 3)
    assert list(dataset.target_names) == [
        "carbophenothion", "oxamyl", "phosmet"
    ]
    # 11 conditions (produce x mixture x concentration level), 3
    # replicate spectra each.
    g = np.asarray(dataset.group_ids)
    assert g.shape == (33,)
    per_group = collections.Counter(g.tolist())
    assert len(per_group) == 11
    assert all(c == 3 for c in per_group.values())
    # All spectra of a condition share the same target row.
    for c in np.unique(g):
        assert np.allclose(dataset.targets[g == c], dataset.targets[g == c][0])
    # The 50:50 folders halve the nominal level, so the per-pesticide
    # levels are 0.25/0.5/1.0/2.5/5.0 ppm (0 = absent).
    t = dataset.targets
    levels = {0.25, 0.5, 1.0, 2.5, 5.0}
    for j in range(3):
        nz = t[:, j][t[:, j] > 0]
        assert set(np.round(nz, 3)) == levels
    # carbophenothion is in both mixtures (30), oxamyl spinach only
    # (15), phosmet strawberry only (15).
    assert int((t[:, 0] > 0).sum()) == 30
    assert int((t[:, 1] > 0).sum()) == 15
    assert int((t[:, 2] > 0).sum()) == 15
    # The spinach control is the all-zero row.
    assert int((t.sum(axis=1) == 0).sum()) == 3
    # Raw as-deposited intensities: arbitrary units, negative
    # baseline-subtracted points down to ~-17, maxima ~7.8e3.
    assert dataset.spectra.min() < 0
    assert dataset.spectra.max() < 8000
    assert not np.isnan(dataset.spectra).any()
    assert not np.isnan(dataset.targets).any()


def test_aged_oils_raman_1a():
    # Single ~1 MB HF-mirror download, small enough to run in CI.
    dataset = MendeleyLoader.load_dataset("aged_oils_raman_1a")
    # Data Set 1A (2016 LBNL campaign): one spectrum per bottle;
    # 100 deposited rows minus one all-NaN placeholder = 99 spectra
    # on the native 1,340-point -18.1-1,974.2 cm^-1 axis (as is).
    assert dataset.spectra.shape == (99, 1340)
    assert dataset.raman_shifts[0] == pytest.approx(-18.087)
    assert dataset.raman_shifts[-1] == pytest.approx(1974.170)
    assert np.all(np.diff(dataset.raman_shifts) > 0)
    # Single-target regression: peroxide value by titration.
    assert dataset.targets.shape == (99,)
    assert dataset.target_names == ["peroxide_value"]
    pv = dataset.targets
    assert not np.isnan(pv).any()
    assert pv.min() == pytest.approx(1.5240, abs=0.01)
    assert 1.0 < pv.min() and pv.max() < 170.0
    # One spectrum per bottle: no group structure.
    assert dataset.group_ids is None
    # Raw counts as deposited (arbitrary units).
    assert dataset.spectra.min() > 27000
    assert dataset.spectra.max() < 22e6
    assert not np.isnan(dataset.spectra).any()


def test_aged_oils_raman_2():
    # Single ~1.6 MB HF-mirror download, small enough to run in CI.
    dataset = MendeleyLoader.load_dataset("aged_oils_raman_2")
    # Data Set 2 (University of Delaware campaign): 215 replicate
    # spectra of 53 bottles (no sample ids in the deposit) on the
    # native 1,044-point 18.0-2,087.9 cm^-1 axis (as is).
    assert dataset.spectra.shape == (215, 1044)
    assert dataset.raman_shifts[0] == pytest.approx(17.983)
    assert dataset.raman_shifts[-1] == pytest.approx(2087.858)
    assert np.all(np.diff(dataset.raman_shifts) > 0)
    assert dataset.targets.shape == (215,)
    assert dataset.target_names == ["peroxide_value"]
    pv = dataset.targets
    assert not np.isnan(pv).any()
    assert pv.min() == pytest.approx(0.46, abs=0.01)
    assert 0.4 < pv.min() and pv.max() < 81.0
    # Replicates of the 53 bottles, but no per-spectrum sample ids
    # are in the deposit, so no group structure is shipped.
    assert dataset.group_ids is None
    # Different probe from Data Set 1A: lower counts, some negative
    # points.
    assert dataset.spectra.min() < 0
    assert dataset.spectra.max() < 42000
    assert not np.isnan(dataset.spectra).any()


def test_olive_sunflower_mixtures():
    # Single ~2.2 MB direct data.gouv.fr download, small enough to
    # run in CI (static CDN, no WAF).
    dataset = MiscLoader.load_dataset("olive_sunflower_mixtures")
    # 605 spectra on the native 710-point 500.5-2,499.3 cm^-1 axis
    # (2.2-3.6 cm^-1 non-uniform spacing, as is).
    assert dataset.spectra.shape == (605, 710)
    assert dataset.raman_shifts[0] == pytest.approx(500.48)
    assert dataset.raman_shifts[-1] == pytest.approx(2499.305)
    assert np.all(np.diff(dataset.raman_shifts) > 0)
    # Single-target regression: EVOO percentage of the blend.
    assert dataset.targets.shape == (605,)
    assert dataset.target_names == ["evoo_percent"]
    ev = dataset.targets
    assert not np.isnan(ev).any()
    assert ev.min() == 0.0 and ev.max() == 100.0
    # 26 distinct blend levels (0, 10, 15, ..., 91, 100).
    assert len(set(ev.tolist())) == 26
    # Group = (date, EVOO %) physical blend: 5 scans per blend
    # (S1-S5), 10 blends re-measured once and 1 twice the same day
    # -> 109 groups of 5/10/15 spectra.
    assert dataset.group_ids is not None
    assert dataset.group_ids.shape == (605,)
    sizes = collections.Counter(dataset.group_ids.tolist())
    assert len(sizes) == 109
    assert set(sizes.values()) == {5, 10, 15}
    assert sum(sizes.values()) == 605
    # All spectra of one blend share its EVOO % target.
    by_group = collections.defaultdict(list)
    for i, gid in enumerate(dataset.group_ids.tolist()):
        by_group[gid].append(ev[i])
    for vals in by_group.values():
        assert np.allclose(np.array(vals), vals[0])
    # Raw counts as deposited (arbitrary units).
    assert dataset.spectra.min() > 600
    assert dataset.spectra.max() < 13000
    assert not np.isnan(dataset.spectra).any()


def test_cwa_tep():
    # Single ~2.3 MB HF-mirror download + a few seconds of xlsx
    # parsing, small enough to run in CI.
    dataset = MendeleyLoader.load_dataset("cwa_tep")
    # 683 raw 1064 nm FT-Raman spectra of TEP solutions at 11
    # concentration levels, on the native 512-point 36.84-2,546.1
    # cm^-1 axis (as is).
    assert dataset.spectra.shape == (683, 512)
    assert dataset.raman_shifts[0] == pytest.approx(36.844, abs=0.01)
    assert dataset.raman_shifts[-1] == pytest.approx(2546.084, abs=0.01)
    assert np.all(np.diff(dataset.raman_shifts) > 0)
    # Single-target regression: fractional concentration (1.0 =
    # pure compound).
    assert dataset.targets.shape == (683,)
    assert dataset.target_names == ["tep_concentration"]
    conc = dataset.targets
    assert not np.isnan(conc).any()
    assert conc.min() == 0.005 and conc.max() == 1.0
    assert len(set(conc.tolist())) == 11
    # No group structure (Number is a global spectrum index).
    assert dataset.group_ids is None
    # Raw 16-bit counts (7 pure-compound spectra hit the ceiling).
    assert dataset.spectra.min() > 10000
    assert dataset.spectra.max() == 65535
    assert int((dataset.spectra == 65535).any(axis=1).sum()) == 7
    assert not np.isnan(dataset.spectra).any()


def test_cwa_dimp():
    # Single ~2.4 MB HF-mirror download + a few seconds of xlsx
    # parsing, small enough to run in CI.
    dataset = MendeleyLoader.load_dataset("cwa_dimp")
    assert dataset.spectra.shape == (723, 512)
    assert dataset.raman_shifts[0] == pytest.approx(36.844, abs=0.01)
    assert dataset.raman_shifts[-1] == pytest.approx(2546.084, abs=0.01)
    assert np.all(np.diff(dataset.raman_shifts) > 0)
    assert dataset.targets.shape == (723,)
    assert dataset.target_names == ["dimp_concentration"]
    conc = dataset.targets
    assert not np.isnan(conc).any()
    assert conc.min() == 0.005 and conc.max() == 1.0
    assert len(set(conc.tolist())) == 11
    assert dataset.group_ids is None
    assert dataset.spectra.min() > 14000
    assert dataset.spectra.max() == 65535
    assert int((dataset.spectra == 65535).any(axis=1).sum()) == 8
    assert not np.isnan(dataset.spectra).any()


def test_cwa_dmmp():
    # Single ~2.4 MB HF-mirror download + a few seconds of xlsx
    # parsing, small enough to run in CI.
    dataset = MendeleyLoader.load_dataset("cwa_dmmp")
    assert dataset.spectra.shape == (717, 512)
    assert dataset.raman_shifts[0] == pytest.approx(36.844, abs=0.01)
    assert dataset.raman_shifts[-1] == pytest.approx(2546.084, abs=0.01)
    assert np.all(np.diff(dataset.raman_shifts) > 0)
    assert dataset.targets.shape == (717,)
    assert dataset.target_names == ["dmmp_concentration"]
    conc = dataset.targets
    assert not np.isnan(conc).any()
    assert conc.min() == 0.005 and conc.max() == 1.0
    assert len(set(conc.tolist())) == 11
    assert dataset.group_ids is None
    assert dataset.spectra.min() > 13000
    # The detector saturates at the 16-bit ceiling for all spectra
    # of the 0.1, 0.75 and 1.0 levels (88 + 58 + 52).
    assert dataset.spectra.max() == 65535
    assert int((dataset.spectra == 65535).any(axis=1).sum()) == 198
    assert not np.isnan(dataset.spectra).any()

