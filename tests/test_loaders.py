"""
A general checkup of loader's implementation.
"""

from raman_data import raman_data
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.KaggleLoader import KaggleLoader
from raman_data.loaders.HuggingFaceLoader import HuggingFaceLoader
from raman_data.loaders.ZenodoLoader import ZenodoLoader
from raman_data.loaders.MiscLoader import MiscLoader
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

