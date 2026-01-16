"""
A general checkup of loader's implementation.
"""

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.KaggleLoader import KaggleLoader
from raman_data.loaders.HuggingFaceLoader import HuggingFaceLoader
from raman_data.loaders.ZenodoLoader import ZenodoLoader
from raman_data.loaders.MiscLoader import MiscLoader
import pytest
import os

__LOADERS = [
    KaggleLoader,
    HuggingFaceLoader,
    ZenodoLoader,
    MiscLoader
]

def test_interfacing():
    for loader in __LOADERS:
        # This includes BaseLoader's __subclasshook__ method
        assert issubclass(loader, BaseLoader)
        assert hasattr(loader, 'DATASETS')

@pytest.mark.skipif(os.environ.get('CI') is not None, reason="Zenodo dataset is huge for CI")
def test_zen_loader_download():
    # Using a known dataset ID from ZenodoLoader.DATASETS
    test_dataset_name = list(ZenodoLoader.DATASETS.keys())[0]
    download_dir = ZenodoLoader.download_dataset(dataset_name=test_dataset_name)
    assert download_dir is not None
    assert os.path.isdir(download_dir)
    assert len(os.listdir(download_dir)) > 0

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

def test_misc_loader_download():
    # Only checks that the download_dataset returns a path and creates the info file
    for name in MiscLoader.get_dataset_names():
        path = MiscLoader.download_dataset(name)
        assert path is not None
        assert os.path.exists(os.path.join(path, "onedrive_url.txt"))

def test_misc_loader_load_missing():
    # Should print missing file warning and return None if files are missing
    dataset = MiscLoader.load_dataset("deepr_denoising", cache_path="/tmp/nonexistent_misc")
    assert dataset is None
