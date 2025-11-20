"""
A general checkup of loader's implementation.
"""

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.KagLoader import KagLoader
from raman_data.loaders.HugLoader import HugLoader
from raman_data.loaders.ZenLoader import ZenLoader

__LOADERS = [
    KagLoader,
    HugLoader,
    ZenLoader
]

def test_interfacing():
    for loader in __LOADERS:
        # This includes ILoader's __subclasshook__ method
        assert issubclass(loader, ILoader)
        assert hasattr(loader, 'DATASETS')

@pytest.mark.skipif(os.environ.get('CI') is not None, reason="Zenodo dataset is huge for CI")
def test_zen_loader_download():
    # Using a known dataset ID from ZenLoader.DATASETS
    dataset_id = list(ZenLoader.DATASETS.keys())[0]
    download_dir = ZenLoader.download_dataset(dataset_id=dataset_id, dataset_name=dataset_id) # TODO check if dataset_name should be dataset_id
    assert download_dir is not None
    assert os.path.isdir(download_dir)
    assert len(os.listdir(download_dir)) > 0

@pytest.mark.skipif(os.environ.get('CI') is not None, reason="Zenodo dataset is huge for CI")
def test_zen_loader_load():
    # Using a known dataset ID from ZenLoader.DATASETS
    dataset_id = list(ZenLoader.DATASETS.keys())[0]
    download_dir = ZenLoader.load_dataset(dataset_id=dataset_id, dataset_name=dataset_id) # TODO check if dataset_name should be dataset_id
    assert download_dir is not None
    assert os.path.isdir(download_dir)
    assert len(os.listdir(download_dir)) > 0
