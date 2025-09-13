from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.KagLoader import KagLoader

__LOADERS = [
    KagLoader
]

def test_interfacing():
    for loader in __LOADERS:
        assert issubclass(loader, ILoader)
        assert hasattr(loader, 'DATASETS')
