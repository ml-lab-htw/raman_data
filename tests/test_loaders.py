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
