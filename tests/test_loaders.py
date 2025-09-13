"""
A general checkup of loader's implementation.
"""

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.KagLoader import KagLoader

__LOADERS = [
    KagLoader
]

def test_interfacing():
    for loader in __LOADERS:
        # This includes ILoader's __subclasshook__ method
        assert issubclass(loader, ILoader)
        assert hasattr(loader, 'DATASETS')
