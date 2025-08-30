from abc import ABCMeta, abstractmethod
from typing import Optional

from numpy import ndarray

class ILoader(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'download_dataset') and 
                callable(subclass.download_dataset) and
                hasattr(subclass, 'load_dataset') and 
                callable(subclass.load_dataset) or
                NotImplementedError)
    
    @abstractmethod
    def download_dataset(
        dataset_name: str,
        file_name: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def load_dataset(
        dataset_name: str,
        file_name: str,
        cache_dir: Optional[str] = None
    ) -> ndarray:
        raise NotImplementedError
    
