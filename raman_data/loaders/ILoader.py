from abc import ABCMeta, abstractmethod
from typing import Optional

from raman_data.types import RamanDataset

class ILoader(metaclass=ABCMeta):
    """
    The general interface of all loaders.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Checks whether a subclass has needed properties.

        Args:
            subclass (class): A class to check inheritance of.

        Returns:
            bool: True, if the subclass has required properties.
                  False otherwise.
        """
        if not (hasattr(subclass, 'download_dataset') and
            callable(subclass.download_dataset) and
            hasattr(subclass, 'load_dataset') and
            callable(subclass.load_dataset)):
            return False
        
        try:
            subclass.download_dataset('')
            subclass.load_dataset('', '')
        except NotImplementedError:
            return False
        
        return True


    @abstractmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> str | None:
        """
        Downloads certain dataset into a predefined cache folder.

        Args:
            dataset_name (str): The name of a dataset to download.
            cache_path (str, optional): The path to save the dataset to.
                                        If None, uses the lastly saved path.

        Raises:
            NotImplementedError: If not implemented raises the error by default.

        Returns:
            str|None: The path the dataset is downloaded to.
                      If the dataset isn't on the list of a loader,
                      returns None.
        """
        raise NotImplementedError


    @abstractmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> RamanDataset | None:
        """
        Loads certain dataset from cache folder.
        If the dataset isn't in the cache folder, downloads it into that folder.

        Args:
            dataset_name (str): The name of a dataset.
            cache_path (str, optional): The path to the dataset's folder.
                                        If None, uses the lastly saved path.
                                        If "default", sets the default path ('~/.cache').

        Raises:
            NotImplementedError: If not implemented raises the error by default.

        Returns:
            RamanDataset|None: A RamanDataset object containing
                                the data, target, spectra and metadata.
                                If the dataset isn't on the list of a loader
                                or load fails, returns None.
        """
        raise NotImplementedError

