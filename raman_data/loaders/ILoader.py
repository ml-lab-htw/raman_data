from abc import ABCMeta, abstractmethod
from typing import Optional

from numpy import ndarray

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
        file_name: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> str | None:
        """
        Downloads certain dataset into a predefined cache folder.

        Args:
            dataset_name (str): The name of a dataset to download.
            file_name (str, optional): The name of a specific dataset's file to download.
                                       If None, downloads whole dataset.
            cache_dir (str, optional): The path to save the dataset to.
                                       If None, uses the lastly saved path.

        Raises:
            NotImplementedError: If not implemented raises the error by default.

        Returns:
            str: The path the dataset is downloaded to.
            If the dataset isn't on the list of a loader, returns None.
        """
        raise NotImplementedError


    @abstractmethod
    def load_dataset(
        dataset_name: str,
        file_name: str,
        cache_dir: Optional[str] = None
    ) -> ndarray | None:
        """
        Loads certain dataset's file from cache folder as a numpy array.
        If requested file isn't in the cache folder, downloads it into that folder.

        Args:
            dataset_name (str): The name of a dataset.
            file_name (str): The name of a specific dataset's file to load.
            cache_dir (str, optional): The path to look for the file at.
                                       If None, uses the lastly saved path.

        Raises:
            NotImplementedError: If not implemented raises the error by default.

        Returns:
            ndarray: A numpy array representing the loaded file.
            If the dataset isn't on the list of a loader, returns None.
        """
        raise NotImplementedError

