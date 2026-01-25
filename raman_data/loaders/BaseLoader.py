from abc import ABCMeta, abstractmethod
from typing import Optional, Dict
import logging

from datasets import DatasetInfo

from raman_data.types import RamanDataset

class BaseLoader(metaclass=ABCMeta):
    """
    The general interface of all loaders.
    """

    DATASETS: Dict[str, DatasetInfo] = {}
    logger = logging.getLogger(__name__)

    @staticmethod
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


    @staticmethod
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
                                the data, targets, spectra and metadata.
                                If the dataset isn't on the list of a loader
                                or load fails, returns None.
        """
        raise NotImplementedError(f"load_dataset not implemented in {__class__.__name__} for {dataset_name}")

    @classmethod
    def is_dataset_available(cls, dataset_name: str) -> bool:
        """
        Check if a dataset is available through this loader.

        Args:
            dataset_name: The name of the dataset to check.

        Returns:
            bool: True if the dataset is available, False otherwise.
        """
        # Default implementation subclasses can use
        return dataset_name in cls.DATASETS

    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Optional[DatasetInfo]:
        """
        Get metadata information for a specific dataset.

        Args:
            dataset_name: The name of the dataset.

        Returns:
            DatasetInfo object or None if dataset not found.
        """
        return cls.DATASETS.get(dataset_name)

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        """
        Get list of all available dataset names.

        Returns:
            List of dataset name strings.
        """
        return list(cls.DATASETS.keys())

    @classmethod
    def get_loader_name(cls) -> str:
        """
        Get the name of this loader.
        Returns the class name by default. Override for custom names.
        """
        return cls.__name__

    @classmethod
    def list_datasets(cls) -> None:
        """
        Print formatted list of available datasets.
        Subclasses can override this for custom formatting.
        """
        loader_name = cls.get_loader_name()
        cls.logger.info(f"\n{loader_name} Datasets:")
        cls.logger.info("=" * 50)

        if not cls.DATASETS:
            cls.logger.info("No datasets available.")
            return

        for dataset_name, info in cls.DATASETS.items():
            if hasattr(info, 'task_type'):
                cls.logger.info(f"\n• {dataset_name}")
                cls.logger.info(f"  Task Type: {info.task_type}")
            else:
                cls.logger.info(f"\n• {dataset_name}")

        cls.logger.info("\n" + "=" * 50)
