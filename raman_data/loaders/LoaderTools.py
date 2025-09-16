"""
General functions and enums meant to be used while loading certain dataset.
"""

from enum import Enum

class CACHE_DIR(Enum):
    """
    An enum contains names of environment variables used
    by certain loaders for saving their cache directories.
    """
    Kaggle = "KAGGLEHUB_CACHE"
    HuggingFace = "HF_HOME"


class TASK_TYPE(Enum):
    """
    An enum contains possible task types of a
    certain dataset.
    """
    Classification = 0
    Regression = 1


from raman_data.loaders.ILoader import ILoader
from typing import Optional, List
import os

class LoaderTools:
    """
    A static class contains general methods that
    can be used while loading datasets.
    """
    @staticmethod
    def get_cache_root(
        env_var: CACHE_DIR
    ) -> str | None:
        """
        Retrieves the cache path of a certain loader.

        Args:
            env_var (CACHE_DIR): The name of loader's environment variable.

        Returns:
            str|None: The saved cache path or
                      None, if the path wasn't specified earlier.
        """
        try:
            return os.environ[env_var.value]
        except (KeyError):
            return None

    
    @staticmethod
    def set_cache_root(
        path: str,
        loader_key: Optional[CACHE_DIR] = None
    ) -> None:
        """
        Sets the given path as the cache directory either for a specific
        or for all loaders.

        Args:
            path (str): The path to save datasets to or
                        "default" to reset previously saved path.
            loader_key (CACHE_DIR, optional): The name of loader's
            environment variable that stores the cache path. If None,
            sets the given path for all loaders.
        """
        path = None if path == "default" else path
        
        if not (loader_key is None):
            os.environ[loader_key.value] = path
            print(f"[!] Cache root folder for {loader_key.name}'s loader is set to: {path}")
            
            return
        
        for env_var in CACHE_DIR:
            os.environ[env_var.value] = path
        print(f"[!] Cache root folder is set to: {path}")


    @staticmethod
    def is_dataset_available(
        dataset_name: str,
        datasets: List[str]
    ) -> bool:
        """
        Checks whether given dataset's name is in the given list.

        Args:
            dataset_name (str): The name of a dataset to look for.
            datasets (List[str]): The list of datasets to look among
                                  (typically the list of a loader itself).

        Returns:
            bool: True, if the dataset is on the list. False otherwise.
        """
        check = dataset_name in datasets
        if not check:
            print(f"[!] Dataset {dataset_name} is not on the loader's list.")
        
        return check


    @staticmethod
    def list_datasets(
        loader: ILoader,
    ) -> None:
        """
        Prints a formatted list of datasets of a certain loader.

        Args:
            loader (ILoader): The loader to list datasets of.
        """
        print(f"[*] Datasets available with {loader.__qualname__}:")
        for dataset_name, task_type in loader.DATASETS.items():
            print(f" |-> Name: {dataset_name} | Task type: {task_type.name}")
