"""
Internal functions for loading and listing datasets.
"""

from typing import List, Optional

from cachetools import LRUCache, cached

from .loaders.MiscLoader import MiscLoader
from .types import RamanDataset

from raman_data.loaders.KaggleLoader import KaggleLoader
from raman_data.loaders.HuggingFaceLoader import HuggingFaceLoader
from raman_data.loaders.ZenodoLoader import ZenodoLoader
from raman_data.types import TASK_TYPE, APPLICATION_TYPE, DatasetInfo

__LOADERS = [
    KaggleLoader,
    HuggingFaceLoader,
    ZenodoLoader,
    MiscLoader,
]

# Create a global LRU cache instance with a capacity of 1
lru_cache = LRUCache(maxsize=1)

def list_datasets(
        task_type: Optional[TASK_TYPE] = None,
        application_type: Optional[APPLICATION_TYPE] = None,
) -> List[str]:
    """
    Lists the available Raman spectroscopy datasets.

    Args:
        task_type: If specified, filters the datasets by task type.
                   Can be 'TASK_TYPE.Classification' or 'TASK_TYPE.Regression'.
        application_type: If specified, filters the datasets by application domain.

    Returns:
        A list of available dataset names.
    """

    datasets = {}

    for loader in __LOADERS:
        for name, dataset_info in loader.DATASETS.items():
            datasets.update({name: dataset_info})

    result = datasets.items()

    if task_type:
        result = [(name, info) for name, info in result if info.task_type == task_type]

    if application_type:
        result = [(name, info) for name, info in result if info.application_type == application_type]

    return [name for name, info in result]


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """
    Returns the DatasetInfo object for a given dataset name.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        DatasetInfo: The dataset's metadata and configuration.

    Raises:
        ValueError: If the dataset name is not found.
    """
    for loader in __LOADERS:
        if dataset_name in loader.DATASETS:
            return loader.DATASETS[dataset_name]

    raise ValueError(f"Dataset '{dataset_name}' not found. "
                     f"Available datasets: {list_datasets()}")


@cached(cache=lru_cache)
def load_dataset(
        dataset_name: str,
        cache_dir: Optional[str] = None,
        load_data: bool = True
) -> RamanDataset | None:
    """
    (Down-)Loads a specific Raman spectroscopy dataset.

    When called for the first time, it will download the data from its original source
    and store it in the cache directory. Subsequent calls will load the data from the cache.

    Args:
        dataset_name: The name of the dataset to load.
        cache_dir: The directory to use for caching the data. If None, a default
                   directory will be used.

    Returns:
        RamanDataset|None: A RamanDataset object containing
                           the data, targets, spectra and metadata or
                           None if load process fails.

    Raises:
        ValueError: If the dataset name is not found.
    """

    if dataset_name not in list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' not found. "
                         f"Available datasets: {list_datasets()}")

    get_dataset = None

    for loader in __LOADERS:
        if not (dataset_name in loader.DATASETS):
            continue

        get_dataset = loader.load_dataset
        break

    return get_dataset(dataset_name, cache_dir, load_data)


def pretty_name(dataset_name: str) -> str:

    if dataset_name not in list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' not found. "
                         f"Available datasets: {list_datasets()}")

    name = ""
    for loader in __LOADERS:
        if not (dataset_name in loader.DATASETS):
            continue

        name = loader.DATASETS[dataset_name].name
        break

    return name