"""
Internal functions for loading and listing datasets.
"""

from typing import List, Optional
from .types import RamanDataset

from raman_data.loaders.KagLoader import KagLoader
from raman_data.loaders.HugLoader import HugLoader
from raman_data.loaders.ZenLoader import ZenLoader
from raman_data.loaders.ZipLoader import ZipLoader
from raman_data.types import TASK_TYPE

__LOADERS = [
    KagLoader,
    HugLoader,
    ZenLoader,
    #ZipLoader
]

def list_datasets(
    task_type: Optional[TASK_TYPE] = None
) -> List[str]:
    """
    Lists the available Raman spectroscopy datasets.

    Args:
        task_type: If specified, filters the datasets by task type.
                   Can be 'TASK_TYPE.Classification' or 'TASK_TYPE.Regression'.

    Returns:
        A list of available dataset names.
    """

    datasets = {}

    for loader in __LOADERS:
        for name, dataset_info in loader.DATASETS.items():
            datasets.update({name: dataset_info})

    if task_type:
        return [name for name, dataset_info in datasets.items() if dataset_info.task_type == task_type]
    
    return list(datasets.keys())


def load_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None
) -> RamanDataset:
    """
    (Down-)Loads a specific Raman spectroscopy dataset.

    When called for the first time, it will download the data from its original source
    and store it in the cache directory. Subsequent calls will load the data from the cache.

    Args:
        dataset_name: The name of the dataset to load.
        cache_dir: The directory to use for caching the data. If None, a default
                   directory will be used.

    Returns:
        A RamanDataset object containing the data, target, and metadata.

    Raises:
        ValueError: If the dataset name is not found.
    """
    if dataset_name not in list_datasets():
        raise ValueError(f"Dataset '{dataset_name}' not found. "
                         f"Available datasets: {list_datasets()}")

    get_dataset = None

    raman_data = None
    raman_target = []
    raman_meta = {
        "name": f"{dataset_name}",
        "source": "dummy",
        "description": "This is a dummy dataset for demonstration purposes."
    }
    
    for loader in __LOADERS:
        if not (dataset_name in loader.DATASETS):
            continue
        
        get_dataset = loader.load_dataset
        break

    raman_data, temp, raman_target = get_dataset(dataset_name, cache_dir)

    return RamanDataset(
        data=raman_data,
        target=raman_target,
        metadata=raman_meta
    )

