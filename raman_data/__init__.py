"""
A unified API for loading and accessing Raman spectroscopy datasets.
"""

__all__ = [
    "TASK_TYPE",
    "raman_data"
]

from typing import List, Optional, Union

from .types import RamanDataset
from . import datasets
from .types import TASK_TYPE




def raman_data(
    dataset_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    task_type: Optional[TASK_TYPE] = None
) -> Union[RamanDataset, List[str]]:
    """
    Main function to interact with Raman datasets.

    - If 'name' is provided, it loads the specified dataset.
    - If 'name' is None, it lists available datasets, optionally filtered by 'task_type'.

    Args:
        dataset_name: The name of the dataset to load. If None, lists datasets.
        cache_dir: The directory to use for caching the data.
        task_type: Filters the dataset list by task type ('classification' or 'regression').

    Returns:
        - A RamanDataset object if 'name' is specified.
        - A list of dataset names if 'name' is None.
    """
    if dataset_name is None:
        return datasets.list_datasets(task_type=task_type)
    else:
        dataset = datasets.load_dataset(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
        )

        if not dataset is None:
            dataset.task_type = task_type
            dataset.name = dataset_name

        return dataset
