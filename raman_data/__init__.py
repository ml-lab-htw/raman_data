"""
A unified API for loading and accessing Raman spectroscopy datasets.
"""

from typing import List, Optional, Literal, Union

from .types import RamanDataset
from . import datasets

def raman_data(
    name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    task_type: Optional[Literal['classification', 'regression']] = None
) -> Union[RamanDataset, List[str]]:
    """
    Main function to interact with Raman datasets.

    - If 'name' is provided, it loads the specified dataset.
    - If 'name' is None, it lists available datasets, optionally filtered by 'task_type'.

    Args:
        name: The name of the dataset to load. If None, lists datasets.
        cache_dir: The directory to use for caching the data.
        task_type: Filters the dataset list by task type ('classification' or 'regression').

    Returns:
        - A RamanDataset object if 'name' is specified.
        - A list of dataset names if 'name' is None.
    """
    if name is None:
        return datasets.list_datasets(task_type=task_type)
    else:
        return datasets.load_dataset(name=name, cache_dir=cache_dir)
