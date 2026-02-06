"""
A unified API for loading and accessing Raman spectroscopy datasets.
"""

__all__ = [
    "TASK_TYPE",
    "APPLICATION_TYPE",
    "raman_data",
    "RamanDataset",
]

from typing import List, Optional, Union
import logging

from .types import RamanDataset, TASK_TYPE, APPLICATION_TYPE
from . import datasets

logger = logging.getLogger(__name__)


def raman_data(
    dataset_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    task_type: Optional[TASK_TYPE] = None,
    application_type: Optional[APPLICATION_TYPE] = None,
    load_data: bool = True,
) -> Union[RamanDataset, List[str]]:
    """
    Main function to interact with Raman datasets.

    - If 'name' is provided, it loads the specified dataset.
    - If 'name' is None, it lists available datasets, optionally filtered by 'task_type'
      and/or 'application_type'.

    Args:
        dataset_name: The name of the dataset to load. If None, lists datasets.
        cache_dir: The directory to use for caching the data.
        task_type: Filters the dataset list by task type ('classification' or 'regression').
        application_type: Filters the dataset list by application domain.
        load_data: If True, loads the actual spectral data. If False, returns metadata only.

    Returns:
        - A RamanDataset object if 'name' is specified.
        - A list of dataset names if 'name' is None.
    """
    if dataset_name is None:
        logger.debug("Listing available datasets%s", f" filtered by {task_type.name}" if task_type else "")
        return datasets.list_datasets(task_type=task_type, application_type=application_type)
    else:
        logger.info(f"Loading dataset: {dataset_name} (cache_dir={cache_dir})")
        return datasets.load_dataset(dataset_name=dataset_name, cache_dir=cache_dir, load_data=load_data)
