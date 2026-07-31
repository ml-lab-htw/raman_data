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
    is_grouped: Optional[bool] = None,
    has_missing_labels: Optional[bool] = None,
    load_data: bool = True,
) -> list[str] | RamanDataset | None:
    """
    Main function to interact with Raman datasets.

    - If 'name' is provided, it loads the specified dataset.
    - If 'name' is None, it lists available datasets, optionally filtered by 'task_type'
      and/or 'application_type' and/or 'is_grouped' and/or 'has_missing_labels'.

    Args:
        dataset_name: The name of the dataset to load. If None, lists datasets.
        cache_dir: The directory to use for caching the data.
        task_type: Filters the dataset list by task type ('classification' or 'regression').
        application_type: Filters the dataset list by application domain.
        is_grouped: Filters the dataset list by known physical-replicate structure
            (see `DatasetInfo.is_grouped`). `True` for only datasets with confirmed
            replicate structure, `False` for only datasets confirmed to have none.
            Datasets where this hasn't been checked yet are excluded by either
            `True` or `False` -- leave as `None` (the default) to include them.
        has_missing_labels: Filters the dataset list by whether at least one target
            column has missing (NaN) values (see `DatasetInfo.has_missing_labels`).
            `True` for only datasets with confirmed missing labels (candidates for
            semi-supervised benchmarking), `False` for only datasets confirmed
            fully labeled. Datasets where this hasn't been checked yet are
            excluded by either `True` or `False` -- leave as `None` (the default)
            to include them.
        load_data: If True, loads the actual spectral data. If False, returns metadata only.

    Returns:
        - A RamanDataset object if 'name' is specified.
        - A list of dataset names if 'name' is None.
    """
    if dataset_name is None:
        logger.info("Listing available datasets%s", f" filtered by {task_type.name}" if task_type else "")
        return datasets.list_datasets(
            task_type=task_type, application_type=application_type,
            is_grouped=is_grouped, has_missing_labels=has_missing_labels,
        )
    else:
        logger.info("Loading dataset: %s (cache_dir=%s)", dataset_name, cache_dir)
        return datasets.load_dataset(dataset_name=dataset_name, cache_dir=cache_dir, load_data=load_data)
