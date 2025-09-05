"""
Internal functions for loading and listing datasets.
"""

from typing import List, Optional
import numpy as np
from .types import RamanDataset

from raman_data.loaders.KagLoader import KagLoader
from raman_data.loaders.LoaderTools import TASK_TYPE

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
    # Placeholder for all planned datasets (will be removed later)
    datasets_placeholder = {
        "diabetes_kaggle": "classification",
        "covid19_kaggle": "classification",
        "spectroscopy_kaggle": "classification",
        "cells_raman_spectra_kaggle": "classification",
        "SubstrateMixRaman_hf": "regression",
        "Raman_Spectra_Data_github": "classification",
        "zenodo_10779223": "classification",
        "dtu_contrastive": "classification",
        "mendeley_y4md8znppn": "classification",
        "nature_s41467_019_12898_9": "classification",
    }

    loaders = [KagLoader]
    datasets = {}
    
    for loader in loaders:
        for name, task in loader.DATASETS.items():
            datasets.update({name: task})

    if task_type:
        return [name for name, task in datasets.items() if task == task_type]
    return list(datasets.keys())


def load_dataset(
    name: str,
    cache_dir: Optional[str] = None
) -> RamanDataset:
    """
    Loads a specific Raman spectroscopy dataset.

    When called for the first time, it will download the data from its original source
    and store it in the cache directory. Subsequent calls will load the data from the cache.

    Args:
        name: The name of the dataset to load.
        cache_dir: The directory to use for caching the data. If None, a default
                   directory will be used.

    Returns:
        A RamanDataset object containing the data, target, and metadata.

    Raises:
        ValueError: If the dataset name is not found.
    """
    if name not in list_datasets():
        raise ValueError(f"Dataset '{name}' not found. "
                         f"Available datasets: {list_datasets()}")

    # This is a placeholder for the actual data loading logic.
    # The implementation will involve downloading from different sources (Kaggle, HF, etc.)
    # and processing the data into a unified format.
    print(f"Loading dataset: {name}")
    print(f"Cache directory: {cache_dir or 'default_cache_dir'}")

    # Dummy data for now
    dummy_data = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    dummy_target = np.array([[0, 1], [1, 0], [0, 1]])
    dummy_metadata = {
        "name": name,
        "source": "dummy",
        "description": "This is a dummy dataset for demonstration purposes."
    }

    return RamanDataset(
        data=dummy_data,
        target=dummy_target,
        metadata=dummy_metadata
    )

