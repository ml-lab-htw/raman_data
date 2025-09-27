"""
Data structures for the raman_data package.
"""

from dataclasses import dataclass
import numpy as np

from raman_data.loaders.LoaderTools import HASH_TYPE

@dataclass
class RamanDataset:
    """
    A class to represent a Raman spectroscopy dataset.

    Attributes:
        data (np.ndarray): The Raman spectra. Each row is a spectrum, and each column is a Raman shift.
        target (np.ndarray): The target variable(s) for each spectrum. Can be a 1D array for single-target tasks
                         (e.g., class label or concentration) or a 2D array for multi-target tasks.
        metadata (dict): A dictionary containing metadata about the dataset (e.g., source, description).
    """
    data: np.ndarray
    target: np.ndarray
    metadata: dict


@dataclass
class ExternalLink:
    """
    A class to represent a URL of a dataset hosted on a website which
    doesn't provide any API.
    
    Attributes:
        name (str): The name of a dataset to show in the list.
        url (str): The URL of a dataset used to download it.
        checksum (str, optional): The hash value of the downloaded file to ensure
                                  download's security.
        checksum_type (HASH_TYPE, optional): The type of the specified checksum.
    """
    name: str
    url: str
    checksum: str | None = None
    checksum_type: HASH_TYPE | None = None

