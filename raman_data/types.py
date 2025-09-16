"""
Data structures for the raman_data package.
"""

from dataclasses import dataclass
import numpy as np

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
class ZenodoFileInfo:
    """
    A class holding information about a downloadeble file from Zenodo.
    
    Attributes:
        id (str): A 39 alphanumerical unique identifier.
        key (str): The name of the file. 
        size (int): The size of the file in bytes.
        checksum (str): The md5 hexadecimal hash. 
        download_link (str): The link for downloading this file.
        links (dict[str, str]): A dictonary of all associated links.
    """
    id: str
    key: str
    size: int
    checksum: str
    download_link: str
    links: dict[str, str]