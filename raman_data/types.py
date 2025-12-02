"""
Data structures and enums for the raman_data package.
"""

from enum import Enum
import hashlib

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


class CACHE_DIR(Enum):
    """
    An enum contains names of environment variables used
    by certain loaders for saving their cache directories.
    """
    Kaggle = "KAGGLEHUB_CACHE"
    HuggingFace = "HF_HOME"
    Zenodo = "ZEN_CACHE"
    Zip = "ZIP_CACHE"


class TASK_TYPE(Enum):
    """
    An enum contains possible task types of a
    certain dataset.
    """
    Classification = 0
    Regression = 1


class HASH_TYPE(Enum):
    """
    An enum contains possible hash types of a
    certain dataset's checksum. Each enums' value
    is a respective `hashlib`'s generating function
    which outputs the related hash upon execution.
    """
    md5 = hashlib.md5
    sha256 = hashlib.sha256


@dataclass
class RamanDataset:
    """
    A class to represent a Raman spectroscopy dataset.

    Attributes:
        data (np.ndarray): The Raman spectra. Each row is a spectrum, and each column is a Raman shift.
        target (np.ndarray): The target variable(s) for each spectrum. Can be a 1D array for single-target tasks
                         (e.g., class label or concentration) or a 2D array for multi-target tasks.
        metadata (dict[str, str]): A dictionary containing metadata about the dataset (e.g., source, description).
    """
    data: np.ndarray
    target: np.ndarray
    spectra: np.ndarray
    metadata: dict[str, str]


@dataclass(init=False)
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
    
    def __init__(
        self, 
        id: str, 
        key: str, 
        size: int,
        checksum: str, 
        download_link: str, 
        links: dict[str, str]
    ) -> None:
        
        self.id = id
        self.key = key
        self.size = size
        self.checksum = checksum.removeprefix("md5:").strip()
        self.download_link = download_link
        self.links = links
 
 
@dataclass
class DatasetInfo:
    """
    A class to represent dataset's information for its preparation.
    
    Attributes:
        task_type (TASK_TYPE): The task type of the dataset
                               e.g. Classification or Regression.
        id (str): An internal id to distinguish between sub-datasets.
        loader (Callable): The function to format the dataset.
        metadata (dict[str, str]): Some non-functional information about the dataset.
    """
    task_type: TASK_TYPE
    id: str
    loader: Callable[[str], Tuple[np.ndarray, np.ndarray, np.ndarray] | None]
    metadata : dict[str, str]
    
    
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

