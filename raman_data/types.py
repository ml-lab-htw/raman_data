"""
Data structures and enums for the raman_data package.
"""

from enum import Enum
import hashlib

from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
import numpy as np
import pandas as pd


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
    Unknown = 0
    Classification = 1
    Regression = 2


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
        name (str): The name of the dataset.
        task_type (TASK_TYPE): The task type of the dataset
                               e.g. Classification or Regression.
        spectra (np.ndarray): The Raman spectra intensity data. Each row is a spectrum,
                              and each column corresponds to a Raman shift value.
        targets (np.ndarray): The target variable(s) for each spectrum. Can be a 1D array for single-target tasks
                         (e.g., class label or concentration) or a 2D array for multi-target tasks.
        raman_shifts (np.ndarray): The wavenumber/Raman shift values (x-axis) in cm⁻¹.
        metadata (dict[str, str]): A dictionary containing metadata about the dataset (e.g., source, description).
    """
    spectra: np.ndarray
    targets: np.ndarray
    raman_shifts: np.ndarray
    metadata: dict[str, str]
    name: str = ""
    task_type: TASK_TYPE = TASK_TYPE.Unknown

    @property
    def n_spectra(self) -> int:
        """
        Get the number of spectra in the dataset.

        Returns:
            int: The number of individual spectra (rows in the spectra array).
        """
        return self.spectra.shape[0]

    @property
    def n_frequencies(self) -> int:
        """
        Get the number of frequency points per spectrum.

        Returns:
            int: The number of frequency/wavenumber points (columns in spectra array),
                 or 0 if spectra is 1-dimensional.
        """
        return self.spectra.shape[1] if len(self.spectra.shape) > 1 else 0

    @property
    def n_raman_shifts(self) -> int:
        """
        Get the number of Raman shift values.

        Returns:
            int: The number of Raman shift/wavenumber values, or 0 if empty.
        """
        return self.raman_shifts.shape[0] if len(self.raman_shifts.shape) > 0 else 0

    @property
    def n_classes(self) -> Optional[int]:
        """
        Get the number of unique classes for classification tasks.

        Returns:
            int | None: The number of unique class labels if the task type
                        is Classification, None otherwise.
        """
        if self.task_type == TASK_TYPE.Classification:
            return len(np.unique(self.targets))
        return None

    @property
    def class_names(self) -> Optional[List[str]]:
        """
        Get the unique class names for classification tasks.

        Returns:
            list[str] | None: A list of unique class names if the task type
                              is Classification, None otherwise.
        """
        if self.task_type == TASK_TYPE.Classification:
            return list(np.unique(self.targets))
        return None

    @property
    def target_range(self):
        """
        Get the range of target values for regression tasks.

        Returns:
            tuple[float, float] | None: A tuple containing (min, max) target values
                                        if the task type is Regression, None otherwise.
        """
        if self.task_type == TASK_TYPE.Regression:
            return (np.min(self.targets), np.max(self.targets))
        return None

    @property
    def min_shift(self):
        """
        Get the minimum Raman shift value.

        Returns:
            float: The minimum wavenumber/Raman shift value in cm⁻¹.
        """
        return self.raman_shifts.min()

    @property
    def max_shift(self):
        """
        Get the maximum Raman shift value.

        Returns:
            float: The maximum wavenumber/Raman shift value in cm⁻¹.
        """
        return self.raman_shifts.max()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Returns:
            DataFrame with spectral data, wavenumbers as columns, and targets as last column.
        """
        df = pd.DataFrame(self.spectra.T, columns=self.raman_shifts)
        df["targets"] = self.targets
        return df


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
        """
        Initialize a ZenodoFileInfo instance.

        Args:
            id: A 39-character alphanumerical unique identifier.
            key: The name of the file.
            size: The size of the file in bytes.
            checksum: The md5 hexadecimal hash (with or without 'md5:' prefix).
            download_link: The direct URL for downloading this file.
            links: A dictionary of all associated links.
        """
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
    loader: Callable[[str], Tuple[np.ndarray, np.ndarray, np.ndarray] | pd.DataFrame | None]
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

