"""
Data structures and enums for the raman_data package.
"""

from enum import Enum
import hashlib

from dataclasses import dataclass, field
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
    Zenodo = "ZENODO_CACHE"
    Zip = "ZIP_CACHE"
    Misc = "MISC_CACHE"


class TASK_TYPE(Enum):
    """
    An enum contains possible task types of a
    certain dataset.
    """
    Unknown = 0
    Classification = 1
    Regression = 2
    Denoising = 3
    SuperResolution = 4

    def __str__(self):
        return self.name


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
    metadata: dict[str, str] = field(default_factory=dict)
    target_names: List[str] = field(default_factory=list)
    name: str = ""
    task_type: TASK_TYPE = TASK_TYPE.Unknown

    @property
    def n_spectra(self) -> int:
        """
        Get the number of spectra in the dataset.

        Returns:
            int: The number of individual spectra (rows in the spectra array).
        """
        if isinstance(self.spectra, list):
            return len(self.spectra)
        else:
            return self.spectra.shape[0]

    @property
    def n_frequencies(self) -> int:
        """
        Get the number of frequency points per spectrum.

        Returns:
            int: The number of frequency/wavenumber points (columns in spectra array),
                 or 0 if spectra is 1-dimensional.
        """
        if isinstance(self.spectra, list):
            raise ValueError("spectra with multiple frequencies")
        return self.spectra.shape[1] if len(self.spectra.shape) > 1 else 0

    @property
    def n_raman_shifts(self) -> int:
        """
        Get the number of Raman shift values.

        Returns:
            int: The number of Raman shift/wavenumber values, or 0 if empty.
        """
        if isinstance(self.spectra, list):
            raise ValueError("spectra with multiple raman shifts")
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
        return self.target_names

    @property
    def target_range(self):
        """
        Get the range of target values for regression tasks.

        Returns:
            tuple[float, float] | None: A tuple containing (min, max) target values
                                        if the task type is Regression, None otherwise.
        """
        if self.task_type == TASK_TYPE.Regression:
            return np.min(self.targets), np.max(self.targets)
        return None

    @property
    def min_shift(self):
        """
        Get the minimum Raman shift value.

        Returns:
            float: The minimum wavenumber/Raman shift value in cm⁻¹.
        """
        if isinstance(self.raman_shifts, list):
            return min([rs.min() for rs in self.raman_shifts])
        else:
            return self.raman_shifts.min()

    @property
    def max_shift(self):
        """
        Get the maximum Raman shift value.

        Returns:
            float: The maximum wavenumber/Raman shift value in cm⁻¹.
        """
        if isinstance(self.raman_shifts, list):
            return max([rs.max() for rs in self.raman_shifts])
        else:
            return self.raman_shifts.max()

    def to_dataframe(self, target_idx) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Returns:
            DataFrame with spectral data, wavenumbers as columns, and targets as last column.
        """
        df = pd.DataFrame(self.spectra, columns=self.raman_shifts)
        if self.targets.ndim == 1:
            df["target"] = self.targets
        else:
            df["target"] = self.targets[:, target_idx]
        df.index.name = "spectrum_id"
        return df

    def __len__(self) -> int:
        """
        Return the number of spectra in the dataset.
        Equivalent to n_spectra.
        """
        return self.n_spectra

    def __repr__(self) -> str:
        return (f"<RamanDataset name='{self.name}' n_spectra={self.n_spectra} n_frequencies={self.n_frequencies} "
                f"task_type={self.task_type.name}>")

    def __str__(self) -> str:
        return (f"RamanDataset: {self.name}\n"
                f"  Spectra: {self.n_spectra} x {self.n_frequencies}\n"
                f"  Task type: {self.task_type.name}\n"
                f"  Metadata: {self.metadata}")

    def __getitem__(self, idx):
        """
        Allow indexing and slicing. Returns a new RamanDataset for slices, or a tuple for single index.
        """
        if isinstance(idx, slice):
            return RamanDataset(
                spectra=self.spectra[idx],
                targets=self.targets[idx],
                raman_shifts=self.raman_shifts,
                metadata=self.metadata,
                target_names=self.target_names,
                name=self.name,
                task_type=self.task_type
            )
        else:
            return (self.spectra[idx], self.targets[idx])

    def __iter__(self):
        """
        Iterate over spectra and targets as tuples.
        """
        for i in range(self.n_spectra):
            yield (self.spectra[i], self.targets[i])

    def __contains__(self, item):
        """
        Check if a (spectrum, target) tuple is in the dataset.
        """
        for i in range(self.n_spectra):
            if np.array_equal(self.spectra[i], item[0]) and np.array_equal(self.targets[i], item[1]):
                return True
        return False

    def __eq__(self, other):
        if not isinstance(other, RamanDataset):
            return False
        return (np.array_equal(self.spectra, other.spectra) and
                np.array_equal(self.targets, other.targets) and
                np.array_equal(self.raman_shifts, other.raman_shifts) and
                self.target_names == other.target_names and
                self.name == other.name and
                self.task_type == other.task_type)

    def __bool__(self):
        return self.n_spectra > 0


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

