"""
Data structures for the raman_data package.
"""

from dataclasses import dataclass
from enum import Enum
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
class ExternalLink:
    name: str
    url: str
    checksum: str | None = None
    checksum_type: Enum | None = None

