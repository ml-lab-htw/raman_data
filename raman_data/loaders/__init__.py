"""
Loaders module for the raman_data package.

This module provides various loader classes for downloading and loading
Raman spectroscopy datasets from different sources:

- KaggleLoader: Datasets hosted on Kaggle
- HuggingFaceLoader: Datasets hosted on HuggingFace
- ZenodoLoader: Datasets hosted on Zenodo
- ZipLoader: Datasets from external URLs (no API)
- BaseLoader: Abstract base class defining the loader interface
- MiscLoader: Loader for miscellaneous datasets from various other sources
- LoaderTools: Utility functions for all loaders
- CACHE_DIR: Enumeration of cache directory types
- TASK_TYPE: Enumeration of different task types for datasets
"""

__all__ = [
    "KaggleLoader",
    "HuggingFaceLoader",
    "ZenodoLoader",
    "MiscLoader",
    "LoaderTools",
    "CACHE_DIR",
    "TASK_TYPE",
]

from raman_data.types import CACHE_DIR, TASK_TYPE
