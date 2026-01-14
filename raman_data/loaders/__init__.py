"""
Loaders module for the raman_data package.

This module provides various loader classes for downloading and loading
Raman spectroscopy datasets from different sources:

- KagLoader: Datasets hosted on Kaggle
- HugLoader: Datasets hosted on HuggingFace
- ZenLoader: Datasets hosted on Zenodo
- ZipLoader: Datasets from external URLs (no API)
- ILoader: Abstract base class defining the loader interface
- LoaderTools: Utility functions for all loaders
"""

__all__ = [
    "KagLoader",
    "HugLoader",
    "ZenLoader",
    "ZipLoader",
    "ILoader",
    "LoaderTools",
    "CACHE_DIR",
    "TASK_TYPE"
]