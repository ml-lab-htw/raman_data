"""
Loaders module for the raman_data package.

This module provides various loader classes for downloading and loading
Raman spectroscopy datasets from different sources:

- KaggleLoader: Datasets hosted on Kaggle
- HuggingFaceLoader: Datasets hosted on HuggingFace
- ZenodoLoader: Datasets hosted on Zenodo
- ZipLoader: Datasets from external URLs (no API)
- BaseLoader: Abstract base class defining the loader interface
- LoaderTools: Utility functions for all loaders
"""

__all__ = [
    "KaggleLoader",
    "HuggingFaceLoader",
    "ZenodoLoader",
    "ZipLoader",
    "BaseLoader",
    "LoaderTools",
    "CACHE_DIR",
    "TASK_TYPE"
]