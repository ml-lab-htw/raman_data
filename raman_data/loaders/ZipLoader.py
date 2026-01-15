from typing import Optional, Tuple

import os.path
from numpy import ndarray

#* These functions could be useful for specific load() functions
# from numpy import genfromtxt, load,
# from pandas import read_excel

from raman_data.types import DatasetInfo, ExternalLink, CACHE_DIR, TASK_TYPE, HASH_TYPE
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools


class ZipLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets from external URLs.

    This loader handles datasets hosted on websites that don't provide a formal API.
    It downloads files directly via URL, handles ZIP extraction, and supports
    optional checksum verification for data integrity.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import ZipLoader
        >>> dataset = ZipLoader.load_dataset("MIND-Lab_covid+pd_ad_bundle")
        >>> ZipLoader.list_datasets()

    Note:
        This loader is currently disabled in the main datasets module.
    """
    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ziploader")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Zip)

    DATASETS = {
        "MIND-Lab_covid+pd_ad_bundle": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="1",
            loader=...,
            metadata={}
        ),
        "csho33_bacteria_id": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="2",
            loader=...,
            metadata={}
        ),
        "mendeley_surface-enhanced-raman": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="3",
            loader=...,
            metadata={}
        ),
        "dtu_raman-spectrum-matching": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="4",
            loader=...,
            metadata={}
        )
    }

    __LINKS = [
        ExternalLink(
            name="MIND-Lab_covid+pd_ad_bundle",
            url="https://github.com/MIND-Lab/Raman-Spectra-Data/archive/refs/heads/main.zip"
        ),
        ExternalLink(
            name="csho33_bacteria_id",
            url="https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&e=1&st=dmn0jupt&dl=1"
        ),
        ExternalLink(
            name="mendeley_surface-enhanced-raman",
            url="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/y4md8znppn-1.zip",
            checksum="423123bb7df2607825b4fcc7d2178a8b3cfaf8cecfba719f8510d56827658c0d",
            checksum_type=HASH_TYPE.sha256
        ),
        ExternalLink(
            name="dtu_raman-spectrum-matching",
            url="https://data.dtu.dk/ndownloader/files/36144495",
            checksum="f3280bc15f1739baf7d243c4835ab2d4",
            checksum_type=HASH_TYPE.md5
        )
    ]
    """
    List of external dataset links with optional checksum verification.
    
    Each ExternalLink contains the dataset name, download URL, and optionally
    a checksum value and type for integrity verification during download.
    """


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> str | None:
        """
        Download a dataset from an external URL to the local cache.

        Downloads the dataset as a ZIP file, verifies checksum if provided,
        and extracts the contents to the cache directory.

        Args:
            dataset_name: The name of the dataset to download (e.g., "MIND-Lab_covid+pd_ad_bundle").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        cache directory (~/.cache/ziploader).

        Returns:
            str | None: The path where the dataset was extracted, or None if the
                        dataset is not available through this loader.
        """
        if not LoaderTools.is_dataset_available(dataset_name, ZipLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with ZipLoader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zip)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zip)

        print(f"Downloading dataset: {dataset_name}")

        dataset_link = [
            link for link in ZipLoader.__LINKS if link.name == dataset_name
        ][0]
        download_zip_path = LoaderTools.download(
            url=dataset_link.url,
            out_dir_path=cache_path,
            out_file_name=dataset_name,
            hash_target=dataset_link.checksum,
            hash_type=dataset_link.checksum_type,
        )

        print("Unzipping files...")

        download_path = LoaderTools.extract_zip_file_content(
            zip_file_path=download_zip_path,
            unzip_target_subdir=dataset_name
        )

        print(f"Dataset downloaded into {download_path}")

        return download_path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> Tuple[ndarray, ndarray, ndarray] | None:
        """
        Load a dataset from an external source as raw numpy arrays.

        Downloads the dataset if not already cached, then returns the parsed data.

        Args:
            dataset_name: The name of the dataset to load (e.g., "MIND-Lab_covid+pd_ad_bundle").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        cache directory (~/.cache/ziploader).

        Returns:
            tuple | None: A tuple of (raman_shifts, spectra, concentrations) numpy arrays,
                          or (None, None, None) if loading fails.
        """
        if not LoaderTools.is_dataset_available(dataset_name, ZipLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with ZipLoader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zip)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zip)

        if not os.path.exists(os.path.join(cache_path, dataset_name)):
            print(f"[!] Dataset isn't found at: {cache_path}")
            ZipLoader.download_dataset(
                dataset_name=dataset_name,
                cache_path=cache_path
            )

        print(f"Loading dataset from {cache_path}")

        #* These methods could be useful for specific load() functions
        # Converting Excel files with pandas
        # if file_name[-4:] in ["xlsx", ".xls"]:
        #     return read_excel(io=file_path).to_numpy()

        # Converting / reading numpy's native files
        # if file_name[-4:] == ".npy":
        #     return load(file=file_path)

        # Converting CSV files with numpy
        # return genfromtxt(fname=file_path, delimiter=",")
        
        data = ZipLoader.DATASETS[dataset_name].loader(cache_path)
        if data is None:
            return None, None, None

        return data


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(ZipLoader)
