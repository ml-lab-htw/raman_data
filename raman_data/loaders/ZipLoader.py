from typing import Optional

from numpy import genfromtxt, load, ndarray
from pandas import read_excel

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, HASH_TYPE, LoaderTools

from raman_data.types import ExternalLink

import os.path

class ZipLoader(ILoader):
    """
    A static class specified in providing datasets hosted on websites
    which don't provide any API.
    """
    DATASETS = {
        "mind-lab_covid19": TASK_TYPE.Classification,
        "mind-lab_pd_ad": TASK_TYPE.Classification,
        "csho33_bacteria_id": TASK_TYPE.Classification,
        "mendeley_surface-enhanced-raman": TASK_TYPE.Classification,
        "dtu_raman-spectrum-matching": TASK_TYPE.Classification
    }
    
    __LINKS = [
        ExternalLink(
            name="mind-lab_covid19",
            url="https://github.com/MIND-Lab/Raman-Spectra-Data/archive/refs/heads/main.zip"
        ),
        ExternalLink(
            name="mind-lab_pd_ad",
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
    The `__LINKS` property is meant to store URLs of external sources which don't
    provide any API and therefore any datasets' structures.
    """


    @staticmethod
    def download_dataset(
        dataset_name: str,
        file_name: Optional[str] = None,
        cache_path: Optional[str] = None
    ) -> str | None:
        """
        Downloads certain dataset into a predefined cache folder.

        Args:
            dataset_name (str): The name of a dataset to download.
            file_name (str, optional): The name of a specific dataset's file to download.
                                       If None, downloads whole dataset.
                                       *This loader doesn't support
                                       this feature.*
            cache_path (str, optional): The path to save the dataset to.
                                        If None, uses the lastly saved path.
                                        If "default", sets the default path ('~/.cache').

        Returns:
            str: The path the dataset is downloaded to.
            If the dataset isn't on the list of a loader, returns None.
        """
        if not LoaderTools.is_dataset_available(dataset_name, ZipLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with ZipLoader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zip)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zip)
        cache_path = cache_path if cache_path else os.path.join(os.path.expanduser("~"), ".cache", "ziploader")
        
        print(f"Downloading dataset: {dataset_name}")

        dataset_link = [link for link in ZipLoader.__LINKS if link.name == dataset_name][0]
        download_zip_path = LoaderTools.download(
            url=dataset_link.url,
            out_dir_path=cache_path,
            out_file_name=dataset_name,
            hash_target=dataset_link.checksum,
            hash_type=dataset_link.checksum_type
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
        file_name: str,
        cache_path: Optional[str] = None
    ) -> ndarray | None:
        """
        Loads certain dataset's file from cache folder as a numpy array.
        If requested file isn't in the cache folder, downloads **the whole
        dataset** into that folder.

        Args:
            dataset_name (str): The name of a dataset.
            file_name (str): The name of a specific dataset's file to load.
            cache_path (str, optional): The path to look for the file at.
                                        If None, uses the lastly saved path.
                                        If "default", sets the default path ('~/.cache').

        Returns:
            ndarray: A numpy array representing the loaded file.
            If the dataset isn't on the list of a loader, returns None.
        """
        if not LoaderTools.is_dataset_available(dataset_name, ZipLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with ZipLoader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zip)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zip)
        cache_path = cache_path if cache_path else os.path.join(os.path.expanduser("~"), ".cache", "ziploader")

        if not os.path.exists(os.path.join(cache_path, dataset_name, file_name)):
            print(f"[!] Dataset's file {file_name} not found at: {cache_path}")
            ZipLoader.download_dataset(
                dataset_name=dataset_name,
                cache_path=cache_path
            )

        print(f"Loading dataset from {cache_path}")

        file_path = os.path.join(cache_path, dataset_name, file_name)

        # Converting Excel files with pandas
        if file_name[-4:] in ["xlsx", ".xls"]:
            return read_excel(io=file_path).to_numpy()

        # Converting / reading numpy's native files
        if file_name[-4:] == ".npy":
            return load(file=file_path)

        # Converting CSV files with numpy
        return genfromtxt(fname=file_path, delimiter=",")


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(ZipLoader)

