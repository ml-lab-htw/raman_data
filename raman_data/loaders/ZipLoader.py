from typing import Optional

from numpy import ndarray

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
            url="https://uc62fbf42928de66f8f4eefa3695.dl.dropboxusercontent.com/zip_download_get/CVVyB-vq9ZONS4160D_U7daE999pFmJvXvKkXrDZbv540AGQFT_KjRfiUPWCNuIdLZml5W6vheLM_trgEgCkvkZLq7ooFLEZdlVM3CwTVOfjCw?_download_id=9898965341343273779645420073714159256370985042715217280190759748&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1"
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

        Raises:
            NotImplementedError: If not implemented raises the error by default.

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
        if not LoaderTools.is_dataset_available(dataset_name, ZipLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with ZipLoader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zip)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zip)
        
        print(f"Loading dataset from " \
              f"{cache_path if cache_path else 'default folder (~/.cache)'}")

        # * Downloading magic
        # * Converting magic

        return None


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(ZipLoader)

