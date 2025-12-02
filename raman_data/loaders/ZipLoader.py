from typing import Optional, Tuple

import os.path
from numpy import ndarray

#* These functions could be useful for specific load() functions
# from numpy import genfromtxt, load,
# from pandas import read_excel

from raman_data.types import DatasetInfo, ExternalLink, CACHE_DIR, TASK_TYPE, HASH_TYPE
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import LoaderTools


class ZipLoader(ILoader):
    """
    A static class specified in providing datasets hosted on websites
    which don't provide any API.
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
    The `__LINKS` property is meant to store URLs of external sources which don't
    provide any API and therefore any datasets' structures.
    """


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> str | None:
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
            out_file_name=f"{dataset_name}.zip",
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
