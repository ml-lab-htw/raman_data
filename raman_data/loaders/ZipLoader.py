from typing import Optional, Tuple

import os
import numpy as np
from numpy import genfromtxt, load
#* These functions could be useful for specific load() functions
# from pandas import read_excel

from raman_data.types import DatasetInfo, ExternalLink, RamanDataset, CACHE_DIR, TASK_TYPE, HASH_TYPE
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import LoaderTools


class ZipLoader(ILoader):
    """
    A static class specified in providing datasets hosted on websites
    which don't provide any API.
    """
    @staticmethod
    def __load_mind_lab_bundle(
        id: str,
        dataset_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        dataset_root = os.path.join(dataset_path, "Raman-Spectra-Data-main")

        if id == "COV":
            dataset_root = os.path.join(dataset_root, "covid_dataset")
        else:
            dataset_root = os.path.join(dataset_root, "pd_ad_dataset")
        data_dirs = os.listdir(dataset_root)
        
        raman_shifts = []
        spectra = []
        target = []
        
        for data_dir in data_dirs:
            # Skipping an extra user_information.csv
            if data_dir == "user_information.csv":
                continue
            
            raman_shifts_path = os.path.join(dataset_root, data_dir, "raman_shift.csv")
            raman_shifts_data = genfromtxt(raman_shifts_path)
            raman_shifts.append(raman_shifts_data)
            
            spectra_path = os.path.join(dataset_root, data_dir, "spectra.csv")
            spectra_data = genfromtxt(spectra_path, delimiter=',')
            spectra.append(spectra_data)
            
            target_path = os.path.join(dataset_root, data_dir, "user_information.csv")
            target_data = genfromtxt(target_path,
                                     dtype=int,
                                     delimiter=',',
                                     skip_header=1)[-1]
            target.append(target_data)

        return raman_shifts, spectra, target
    
    
    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ziploader")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Zip)

    DATASETS = {
        "MIND-Lab_Raman-Spectra-Data_covid-dataset": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="COV",
            loader=__load_mind_lab_bundle,
            metadata={
                "full_name" : "MIND-Lab_covid_dataset",
                "source": "https://github.com/MIND-Lab/Raman-Spectra-Data",
                "paper": "https://pubmed.ncbi.nlm.nih.gov/38335817/",
                "description": "Datasets used for the experimental computations of paper \"An Integrated Computational Pipeline for Machine Learning-Driven Diagnosis based on Raman Spectra of saliva samples\"."
            },
            base_name="MIND-Lab_Raman-Spectra-Data"
        ),
        "MIND-Lab_Raman-Spectra-Data_pd-ad-dataset": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="PD",
            loader=__load_mind_lab_bundle,
            metadata={
                "full_name" : "MIND-Lab_pd_ad_dataset",
                "source": "https://github.com/MIND-Lab/Raman-Spectra-Data",
                "paper": "https://pubmed.ncbi.nlm.nih.gov/38335817/",
                "description": "Datasets used for the experimental computations of paper \"An Integrated Computational Pipeline for Machine Learning-Driven Diagnosis based on Raman Spectra of saliva samples\"."
            },
            base_name="MIND-Lab_Raman-Spectra-Data"
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
            name="MIND-Lab_Raman-Spectra-Data",
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

        dataset_base_name = [
            info.base_name for name, info in ZipLoader.DATASETS.items() if name == dataset_name
        ][0]
        dataset_link = [
            link for link in ZipLoader.__LINKS if link.name == dataset_base_name
        ][0]
        download_zip_path = LoaderTools.download(
            url=dataset_link.url,
            out_dir_path=cache_path,
            out_file_name=f"{dataset_base_name}.zip",
            hash_target=dataset_link.checksum,
            hash_type=dataset_link.checksum_type
        )

        print("Unzipping files...")

        download_path = LoaderTools.extract_zip_file_content(
            zip_file_path=download_zip_path,
            unzip_target_subdir=dataset_base_name
        )

        print(f"Dataset downloaded into {download_path}")

        return download_path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> RamanDataset | None:
        if not LoaderTools.is_dataset_available(dataset_name, ZipLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with ZipLoader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zip)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zip)

        dataset_base_name = [
            info.base_name for name, info in ZipLoader.DATASETS.items() if name == dataset_name
        ][0]

        if not os.path.exists(os.path.join(cache_path, dataset_base_name)):
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

        dataset_id = ZipLoader.DATASETS[dataset_name].id
        data = ZipLoader.DATASETS[dataset_name].loader(
            id=dataset_id,
            dataset_path=os.path.join(cache_path, dataset_base_name))
        if data is None:
            return None, None, None

        raman_shifts, spectra, concentrations = data
        return RamanDataset(
            data=raman_shifts,
            target=concentrations,
            spectra=spectra,
            metadata=ZipLoader.DATASETS[dataset_name].metadata
        )


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(ZipLoader)
