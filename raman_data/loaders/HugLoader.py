from typing import Optional

import datasets
#from datasets import load_dataset
from numpy import ndarray
import pandas as pd
import numpy as np

from raman_data import types
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools

class HugLoader(ILoader):
    """
    A static class specified in providing datasets hosted on HuggingFace.
    """
    DATASETS = {
        "chlange/SubstrateMixRaman": TASK_TYPE.Regression
    }


    def load_substarteMix(data: pd.DataFrame) -> np.ndarray|None:

        df = pd.concat([pd.DataFrame(data["train"]),pd.DataFrame(data["test"]) ,pd.DataFrame(data["validation"])], ignore_index=True)
            
        end_data_index = len(df.columns.values)-8

        concentrations = df.loc[:, "Glucose":].to_numpy()
        spectra = np.array(df.columns.values[:end_data_index])
        raman_shifts = df.loc[:, :"3384.7"].to_numpy().T

        return raman_shifts, spectra, concentrations


    DATASETS_INFO = {
        "chlange/SubstrateMixRaman" : types.datasetInfo(
                                                    task_type=TASK_TYPE.Regression,
                                                    id=None,
                                                    loader=load_substarteMix)
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        file_name: Optional[str] = None,
        cache_path: Optional[str] = None
    ) -> str | None:
        if not LoaderTools.is_dataset_available(dataset_name, HugLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        print(f"Downloading HuggingFace dataset: {dataset_name}")
        datasets.load_dataset(
            path=dataset_name,
            data_files=file_name,
            cache_dir=cache_path
        )

        cache_path = cache_path if cache_path else "~/.cache/huggingface"
        print(f"Dataset downloaded into {cache_path}")

        return cache_path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        file_name: str,
        cache_path: Optional[str] = None
    ) -> ndarray | None:
        if not LoaderTools.is_dataset_available(dataset_name, HugLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        print(f"Loading HuggingFace dataset from " \
              f"{cache_path if cache_path else 'default folder (~/.cache/huggingface)'}")

        dataDict = datasets.load_dataset(
            path=dataset_name,
            cache_dir=cache_path
        )

        return HugLoader.DATASETS_INFO[dataset_name].loader(dataDict)


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(HugLoader)

