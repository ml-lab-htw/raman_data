from typing import Optional, Tuple

import datasets
import pandas as pd
import numpy as np

from raman_data.types import DatasetInfo
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools


class HugLoader(ILoader):
    """
    A static class specified in providing datasets hosted on HuggingFace.
    """
    def __load_substarteMix(
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        df = pd.concat(
            [
                pd.DataFrame(data["train"]),
                pd.DataFrame(data["test"]),
                pd.DataFrame(data["validation"]),
            ],
            ignore_index=True,
        )

        end_data_index = len(df.columns.values) - 8

        raman_shifts = df.loc[:, :"3384.7"].to_numpy().T
        spectra = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, "Glucose":].to_numpy()

        return raman_shifts, spectra, concentrations


    def __load_EcoliFermentation(
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        df = pd.concat(
            [
                pd.DataFrame(data["train"]),
                pd.DataFrame(data["test"]),
                pd.DataFrame(data["validation"]),
            ],
            ignore_index=True,
        )

        end_data_index = len(df.columns.values) - 2

        raman_shifts = df.loc[:, :"3384.7"].to_numpy().T
        spectra = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, :"Glucose"].to_numpy()

        return raman_shifts, spectra, concentrations


    DATASETS = {
        "chlange/SubstrateMixRaman": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id=None,
            loader=__load_substarteMix
        ),
        "chlange/RamanSpectraEcoliFermentation": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id=None,
            loader=__load_EcoliFermentation
        )
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
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
            cache_dir=cache_path
        )

        cache_path = cache_path if cache_path else "~/.cache/huggingface"
        print(f"Dataset downloaded into {cache_path}")

        return cache_path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not LoaderTools.is_dataset_available(dataset_name, HugLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        print(
            f"Loading HuggingFace dataset from " \
            f"{cache_path if cache_path else 'default folder (~/.cache/huggingface)'}"
        )

        dataDict = datasets.load_dataset(path=dataset_name, cache_dir=cache_path)
        data = HugLoader.DATASETS[dataset_name].loader(dataDict)

        if data is None:
            return None, None, None

        return data


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(HugLoader)
