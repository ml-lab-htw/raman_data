from typing import Optional

from datasets import load_dataset
from numpy import ndarray

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools

class HugLoader(ILoader):
    """
    A static class specified in providing datasets hosted on HuggingFace.
    """
    DATASETS = {
        "chlange/SubstrateMixRaman": TASK_TYPE.Regression
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
        load_dataset(
            path=dataset_name,
            data_files=file_name,
            cache_dir=cache_path
        )
        print(f"Dataset downloaded into " \
              f"{cache_path if cache_path else 'default folder (~/.cache)'}")

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
              f"{cache_path if cache_path else 'default folder (~/.cache)'}")

        df = load_dataset(
            path=dataset_name,
            data_files=file_name,
            cache_dir=cache_path
        )
        nd = df.with_format("numpy")

        return nd


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(HugLoader)

