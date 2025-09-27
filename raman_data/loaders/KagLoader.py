from typing import Optional

from kagglehub import load_dataset, dataset_download
from kagglehub import KaggleDatasetAdapter
from numpy import ndarray

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools

class KagLoader(ILoader):
    """
    A static class specified in providing datasets hosted on Kaggle.
    """
    DATASETS = {
        "codina/raman-spectroscopy-of-diabetes": TASK_TYPE.Classification,
        "sergioalejandrod/raman-spectroscopy": TASK_TYPE.Classification,
        "andriitrelin/cells-raman-spectra": TASK_TYPE.Classification
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        file_name: Optional[str] = None,
        cache_path: Optional[str] = None
    ) -> str | None:
        if not LoaderTools.is_dataset_available(dataset_name, KagLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)

        print(f"Downloading Kaggle dataset: {dataset_name}")
        path = dataset_download(
            handle=dataset_name,
            path=file_name
        )
        print(f"Dataset downloaded into {path}")

        return path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        file_name: str,
        cache_path: Optional[str] = None
    ) -> ndarray | None:
        if not LoaderTools.is_dataset_available(dataset_name, KagLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)

        print(f"Loading Kaggle dataset from " \
              f"{cache_path if cache_path else 'default folder (~/.cache/kagglehub)'}")

        df = load_dataset(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=dataset_name,
            path=file_name,
        )
        nd = df.to_numpy()

        return nd


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(KagLoader)

