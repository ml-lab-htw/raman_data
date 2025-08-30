from typing import Optional

from kagglehub import load_dataset, dataset_download
from kagglehub import KaggleDatasetAdapter
from numpy import ndarray

from ILoader import ILoader
from LoaderTools import CACHE_DIRS, LoaderTools

class KagLoader(ILoader):
    DATASETS = [
        "codina/raman-spectroscopy-of-diabetes",
        "sergioalejandrod/raman-spectroscopy",
        "andriitrelin/cells-raman-spectra"
    ]


    def download_dataset(
        dataset_name: str,
        file_name: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> str:
        if (not LoaderTools.is_dataset_available(dataset_name, KagLoader.DATASETS)):
            print(f"[!] Cannot download {dataset_name} dataset with Kaggle loader")
            return 

        if cache_dir != None:
            LoaderTools.set_cache_root(cache_dir, CACHE_DIRS.Kaggle)

        print("Loading Kaggle dataset...")
        path = dataset_download(
            handle="codina/raman-spectroscopy-of-diabetes",
            path=file_name
        )
        print(f"Dataset downloaded into {path}")

        return path


    def load_dataset(
        dataset_name: str,
        file_name: str,
        cache_dir: Optional[str] = None
    ) -> ndarray:
        if cache_dir != None:
            LoaderTools.set_cache_root(cache_dir, CACHE_DIRS.Kaggle)

        print(f"Loading Kaggle dataset into {LoaderTools.get_cache_root(CACHE_DIRS.Kaggle)}")

        df = load_dataset(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=dataset_name,
            path=file_name,
        )
        nd = df.to_numpy()

        return nd


    def list_datasets() -> None:
        LoaderTools.list_datasets(CACHE_DIRS.Kaggle, KagLoader.DATASETS)

