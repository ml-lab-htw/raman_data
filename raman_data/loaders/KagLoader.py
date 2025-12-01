from typing import Optional, Tuple

from kagglehub import dataset_load, dataset_download
from kagglehub import KaggleDatasetAdapter
import numpy as np

from raman_data.types import DatasetInfo
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools


class KagLoader(ILoader):
    """
    A static class specified in providing datasets hosted on Kaggle.
    """
    @staticmethod
    def __load_diabetes(
        id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        file_handle = "codina/raman-spectroscopy-of-diabetes"

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path=f"{id}.csv"
        )

        if id == "AGEs":
            raman_shifts = df.loc[1:, "Var802":].to_numpy().T
            spectra = df.loc[0, "Var802":].to_numpy()
            concentration = df.loc[1:, "AGEsID"].to_numpy()
        else:
            raman_shifts = df.loc[1:, "Var2":].to_numpy().T
            spectra = df.loc[0, "Var2":].to_numpy()
            concentration = df.loc[1:, "has_DM2"].to_numpy()

        return raman_shifts, spectra, concentration


    @staticmethod
    def __load_sergioalejandrod(
        id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        file_handle = "sergioalejandrod/raman-spectroscopy"
        header = ["Gly, 40 mM", "Leu, 40 mM", "Phe, 40 mM", "Trp, 40 mM"]

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path="AminoAcids_40mM.xlsx",
            pandas_kwargs={"sheet_name": f"Sheet{id}"}
        )

        raman_shifts = df.loc[1:, 4.5:].to_numpy().T
        spectra = df.loc[:, header[(int(id) - 1)]].to_numpy()
        concentration = np.array(df.columns.values[2:], dtype=float)

        return raman_shifts, spectra, concentration


    @staticmethod
    def __load_andriitrelin():
        raise NotImplementedError


    DATASETS = {
        "codina/diabetes/AGEs": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="AGEs",
            loader=__load_diabetes
        ),
        "codina/diabetes/earLobe": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="earLobe",
            loader=__load_diabetes
        ),
        "codina/diabetes/innerArm": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="innerArm",
            loader=__load_diabetes
        ),
        "codina/diabetes/thumbNail": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="thumbNail",
            loader=__load_diabetes
        ),
        "codina/diabetes/vein": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="vein",
            loader=__load_diabetes
        ),
        "sergioalejandrod/AminoAcids/glycine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="1",
            loader=__load_sergioalejandrod
        ),
        "sergioalejandrod/AminoAcids/leucine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="2",
            loader=__load_sergioalejandrod
        ),
        "sergioalejandrod/AminoAcids/phenylalanine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="3",
            loader=__load_sergioalejandrod
        ),
        "sergioalejandrod/AminoAcids/tryptophan": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="4",
            loader=__load_sergioalejandrod
        ),
        # "andriitrelin/cells-raman-spectra": DatasetInfo(
        #     task_type=TASK_TYPE.Classification,
        #     id=None,
        #     loader=__load_andriitrelin
        # )
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None,
    ) -> str | None:
        if not LoaderTools.is_dataset_available(dataset_name, KagLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        print(f"Downloading Kaggle dataset: {dataset_name}")
        
        path = dataset_download(handle=dataset_name, path=cache_path)
        print(f"Dataset downloaded into {path}")

        return path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not LoaderTools.is_dataset_available(dataset_name, KagLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        print(
            f"Loading Kaggle dataset from "
            f"{cache_path if cache_path else 'default folder (~/.cache/kagglehub)'}"
        )

        dataset_id = KagLoader.DATASETS[dataset_name].id
        data = KagLoader.DATASETS[dataset_name].loader(dataset_id)

        if data is None:
            return None, None, None

        return data

    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(KagLoader)
