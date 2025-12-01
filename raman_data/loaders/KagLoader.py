import os

from typing import Optional

from kagglehub import dataset_load, dataset_download
from kagglehub import KaggleDatasetAdapter
import numpy as np

from raman_data import types
from raman_data.types import DatasetInfo
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools

class KagLoader(ILoader):
    """
    A static class specified in providing datasets hosted on Kaggle.
    """

    @staticmethod
    def load_diabetes(id: str) -> np.ndarray|None:
        
        file_handle = "codina/raman-spectroscopy-of-diabetes"

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path= id + ".csv"
        )

        if id == "AGEs":
            spectra = df.loc[0, "Var802":].to_numpy()
            raman_shifts = df.loc[1:, "Var802":].to_numpy().T
            concentration = df.loc[1:, "AGEsID"].to_numpy()
        else:
            spectra = df.loc[0, "Var2":].to_numpy()
            raman_shifts = df.loc[1:, "Var2":].to_numpy().T
            concentration = df.loc[1:, "has_DM2"].to_numpy()

        return raman_shifts, spectra, concentration

    
    @staticmethod
    def load_sergioalejandrod(id:str) -> np.ndarray|None:
        file_handle = "sergioalejandrod/raman-spectroscopy"
        header = ["Gly, 40 mM", "Leu, 40 mM", "Phe, 40 mM", "Trp, 40 mM"]


        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path="AminoAcids_40mM.xlsx",
            pandas_kwargs={"sheet_name": f"Sheet{id}"}
        )

        spectra = df.loc[:, header[(int(id)-1)]].to_numpy()
        raman_shifts = df.loc[1:, 4.5:].to_numpy().T
        concentration = np.array(df.columns.values[2:], dtype=float)

        return raman_shifts, spectra, concentration
    
    
    @staticmethod
    def load_andriitrelin():
        raise NotImplementedError

    DATASETS = {
        "codina/diabetes/AGEs" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="AGEs",
                                                loader=load_diabetes),
        "codina/diabetes/earLobe" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="earLobe",
                                                loader=load_diabetes),
        "codina/diabetes/innerArm" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="innerArm",
                                                loader=load_diabetes),
        "codina/diabetes/thumbNail" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="thumbNail",
                                                loader=load_diabetes),
        "codina/diabetes/vein" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="vein",
                                                loader=load_diabetes),

        "sergioalejandrod/AminoAcids/glycine" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="1",
                                                loader=load_sergioalejandrod),
        "sergioalejandrod/AminoAcids/leucine" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="2",
                                                loader=load_sergioalejandrod),
        "sergioalejandrod/AminoAcids/phenylalanine" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="3",
                                                loader=load_sergioalejandrod),
        "sergioalejandrod/AminoAcids/tryptophan" : DatasetInfo(
                                                task_type=TASK_TYPE.Classification,
                                                id="4",
                                                loader=load_sergioalejandrod),

        #"andriitrelin/cells-raman-spectra" : types.datasetInfo(
        #                                                    task_type=TASK_TYPE.Classification,
        #                                                    id=None,
        #                                                    loader=load_andriitrelin)
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
    ) -> np.ndarray | None:
        if not LoaderTools.is_dataset_available(dataset_name, KagLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)

        print(f"Loading Kaggle dataset from " \
              f"{cache_path if cache_path else 'default folder (~/.cache/kagglehub)'}")

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

