from typing import Optional, Tuple

from kagglehub import dataset_load, dataset_download
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import LoaderTools


class KagLoader(ILoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on Kaggle.

    This loader provides access to datasets stored on Kaggle, handling
    download, caching, and formatting of the data into RamanDataset objects.
    Requires Kaggle API credentials to be configured.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import KagLoader
        >>> dataset = KagLoader.load_dataset("codina/diabetes/AGEs")
        >>> KagLoader.list_datasets()

    Note:
        Kaggle API credentials must be set up before using this loader.
        See: https://www.kaggle.com/docs/api
    """

    @staticmethod
    def __load_diabetes(
        id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the diabetes Raman spectroscopy dataset.

        Args:
            id: The specific sub-dataset identifier (e.g., "AGEs", "earLobe", "innerArm").

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        file_handle = "codina/raman-spectroscopy-of-diabetes"

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path=f"{id}.csv"
        )

        if id == "AGEs":
            spectra = df.loc[1:, "Var802":].to_numpy().T
            raman_shifts = df.loc[0, "Var802":].to_numpy()
            concentration = df.loc[1:, "AGEsID"].to_numpy()
        else:
            spectra = df.loc[1:, "Var2":].to_numpy().T
            raman_shifts = df.loc[0, "Var2":].to_numpy()
            concentration = df.loc[1:, "has_DM2"].to_numpy()

        return spectra, raman_shifts, concentration


    @staticmethod
    def __load_sergioalejandrod(
        id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the amino acids Raman spectroscopy dataset.

        Args:
            id: The sheet number identifier (1-4) corresponding to different amino acids.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        file_handle = "sergioalejandrod/raman-spectroscopy"
        header = ["Gly, 40 mM", "Leu, 40 mM", "Phe, 40 mM", "Trp, 40 mM"]

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path="AminoAcids_40mM.xlsx",
            pandas_kwargs={"sheet_name": f"Sheet{id}"}
        )

        spectra = df.loc[1:, 4.5:].to_numpy()
        raman_shifts = df.loc[1:, header[(int(id) - 1)]].to_numpy()
        concentration = np.array(df.columns.values[2:], dtype=float)

        return spectra, raman_shifts, concentration


    @staticmethod
    def __load_andriitrelin():
        """Load the andriitrelin cells Raman spectra dataset (not implemented)."""
        raise NotImplementedError


    @staticmethod
    def __load_cancer_cells(
        id: str
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the cancer cells SERS spectra dataset.

        Args:
            id: The dataset identifier.

        Returns:
            A tuple of (raman_shifts, spectra, concentrations) arrays,
            or None if parsing fails.

        Note:
            This method is not yet implemented.
        """
        raise NotImplementedError

        file_handle = "mathiascharconnet/cancer-cells-sers-spectra"
        lable_list = ["A", "A-S", "G", "G-S", "HPM", "HPM-S", "HF", "HF-S", "ZAM", "ZAM-S", "DMEM", "DMEM-S"]
        file_list = {"(COOH)2.csv":None, 
                     "COOH.csv":None, 
                     "NH2.csv":None}

        for lable in lable_list:
            for file in file_list.keys():

                df = dataset_load(
                    adapter=KaggleDatasetAdapter.PANDAS,
                    handle=file_handle,
                    path=f"{lable}/{file}"
                )

                if file_list[file] is None:
                    file_list[file] = df
                else:
                    file_list[file] = pd.concat([file_list[file], df])
 
        spectra = np.linspace(100, 4278, 2090)


    DATASETS = {
        "codina/diabetes/AGEs": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="AGEs",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina/raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina/diabetes/earLobe": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="earLobe",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina/raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina/diabetes/innerArm": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="innerArm",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina/raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina/diabetes/thumbNail": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="thumbNail",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina/raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina/diabetes/vein": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="vein",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina/raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "sergioalejandrod/AminoAcids/glycine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="1",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod/raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "sergioalejandrod/AminoAcids/leucine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="2",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod/raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "sergioalejandrod/AminoAcids/phenylalanine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="3",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod/raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "sergioalejandrod/AminoAcids/tryptophan": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="4",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod/raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        # "andriitrelin/cells-raman-spectra": DatasetInfo(
        #     task_type=TASK_TYPE.Classification,
        #     id=None,
        #     loader=__load_andriitrelin
        # ),
        #"mathiascharconnet/cancer-cells-sers-spectra" : DatasetInfo(
        #    task_type=TASK_TYPE.Classification,
        #    id=None,
        #    loader=__load_cancer_cells,
        #    metadata={
        #        "full_name" : "mathiascharconnet/cancer-cells-sers-spectra",
        #        "source" : "https://www.kaggle.com/code/mathiascharconnet/cancer-cells-sers-spectra/input",
        #        "paper" : "https://doi.org/10.1016/j.snb.2020.127660",
        #        "description" : "This dataset was collected in the University of Chemistry and Technology, Prague during work on cancer detection. It contains Raman spectra of the culture medium, corresponding to several kinds of cancer and normal cells. Dataset consists of 12 folders with 3 CSV files in each. Folders are named after specific samples (tabulated below). Each CSV in folder contains spectra of medium, collected on the gold nanourchins functionalized with corresponding moiety. Please refer to the original publication for details."
        #    }
        #)
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None,
    ) -> str | None:
        """
        Download a Kaggle dataset to the local cache.

        Args:
            dataset_name: The name of the dataset to download (e.g., "codina/diabetes/AGEs").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        Kaggle cache directory (~/.cache/kagglehub).

        Returns:
            str | None: The path where the dataset was downloaded, or None if the
                        dataset is not available through this loader.
        """
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
    ) -> RamanDataset | None:
        """
        Load a Kaggle dataset as a RamanDataset object.

        Downloads the dataset if not already cached, then parses it into
        a standardized RamanDataset format.

        Args:
            dataset_name: The name of the dataset to load (e.g., "codina/diabetes/AGEs").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        Kaggle cache directory (~/.cache/kagglehub).

        Returns:
            RamanDataset | None: A RamanDataset object containing the spectral data,
                                 target values, and metadata, or None if loading fails.
        """
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

        if data is not None:
            spectra, raman_shifts, concentrations = data
            return RamanDataset(
                spectra=spectra,
                target=concentrations,
                raman_shifts=raman_shifts,
                metadata=KagLoader.DATASETS[dataset_name].metadata
            )
        
        return data

    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(KagLoader)
