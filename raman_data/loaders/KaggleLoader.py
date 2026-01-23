import os
from typing import Optional, Tuple
import logging

from kagglehub import dataset_load, dataset_download
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on Kaggle.

    This loader provides access to datasets stored on Kaggle, handling
    download, caching, and formatting of the data into RamanDataset objects.
    Requires Kaggle API credentials to be configured.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import KaggleLoader
        >>> dataset = KaggleLoader.load_dataset("codina_diabetes_AGEs")
        >>> KaggleLoader.list_datasets()

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
            spectra = df.loc[1:, "Var802":].to_numpy()
            raman_shifts = df.loc[0, "Var802":].to_numpy()
            concentration = df.loc[1:, "AGEsID"].to_numpy()
        else:
            spectra = df.loc[1:, "Var2":].to_numpy()
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
        file_handle = "sergioalejandrod_raman-spectroscopy"
        header = ["Gly, 40 mM", "Leu, 40 mM", "Phe, 40 mM", "Trp, 40 mM"]

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path="AminoAcids_40mM.xlsx",
            pandas_kwargs={"sheet_name": f"Sheet{id}"}
        )

        spectra = df.loc[1:, 4.5:].to_numpy().T
        raman_shifts = df.loc[1:, header[(int(id) - 1)]].to_numpy()
        concentration = np.array(df.columns.values[2:], dtype=float)

        return spectra, raman_shifts, concentration


    @staticmethod
    def __load_andriitrelin(
        id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the cells Raman spectra dataset.

        This dataset contains SERS spectra of various cell types (melanoma cells,
        melanocytes, fibroblasts) collected on gold nanourchins with different
        surface functionalizations.

        Args:
            id: The surface functionalization type ("COOH", "NH2", or "(COOH)2").

        Returns:
            A tuple of (spectra, raman_shifts, labels) arrays,
            or None if parsing fails.
        """
        file_handle = "andriitrelin/cells-raman-spectra"

        # Cell type labels (folder names)
        cell_types = [
            "A", "A-S", "G", "G-S", "HF", "HF-S",
            "ZAM", "ZAM-S", "MEL", "MEL-S", "DMEM", "DMEM-S"
        ]

        # Download the dataset first
        cache_path = dataset_download(file_handle)

        all_spectra = []
        all_labels = []
        raman_shifts = np.array([])

        for cell_type in cell_types:
            # Data is in the dataset_i subfolder
            file_path = os.path.join(cache_path, "dataset_i", cell_type, f"{id}.csv")

            if not os.path.exists(file_path):
                logger.warning(f"Warning: File not found: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path, header=None)
                data_np = df.to_numpy()

                raman_shifts = data_np[0, :]
                spectra = data_np[1:, :]

                all_spectra.append(spectra)
                all_labels.extend([cell_type] * spectra.shape[0])

            except Exception as e:
                logger.warning(f"Warning: Could not load {file_path}: {e}")
                continue

        if not all_spectra:
            return None


        spectra = np.vstack(all_spectra)
        
        if spectra.shape[1] < spectra.shape[0]:
            spectra = spectra.T

        labels = np.array(all_labels)

        return spectra, raman_shifts, labels


    # Note: __load_cancer_cells was removed as mathiascharconnet/cancer-cells-sers-spectra
    # requires special Kaggle consent. The same data is available via
    # andriitrelin_cells-raman-spectra which is loaded by __load_andriitrelin above.


    DATASETS = {
        "codina_diabetes_AGEs": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="AGEs",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina_raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina_diabetes_earLobe": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="earLobe",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina_raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina_diabetes_innerArm": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="innerArm",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina_raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina_diabetes_thumbNail": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="thumbNail",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina_raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "codina_diabetes_vein": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="vein",
            loader=__load_diabetes,
            metadata={
                "full_name" : "codina_raman-spectroscopy-of-diabetes",
                "source" : "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                "paper" : "https://doi.org/10.1364/BOE.9.004998",
                "description" : "This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy."
            }
        ),
        "sergioalejandrod_AminoAcids_glycine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="1",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod_raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "sergioalejandrod_AminoAcids_leucine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="2",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod_raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "sergioalejandrod_AminoAcids_phenylalanine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="3",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod_raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "sergioalejandrod_AminoAcids_tryptophan": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="4",
            loader=__load_sergioalejandrod,
            metadata={
                "full_name" : "sergioalejandrod_raman-spectroscopy",
                "source" : "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                "paper" : "https://doi.org/10.1021/acs.analchem.0c03015",
                "description" : "This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method."
            }
        ),
        "andriitrelin_cells-raman-spectra_COOH": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="COOH",
            loader=__load_andriitrelin,
            metadata={
                "full_name" : "andriitrelin_cells-raman-spectra",
                "source" : "https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra",
                "paper" : "https://doi.org/10.1016/j.snb.2020.127660",
                "description" : "SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with COOH moiety. Contains 12 cell type classes for classification."
            }
        ),
        "andriitrelin_cells-raman-spectra_NH2": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="NH2",
            loader=__load_andriitrelin,
            metadata={
                "full_name" : "andriitrelin_cells-raman-spectra",
                "source" : "https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra",
                "paper" : "https://doi.org/10.1016/j.snb.2020.127660",
                "description" : "SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with NH2 moiety. Contains 12 cell type classes for classification."
            }
        ),
        "andriitrelin_cells-raman-spectra_(COOH)2": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="(COOH)2",
            loader=__load_andriitrelin,
            metadata={
                "full_name" : "andriitrelin_cells-raman-spectra",
                "source" : "https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra",
                "paper" : "https://doi.org/10.1016/j.snb.2020.127660",
                "description" : "SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with (COOH)2 moiety. Contains 12 cell type classes for classification."
            }
        ),
        # Note: mathiascharconnet/cancer-cells-sers-spectra requires Kaggle consent.
        # The same data is available via andriitrelin_cells-raman-spectra above.
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None,
    ) -> str | None:
        """
        Download a Kaggle dataset to the local cache.

        Args:
            dataset_name: The name of the dataset to download (e.g., "codina_diabetes_AGEs").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        Kaggle cache directory (~/.cache/kagglehub).

        Returns:
            str | None: The path where the dataset was downloaded, or None if the
                        dataset is not available through this loader.
        """
        if not LoaderTools.is_dataset_available(dataset_name, KaggleLoader.DATASETS):
            logger.error(f"[!] Cannot download {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        logger.debug(f"Downloading Kaggle dataset: {dataset_name}")

        path = dataset_download(handle=dataset_name, path=cache_path)
        logger.debug(f"Dataset downloaded into {path}")

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
            dataset_name: The name of the dataset to load (e.g., "codina_diabetes_AGEs").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        Kaggle cache directory (~/.cache/kagglehub).

        Returns:
            RamanDataset | None: A RamanDataset object containing the spectral data,
                                 target values, and metadata, or None if loading fails.
        """
        if not LoaderTools.is_dataset_available(dataset_name, KaggleLoader.DATASETS):
            logger.error(f"[!] Cannot load {dataset_name} dataset with Kaggle loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        logger.debug(
            f"Loading Kaggle dataset from "
            f"{cache_path if cache_path else 'default folder (~/.cache/kagglehub)'}"
        )

        dataset_id = KaggleLoader.DATASETS[dataset_name].id

        data = KaggleLoader.DATASETS[dataset_name].loader(dataset_id)

        if data is not None:
            spectra, raman_shifts, concentrations = data
            return RamanDataset(
                metadata=KaggleLoader.DATASETS[dataset_name].metadata,
                name=dataset_name,
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=concentrations,
                task_type=KaggleLoader.DATASETS[dataset_name].task_type,
            )

        return data

    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(KaggleLoader)
