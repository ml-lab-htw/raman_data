from typing import Optional, Tuple
import logging

import os, requests
import pandas as pd
import numpy as np

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import  LoaderTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZenodoLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on Zenodo.

    This loader provides access to datasets stored on the Zenodo research
    data repository, handling download, caching, and formatting of the data
    into RamanDataset objects.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import ZenodoLoader
        >>> dataset = ZenodoLoader.load_dataset("sugar_mixtures")
        >>> ZenodoLoader.list_datasets()
    """

    @staticmethod
    def __load_10779223(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the sugar_mixtures Raman dataset (Zenodo ID: 10779223).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (raman_shifts, spectra, concentrations) arrays,
            or None if parsing fails.
        """
        zip_filename = "Raw data.zip"

        try:
            data_dir = LoaderTools.extract_zip_file_content(
                os.path.join(cache_path, "10779223", zip_filename),
                zip_filename.split(".")[0]
            )
        except CorruptedZipFileError as e:
            logger.error(
                f"There seems to be an issue with dataset '10779223/sugar_mixtures'. \n"
            )
            return None

        if data_dir is None:
            logger.error(
                f"There seems to be no file of dataset '10779223/sugar_mixtures'.\n"
            )
            return None

        data_folder_parent = os.path.join(
            data_dir,
            "Raw data",
            "Experimental data from sugar mixtures",
            "Raw datasets for analyses"
        )

        # load the data file
        snr = "Low SNR"  # TODO implement snr selection
        data_folder = os.path.join(data_folder_parent, snr)
        
        # read spectra intensity data with pandas
        data_path = os.path.join(data_folder, "data.pkl")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find data.pkl in {data_path}")

        spectra = pd.read_pickle(data_path)

        # read raman shifts (wavenumbers) with pandas
        raman_shifts_path = os.path.join(data_folder, "spectral_axis.pkl")
        if not os.path.isfile(raman_shifts_path):
            raise FileNotFoundError(
                f"Could not find spectral_axis.pkl in {raman_shifts_path}"
            )

        raman_shifts = pd.read_pickle(raman_shifts_path)

        # read gt with pandas
        gt_path = os.path.join(data_folder, "gt_endmembers.pkl")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Could not find gt_endmembers.pkl in {gt_path}")

        concentrations = pd.read_pickle(gt_path).T

        return spectra, raman_shifts, concentrations


    @staticmethod
    def __load_256329(cache_path: str) -> np.ndarray | None:
        """
        Parse and extract data from the volumetric cells Raman dataset (Zenodo ID: 256329).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.

        Note:
            This method is not yet implemented.
        """
        raise NotImplementedError

    @staticmethod
    def __load_7644521(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the wheat_lines Raman dataset (Zenodo ID: 7644521).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        # data field names in the mat file
        data_keys = ["COM", "COM_125mM", "ML1_125mM", "ML2_125mM"]

        # load data file
        data_path = os.path.join(cache_path, "7644521", "Data.mat")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find Data.mat in {data_path}")

        # read content
        file_content = LoaderTools.read_mat_file(data_path)
        if file_content == None:
            logger.error(
                f"There was an error while reading the dataset '7644521/wheat_lines'.\n"
            )
            return None

        # raman shifts (wavenumbers/x-axis)
        raman_shifts = file_content["Calx"].squeeze()
        spectra_list = []
        concentrations = []

        # spectra intensity data
        for idx, key in enumerate(data_keys):
            data_row = file_content[key]
            spectra_list.append(data_row)
            concentrations.append(np.repeat(idx, data_row.shape[0]))

        spectra = np.concatenate(spectra_list)
        concentrations = np.concatenate(concentrations)

        return spectra, raman_shifts, concentrations


    @staticmethod
    def __load_3572359(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the adenine SERS dataset (Zenodo ID: 3572359).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        # load data file
        data_path = os.path.join(cache_path, "3572359", "ILSdata.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find ILSdata.csv in {data_path}")

        df = pd.read_csv(data_path)
        concentrations = df.pop("conc").to_numpy()
        raman_shifts = np.array(df.columns.values[8:], dtype=int)
        spectra = df.loc[:, "400":].to_numpy()

        return spectra, raman_shifts, concentrations


    __BASE_URL = "https://zenodo.org/api/records/ID/files-archive"
    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "zenodo")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Zenodo)

    DATASETS = {
        "sugar": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="10779223",
            name="Sugar Mixtures",
            loader=__load_10779223,
            metadata={
                "full_name": "Research data supporting \"Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders\"",
                "source": "https://doi.org/10.5281/zenodo.10779223",
                "paper": "https://doi.org/10.1073/pnas.2407439121",
                "description": "Experimental and synthetic Raman data used in Georgiev et al., PNAS (2024) DOI:10.1073/pnas.2407439121."
            }
        ),
        "wheat": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="7644521",
            name="Wheat Lines",
            loader=__load_7644521,
            metadata={
                "full_name": "DIFFERENTIATION OF ADVANCED GENERATION MUTANT wheat_lines: CONVENTIONAL TECHNIQUES VERSUS RAMAN SPECTROSCOPY",
                "source": "https://doi.org/10.5281/zenodo.7644521",
                "paper": "https://doi.org/10.1016/j.foodchem.2023.134703",
                "description": "Raman spectroscopy data used to differentiate between advanced generation mutant wheat lines and their parental cultivars."
            }
        ),
        "adenine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="3572359",
            name="Adenine",
            loader=__load_3572359,
            metadata={
                "full_name": "Surface-Enhanced Raman Spectroscopy (SERS) dataset of adenine",
                "source": "https://doi.org/10.5281/zenodo.3572359",
                "paper": "https://doi.org/10.1021/acsami.9b17424",
                "description": "SERS spectra of adenine molecules on silver nanoparticles, with varying concentrations."
            }
        )
    }
