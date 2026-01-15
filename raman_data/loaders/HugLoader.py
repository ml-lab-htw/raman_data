from typing import Optional, Tuple

import datasets
import pandas as pd
import numpy as np

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import LoaderTools


class HugLoader(ILoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on HuggingFace.

    This loader provides access to datasets stored on HuggingFace's dataset hub,
    handling download, caching, and formatting of the data into RamanDataset objects.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import HugLoader
        >>> dataset = HugLoader.load_dataset("chlange/SubstrateMixRaman")
        >>> HugLoader.list_datasets()
    """

    @staticmethod
    def __load_substarteMix(
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the SubstrateMixRaman dataset.

        Args:
            df: DataFrame containing the raw dataset with Raman spectra and concentrations.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        end_data_index = len(df.columns.values) - 8

        spectra = df.loc[:, :"3384.7"].to_numpy().T
        raman_shifts = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, "Glucose":].to_numpy()

        return spectra, raman_shifts, concentrations


    @staticmethod
    def __load_EcoliFermentation(
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the RamanSpectraEcoliFermentation dataset.

        Args:
            df: DataFrame containing the raw dataset with Raman spectra and glucose data.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        end_data_index = len(df.columns.values) - 2

        spectra = df.loc[:, :"3384.7"].to_numpy().T
        raman_shifts = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, :"Glucose"].to_numpy()

        return spectra, raman_shifts, concentrations


    @staticmethod
    def __load_FuleSpectra(
        df: pd.DataFrame
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parse and extract data from the FuelRamanSpectraBenchtop dataset.

        Args:
            df: DataFrame containing the raw dataset with fuel Raman spectra.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        end_data_index = len(df.columns.values) - 12

        spectra = df.loc[:, :"3801.0"].to_numpy().T
        raman_shifts = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, "Research Octane Number":].to_numpy()

        return spectra, raman_shifts, concentrations


    DATASETS = {
        "chlange/SubstrateMixRaman": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id=None,
            loader=__load_substarteMix,
            metadata={
                "full_name" : "chlange/SubstrateMixRaman",
                "source" : "https://huggingface.co/datasets/chlange/SubstrateMixRaman",
                "paper" : "https://dx.doi.org/10.2139/ssrn.5239248",
                "description" : "This dataset, designed for biotechnological applications, provides a valuable resource for calibrating models used in high-throughput bioprocess development, particularly for bacterial fermentations. It features Raman spectra of samples containing varying, statistically independent concentrations of eight key metabolites, along with mineral salt medium and antifoam."
            }
        ),
        "chlange/RamanSpectraEcoliFermentation": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id=None,
            loader=__load_EcoliFermentation,
            metadata={
                "full_name" : "chlange/RamanSpectraEcoliFermentation",
                "source" : "https://huggingface.co/datasets/chlange/RamanSpectraEcoliFermentation",
                "paper" : "https://doi.org/10.1002/bit.70006",
                "description" : "Dataset Card for Raman Spectra from High-Throughput Bioprocess Fermentations of E. Coli. Raman spectra were obtained during an E. coli fermentation process consisting of a batch and a glucose-limited feeding phase, each lasting about four hours. Samples were automatically collected hourly, centrifuged to separate cells from the supernatant, and the latter was used for both metabolite analysis and Raman measurements. Two Raman spectra of ten seconds each were recorded per sample, with cell removal improving metabolite signal quality. More details can be found in the paper https://doi.org/10.1002/bit.70006"
            }
        ),
        "chlange/FuelRamanSpectraBenchtop": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id=None,
            loader=__load_FuleSpectra,
            metadata={
                "full_name" : "chlange/FuelRamanSpectraBenchtop",
                "source" : "https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop",
                "paper" : "http://dx.doi.org/10.1021/acs.energyfuels.9b02944",
                "description" : "This dataset contains Raman spectra for the analysis and prediction of key parameters in commercial fuel samples (gasoline). It includes spectra of 179 fuel samples from various refineries."
            }
        )
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> str | None:
        """
        Download a HuggingFace dataset to the local cache.

        Args:
            dataset_name: The full name of the HuggingFace dataset (e.g., "chlange/SubstrateMixRaman").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        HuggingFace cache directory (~/.cache/huggingface).

        Returns:
            str | None: The path where the dataset was downloaded, or None if the
                        dataset is not available through this loader.
        """
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
    ) -> RamanDataset | None:
        """
        Load a HuggingFace dataset as a RamanDataset object.

        Downloads the dataset if not already cached, then parses it into
        a standardized RamanDataset format.

        Args:
            dataset_name: The full name of the HuggingFace dataset (e.g., "chlange/SubstrateMixRaman").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        HuggingFace cache directory (~/.cache/huggingface).

        Returns:
            RamanDataset | None: A RamanDataset object containing the spectral data,
                                 target values, and metadata, or None if loading fails.
        """
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

        df = pd.concat(
            [
                pd.DataFrame(dataDict["train"]),
                pd.DataFrame(dataDict["test"]),
                pd.DataFrame(dataDict["validation"]),
            ],
            ignore_index=True,
        )
    
        data = HugLoader.DATASETS[dataset_name].loader(df)

        if data is not None:
            spectra, raman_shifts, concentrations = data
            return RamanDataset(
                metadata=HugLoader.DATASETS[dataset_name].metadata,
                name=dataset_name,
                raman_shifts=raman_shifts,
                spectra=spectra,
                target=concentrations,
                task_type=HugLoader.DATASETS[dataset_name].task_type,
            )
        
        return data


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(HugLoader)
