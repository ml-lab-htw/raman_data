from typing import Optional, Tuple, List
import logging

import datasets
import pandas as pd
import numpy as np

from raman_data.loaders.utils import is_wavenumber
from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on HuggingFace.

    This loader provides access to datasets stored on HuggingFace's dataset hub,
    handling download, caching, and formatting of the data into RamanDataset objects.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import HuggingFaceLoader
        >>> dataset = HuggingFaceLoader.load_dataset("chlange/SubstrateMixRaman")
        >>> HuggingFaceLoader.list_datasets()
    """

    DATASETS = {
        "chlange/SubstrateMixRaman": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="chlange_SubstrateMixRaman",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "chlange/SubstrateMixRaman",
                "source": "https://huggingface.co/datasets/chlange/SubstrateMixRaman",
                "paper": "https://dx.doi.org/10.2139/ssrn.5239248",
                "description": "This dataset, designed for biotechnological applications, provides a valuable resource for calibrating models used in high-throughput bioprocess development, particularly for bacterial fermentations. It features Raman spectra of samples containing varying, statistically independent concentrations of eight key metabolites, along with mineral salt medium and antifoam."
            }
        ),
        "chlange/RamanSpectraEcoliFermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="chlange_RamanSpectraEcoliFermentation",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "chlange/RamanSpectraEcoliFermentation",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraEcoliFermentation",
                "paper": "https://doi.org/10.1002/bit.70006",
                "description": "Dataset Card for Raman Spectra from High-Throughput Bioprocess Fermentations of E. Coli. Raman spectra were obtained during an E. coli fermentation process consisting of a batch and a glucose-limited feeding phase, each lasting about four hours. Samples were automatically collected hourly, centrifuged to separate cells from the supernatant, and the latter was used for both metabolite analysis and Raman measurements. Two Raman spectra of ten seconds each were recorded per sample, with cell removal improving metabolite signal quality. More details can be found in the paper https://doi.org/10.1002/bit.70006"
            }
        ),
        "chlange/FuelRamanSpectraBenchtop": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="chlange_FuelRamanSpectraBenchtop",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "chlange/FuelRamanSpectraBenchtop",
                "source": "https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop",
                "paper": "http://dx.doi.org/10.1021/acs.energyfuels.9b02944",
                "description": "This dataset contains Raman spectra for the analysis and prediction of key parameters in commercial fuel samples (gasoline). It includes spectra of 179 fuel samples from various refineries."
            }
        ),
        "HTW-KI-Werkstatt/FuelRamanSpectraHandheld": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="HTW-KI-Werkstatt_FuelRamanSpectraHandheld",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "HTW-KI-Werkstatt/FuelRamanSpectraHandheld",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/FuelRamanSpectraHandheld",
                "paper": "",
                "description": "Handheld Raman spectra for fuel analysis. Structure similar to FuelRamanSpectraBenchtop."
            }
        ),
        "HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="HTW-KI-Werkstatt_RamanSpectraRalstoniaFermentations",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations",
                "paper": "",
                "description": "Raman spectra collected during Ralstonia fermentations. Dataset structure matches HTW-KI-Werkstatt/FuelRamanSpectraHandheld (wavenumber columns + metadata columns)."
            }
        )
    }

    @staticmethod
    def _load_chlange(
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List] | None:
        """
        Parse and extract data from the RamanSpectraEcoliFermentation dataset.

        Args:
            df: DataFrame containing the raw dataset with Raman spectra and glucose data.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """

        all_features = list(df.keys())
        wavenumber_cols = [col for col in all_features if is_wavenumber(col)]
        substance_cols = [col for col in all_features if not is_wavenumber(col)]
        raman_shifts = np.array([float(wn) for wn in wavenumber_cols])
        spectra = df[wavenumber_cols]
        concentrations = df[substance_cols]
        concentration_names = list(concentrations.columns)

        return spectra.to_numpy(), raman_shifts, concentrations.to_numpy(), concentration_names

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

        if not LoaderTools.is_dataset_available(dataset_name, HuggingFaceLoader.DATASETS):
            logger.error(f"[!] Cannot download {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        logger.debug(f"Downloading HuggingFace dataset: {dataset_name}")

        datasets.load_dataset(
            path=dataset_name,
            cache_dir=cache_path
        )

        cache_path = cache_path if cache_path else "~/.cache/huggingface"
        logger.debug(f"Dataset downloaded into {cache_path}")

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
                                 targets values, and metadata, or None if loading fails.
        """

        if not LoaderTools.is_dataset_available(dataset_name, HuggingFaceLoader.DATASETS):
            logger.error(f"[!] Cannot load {dataset_name} dataset with HuggingFace loader")
            return

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.HuggingFace)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.HuggingFace)

        logger.debug(
            f"Loading HuggingFace dataset from "
            f"{cache_path if cache_path else 'default folder (~/.cache/huggingface)'}"
        )

        data_dict = datasets.load_dataset(path=dataset_name, cache_dir=cache_path)

        splits = []
        if "train" in data_dict:
            splits.append(pd.DataFrame(data_dict["train"]))
        if "test" in data_dict:
            splits.append(pd.DataFrame(data_dict["test"]))
        if "validation" in data_dict:
            splits.append(pd.DataFrame(data_dict["validation"]))

        full_dataset_df = pd.concat(splits, ignore_index=True)

        data = HuggingFaceLoader.DATASETS[dataset_name].loader(full_dataset_df)

        if data is not None:
            spectra, raman_shifts, concentrations, concentration_names = data
            return RamanDataset(
                metadata=HuggingFaceLoader.DATASETS[dataset_name].metadata,
                name=dataset_name,
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=concentrations,
                target_names=concentration_names,
                task_type=HuggingFaceLoader.DATASETS[dataset_name].task_type,
            )
        
        return data

