import logging
from typing import Optional, Tuple, List

import datasets
import numpy as np
import pandas as pd

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import is_wavenumber
from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE, APPLICATION_TYPE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s:%(funcName)s: %(message)s')
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
        >>> dataset = HuggingFaceLoader.load_dataset("bioprocess_substrates")
        >>> HuggingFaceLoader.list_datasets()
    """

    DATASETS = {
        "bioprocess_substrates": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_substrates",
            name="Bioprocess Monitoring",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Bioprocess Monitoring Raman Dataset",
                "hf_key": "chlange/SubstrateMixRaman",
                "source": "https://huggingface.co/datasets/chlange/SubstrateMixRaman",
                "paper": "https://doi.org/10.1016/j.measurement.2025.118884",
                "description": "A benchmark dataset of 6,960 spectra featuring eight key metabolites (glucose, glycerol, acetate, etc.) sampled via a statistically independent uniform distribution. Designed to evaluate regression robustness against common bioprocess correlations, including background effects from mineral salts and antifoam."
            }
        ),
        "ecoli_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="ecoli_fermentation",
            name="E. Coli Fermentation",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "E. Coli Fermentation Raman Dataset",
                "hf_key": "chlange/RamanSpectraEcoliFermentation",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraEcoliFermentation",
                "paper": "https://doi.org/10.1002/bit.70006",
                "description": "Spectra captured during batch and fed-batch fermentation of E. coli. Measurements were performed on the supernatant using a 785 nm spectrometer to track glucose and acetate concentrations in a dynamic, high-throughput bioprocess environment."
            }
        ),
        "fuel_benchtop": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="fuel_benchtop",
            name="Gasoline Properties (Benchtop)",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Gasoline Properties Raman Dataset (Benchtop)",
                "hf_key": "chlange/FuelRamanSpectraBenchtop",
                "source": "https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop",
                "paper": "https://doi.org/10.1016/j.fuel.2018.09.006",
                "description": "Raman spectra from 179 commercial gasoline samples recorded using a benchtop 1064 nm FT-Raman system. Targets include Research Octane Number (RON), Motor Octane Number (MON), and oxygenated additive concentrations."
            }
        ),
        "fuel_handheld": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="FuelRamanSpectraHandheld",
            name="Gasoline Properties (Handheld)",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Gasoline Properties Raman Dataset (Handheld)",
                "hf_key": "HTW-KI-Werkstatt/FuelRamanSpectraHandheld",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/FuelRamanSpectraHandheld",
                "paper": "https://doi.org/10.1021/acs.energyfuels.9b02944",
                "description": "Counterpart to the benchtop fuel dataset, acquired from the same 179 samples using a handheld 785 nm spectrometer. Used for benchmarking model transferability across different hardware and wavelengths."
            }
        ),
        "yeast_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="yeast_fermentation",
            name="Yeast Fermentation",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Ethanolic Yeast Fermentation Raman Dataset",
                "hf_key": "HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations",
                "paper": "https://doi.org/10.1002/bit.27112",
                "description": "This dataset contains Raman spectra acquired during the continuous ethanolic fermentation of sucrose using Saccharomyces cerevisiae (Baker's yeast). To facilitate continuous processing and high-quality optical measurements, the yeast cells were immobilized in calcium alginate beads."
            }
        ),
        "ralstonia_fermentations": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="ralstonia_fermentations",
            name="R. eutropha Copolymer Fermentations",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "R. eutropha Copolymer Fermentation Raman Dataset",
                "hf_key": "HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations",
                "source": "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanSpectraRalstoniaFermentations",
                "paper": "https://doi.org/10.1016/B978-0-443-28824-1.50510-X",
                "description": "Monitoring of P(HB-co-HHx) copolymer synthesis in Ralstonia eutropha batch cultivations. Includes a hybrid mix of experimental and high-fidelity synthetic data to handle high multicollinearity between process variables."
            }
        ),
        "bioprocess_analytes_anton_532": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_anton_532",
            name="Bioprocess Analytes Anton 532",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Anton 532",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesAnton532",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesAnton532",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_anton_785": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_anton_785",
            name="Bioprocess Analytes Anton 785",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Anton 785",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesAnton785",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesAnton785",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_kaiser": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_kaiser",
            name="Bioprocess Analytes Kaiser",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Kaiser",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesKaiser",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesKaiser",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_metrohm": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_metrohm",
            name="Bioprocess Analytes Metrohm",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Metrohm",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesMetrohm",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesMetrohm",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_mettler_toledo": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_mettler_toledo",
            name="Bioprocess Analytes Mettler Toledo",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Mettler Toledo",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesMettlerToledo",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesMettlerToledo",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_tec5": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_tec5",
            name="Bioprocess Analytes Tec5",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Tec5",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesTec5",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesTec5",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_timegate": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_timegate",
            name="Bioprocess Analytes Timegate",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Timegate",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesTimegate",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesTimegate",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
        "bioprocess_analytes_tornado": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="bioprocess_analytes_tornado",
            name="Bioprocess Analytes Tornado",
            loader=lambda df: HuggingFaceLoader._load_chlange(df),
            metadata={
                "full_name": "Raman Spectra of Bioprocess Analytes Tornado",
                "hf_key": "chlange/RamanSpectraBioprocessAnalystesTornado",
                "source": "https://huggingface.co/datasets/chlange/RamanSpectraBioprocessAnalystesTornado",
                "paper": "https://doi.org/10.1016/j.saa.2025.125861",
                "description": "This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground truth concentration labels verified by enzymatic assays, reflecting the concentration ranges typically found in E. coli fermentation processes."
            }
        ),
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
        cache_path: Optional[str] = None,
    ) -> str | None:
        """
        Download a HuggingFace dataset to the local cache.

        Args:
            dataset_name: The full name of the HuggingFace dataset (e.g., "bioprocess_substrates").
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
        cache_path: Optional[str] = None,
        load_data: bool = True,
    ) -> RamanDataset | None:
        """
        Load a HuggingFace dataset as a RamanDataset object.

        Downloads the dataset if not already cached, then parses it into
        a standardized RamanDataset format.

        Args:
            dataset_name: The full name of the HuggingFace dataset (e.g., "bioprocess_substrates").
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

        dataset_key = HuggingFaceLoader.DATASETS[dataset_name].metadata["hf_key"]
        data_dict = datasets.load_dataset(path=dataset_key, cache_dir=cache_path)

        if load_data:
            splits = []
            if "train" in data_dict:
                splits.append(pd.DataFrame(data_dict["train"]))
            if "test" in data_dict:
                splits.append(pd.DataFrame(data_dict["test"]))
            if "validation" in data_dict:
                splits.append(pd.DataFrame(data_dict["validation"]))

            full_dataset_df = pd.concat(splits, ignore_index=True)

            data = HuggingFaceLoader.DATASETS[dataset_name].loader(full_dataset_df)
        else:
            data = None, None, None

        if data is not None:
            spectra, raman_shifts, concentrations, concentration_names = data
            return RamanDataset(
                info=HuggingFaceLoader.DATASETS[dataset_name],
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=concentrations,
                target_names=concentration_names,
            )
        
        return data

