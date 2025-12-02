from typing import Optional, Tuple

import datasets
import pandas as pd
import numpy as np

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import LoaderTools


class HugLoader(ILoader):
    """
    A static class specified in providing datasets hosted on HuggingFace.
    """
    def __load_substarteMix(
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        df = pd.concat(
            [
                pd.DataFrame(data["train"]),
                pd.DataFrame(data["test"]),
                pd.DataFrame(data["validation"]),
            ],
            ignore_index=True,
        )

        end_data_index = len(df.columns.values) - 8

        raman_shifts = df.loc[:, :"3384.7"].to_numpy().T
        spectra = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, "Glucose":].to_numpy()

        return raman_shifts, spectra, concentrations


    def __load_EcoliFermentation(
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        df = pd.concat(
            [
                pd.DataFrame(data["train"]),
                pd.DataFrame(data["test"]),
                pd.DataFrame(data["validation"]),
            ],
            ignore_index=True,
        )

        end_data_index = len(df.columns.values) - 2

        raman_shifts = df.loc[:, :"3384.7"].to_numpy().T
        spectra = np.array(df.columns.values[:end_data_index])
        concentrations = df.loc[:, :"Glucose"].to_numpy()

        return raman_shifts, spectra, concentrations


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
        )
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> str | None:
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
    
        data = HugLoader.DATASETS[dataset_name].loader(dataDict)

        if data is not None:
            raman_shifts, spectra, concentrations = data
            return RamanDataset(
                data=raman_shifts,
                target=concentrations,
                spectra=spectra,
                metadata=HugLoader.DATASETS[dataset_name].metadata
            )
        
        return data


    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(HugLoader)
