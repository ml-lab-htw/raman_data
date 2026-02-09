import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, HASH_TYPE, APPLICATION_TYPE


class FigshareLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on Figshare.

    Downloads datasets via the Figshare API.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "figshare")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Figshare)

    DATASETS = {
        "pharmaceutical_ingredients": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="pharmaceutical_ingredients",
            name="Pharmaceutical Ingredients",
            loader=lambda cache_path: FigshareLoader._load_api(cache_path),
            metadata={
                "full_name": "Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development",
                "source": "https://springernature.figshare.com/ndownloader/articles/27931131/versions/1",
                "paper": "https://doi.org/10.1038/s41597-025-04848-6",
                "citation": [
                    "Flanagan, A.R., Glavin, F.G. Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development. Sci Data 12, 498 (2025)."
                ],
                "description": "A Raman spectral dataset comprising 3,510 spectra from 32 chemical substances. This dataset includes organic solvents and reagents commonly used in API development, along with information regarding the products in the XLSX, and code to visualise and perform technical validation on the data.",
            }
        ),
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from Figshare loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, FigshareLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Figshare)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        FigshareLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = FigshareLoader.DATASETS[dataset_name]

        if load_data:
            result = dataset_info.loader(dataset_cache_path)
            if result is None:
                raise FileNotFoundError(f"Could not load dataset {dataset_name}. Expected files may be missing. Please check logs for details.")
            spectra, raman_shifts, targets, class_names = result
        else:
            spectra = raman_shifts = targets = class_names = None

        return RamanDataset(
            info=dataset_info,
            raman_shifts=raman_shifts,
            spectra=spectra,
            targets=targets,
            target_names=class_names,
        )

    @staticmethod
    def fetch_figshare_metadata(article_id: int) -> dict:
        r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}")
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _load_api(cache_path):

        metadata = FigshareLoader.fetch_figshare_metadata(27931131)
        files = metadata["files"]

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        dataset_cache = os.path.join(
            cache_root,
            "pharmaceutical_ingredient",
        )
        os.makedirs(dataset_cache, exist_ok=True)

        for f in files:
            file_url = f["download_url"]
            file_name = f["name"]
            file_md5 = f.get("computed_md5")

            out_path = os.path.join(dataset_cache, file_name)

            # download only if missing
            if not os.path.exists(out_path):
                LoaderTools.download(
                    url=file_url,
                    out_dir_path=dataset_cache,
                    out_file_name=file_name,
                    hash_target=file_md5,
                    hash_type=HASH_TYPE.md5,
                    referer="https://figshare.com/",
                )

        spectra_path = os.path.join(dataset_cache, "raman_spectra_api_compounds.csv")
        product_info_path = os.path.join(dataset_cache, "API_Product_Information.xlsx")

        spectra_df = pd.read_csv(spectra_path)
        targets = spectra_df["label"]
        raman_shifts = np.array(spectra_df.keys()[: -1].astype(float))
        spectra = spectra_df.values[:, :-1].astype(float)
        # product_info_df = pd.read_excel(product_info_path)

        encoded_targets, target_names = encode_labels(targets)
        return spectra, raman_shifts, encoded_targets, target_names
