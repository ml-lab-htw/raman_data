import logging
import os
from typing import Optional, Tuple

import gdown
import numpy as np

from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE


class GoogleDriveLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on Google Drive.

    Downloads datasets via gdown from shared Google Drive links.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "gdrive")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.GoogleDrive)

    DATASETS = {
        **{
            f"rruff_mineral_{processed.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.MaterialScience,
                id=f"rruff_mineral_{processed.lower()}",
                name=f"RRUFF Database ({processed})",
                loader=lambda cache_path, p=processed: GoogleDriveLoader._load_onewarmheart(
                    cache_path, split=f"mineral_{p.lower()}"),
                metadata={
                    "full_name": f"RRUFF Database - {processed} Spectra",
                    "source": "https://rruff.info/",
                    "paper": "https://doi.org/10.1515/9783110417104-003",
                    "citation": [
                        "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                    ],
                    "description": "Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm).",
                }
            )
            for processed in ["Raw", "Preprocess"]
        },
        **{
            f"organic_compounds_{processed.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Chemical,
                id=f"organic_compounds_{processed.lower()}",
                name=f"Organic Compounds ({processed})",
                loader=lambda cache_path, p=processed: GoogleDriveLoader._load_onewarmheart(
                    cache_path, split=f"organic_{p.lower()}"),
                metadata={
                    "full_name": f"Organic Compounds Multi-Excitation Dataset - {processed}",
                    "source": "https://data.dtu.dk/api/files/36144495",
                    "paper": "https://doi.org/10.1002/jrs.5750",
                    "citation": [
                        "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                    ],
                    "description": f"{processed} Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data.",
                }
            )
            for processed in ["Raw", "Preprocess"]
        },
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from Google Drive loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, GoogleDriveLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.GoogleDrive)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.GoogleDrive)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        GoogleDriveLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = GoogleDriveLoader.DATASETS[dataset_name]

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
    def _load_onewarmheart(cache_path: str, split: str):
        # note that there is another repo loading these datasets differently:
        # https://github.com/lyn1874/raman_spectra_matching_with_contrastive_learning

        shared_root = os.path.join(os.path.dirname(cache_path), "dtu_raman_shared")
        zip_path = os.path.join(shared_root, "public_dataset.zip")
        extracted_dir = os.path.join(shared_root, "data")
        os.makedirs(shared_root, exist_ok=True)

        # Download & extract if needed
        if (
                not os.path.exists(extracted_dir)
                or not os.path.exists(zip_path)
                or not LoaderTools.is_valid_zip(zip_path)
        ):
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception as e:
                    GoogleDriveLoader.logger.error(f"[!] Failed to remove corrupted zip: {e}")

            file_id = "1X5OAdugcHVOF6k9WaWwCAu_IFh_7OwzR"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, zip_path, quiet=False)

            LoaderTools.extract_zip_file_content(zip_path)

            if not LoaderTools.is_valid_zip(zip_path):
                raise CorruptedZipFileError(zip_path)

        if "mineral" in split:
            subfolder = "RRUFF_database"
        elif "organic" in split:
            subfolder = "organics_database"
        else:
            raise ValueError(f"Unknown split name: {split}")

        split_root = os.path.join(extracted_dir, subfolder)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"[!] Expected directory not found: {split_root}")

        if "mineral" in split:
            if "raw" in split:
                raman_shifts_list = np.load(os.path.join(split_root, "xx_raw.npy"), allow_pickle=True).tolist()
                spectra_list = np.load(os.path.join(split_root, "xy_raw.npy"), allow_pickle=True).tolist()
                targets = np.load(os.path.join(split_root, "yclass.npy"), allow_pickle=True).tolist()

                raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)
            elif "preprocess" in split:
                # see: https://github.com/onewarmheart/Raman/blob/master/code-ZR/code/preprocess/rruff_interpolate.py
                raman_shifts = np.linspace(0, 1700, 1024)
                spectra_train = np.load(os.path.join(split_root, "after-preprocess", "xy_train.npy"), allow_pickle=True)
                spectra_val = np.load(os.path.join(split_root, "after-preprocess", "xy_val.npy"), allow_pickle=True)
                spectra = np.concatenate([spectra_train, spectra_val], axis=0)
                targets_train = np.load(os.path.join(split_root, "after-preprocess", "yclass_train.npy"), allow_pickle=True).tolist()
                targets_val = np.load(os.path.join(split_root, "after-preprocess", "yclass_val.npy"), allow_pickle=True).tolist()
                targets = targets_train + targets_val
            else:
                raise ValueError(f"Unknown split type in name: {split}")
        elif "organic" in split:
            if "raw" in split:
                raman_shifts_list = np.load(os.path.join(split_root, "xx.npy"), allow_pickle=True).tolist()
                spectra_list = np.load(os.path.join(split_root, "xy.npy"), allow_pickle=True).tolist()
                targets = np.load(os.path.join(split_root, "yclass.npy"), allow_pickle=True).tolist()

                raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)
            elif "preprocess" in split:
                # see: https://github.com/onewarmheart/Raman/blob/master/code-ZR/code/preprocess/organics_interpolate.py
                raman_shifts = np.linspace(200, 3700, 1100)[:1024] # this is from the original repo. idk why they do this so complicated
                spectra = np.load(os.path.join(split_root, "after-preprocess", "xy.npy"), allow_pickle=True)
                targets = np.load(os.path.join(split_root, "after-preprocess","yclass.npy"), allow_pickle=True).tolist()

        else:
            raise ValueError(f"Unknown split name: {split}")

        class_names = targets.unique().tolist() if hasattr(targets, "unique") else list(set(targets))
        return spectra, raman_shifts, targets, class_names
