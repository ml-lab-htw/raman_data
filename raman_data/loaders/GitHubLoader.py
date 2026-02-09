import logging
import os
from typing import Optional, List

import numpy as np
import pandas as pd

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE


class GitHubLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on GitHub repositories.

    Downloads datasets from GitHub repo archives.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "github")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.GitHub)

    DATASETS = {
        "covid19_salvia": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="covid19_salvia",
            name="Saliva COVID-19",
            loader=lambda cache_path: GitHubLoader._load_mind_dataset(cache_path, "covid_dataset", ["CTRL", "COV+", "COV-"]),
            metadata={
                "full_name": "Saliva COVID-19 Raman Dataset",
                "source": "https://github.com/dpiazza/Raman-Spectra-Data",
                "description": "Curated for non-invasive SARS-CoV-2 screening. Includes ~25 spectral replicates per subject from 101 patients (positive, negative symptomatic, and healthy controls) acquired from dried saliva drops using a 785 nm spectrometer.",
                "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                "citation": [
                    "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                ],
            }
        ),
        **{
            f"{disease.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Medical,
                id=f"{disease.lower()}",
                name=f"Saliva {disease}",
                loader=lambda cache_path, c=disease[0]: GitHubLoader._load_mind_dataset(cache_path, "pd_ad_dataset", [f"{c}D", "CTRL"]),
                metadata={
                    "full_name": f"Saliva Neurodegenerative Disease Raman Dataset ({disease})",
                    "source": "https://github.com/dpiazza/Raman-Spectra-Data",
                    "description": f"Raman spectra from dried saliva drops targeting {disease}'s Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment.",
                    "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                    "citation": [
                        "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                    ],
                }
            )
            for disease in ["Parkinson", "Alzheimer"]
        },
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from GitHub loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, GitHubLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.GitHub)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.GitHub)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        GitHubLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = GitHubLoader.DATASETS[dataset_name]

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
    def _load_mind_dataset(cache_path: str, dataset_subfolder: str, category_filter: List[str]):
        """
        Load MIND-Lab datasets (covid_dataset or pd_ad_dataset).

        The expected layout (inside dataset folder):
          <patient_id>/spectra.csv
                          /raman_shift.csv
                          /user_information.csv

        Returns: spectra, raman_shifts, targets, class_names
        """
        shared_root = os.path.join(os.path.dirname(cache_path), "mind_shared")
        shared_main = os.path.join(shared_root, "Raman-Spectra-Data-main")
        if os.path.isdir(shared_main) and os.listdir(shared_main):
            GitHubLoader.logger.debug(f"Using existing dataset folder at {shared_main}")
        else:
            zip_name = "Raman-Spectra-Data.zip"
            zip_file = os.path.join(shared_root, zip_name)

            if not os.path.exists(shared_root):
                GitHubLoader.logger.debug(f"Attempting to download dataset {dataset_subfolder} to {shared_root}")
                os.makedirs(shared_root, exist_ok=True)

                if not os.path.exists(zip_file):
                    LoaderTools.download(
                        url="https://github.com/MIND-Lab/Raman-Spectra-Data/archive/refs/heads/main.zip",
                        out_dir_path=shared_root,
                        out_file_name=zip_name
                    )

                LoaderTools.extract_zip_file_content(zip_file)

        # Iterate patient folders
        spectra_list = []
        raman_shifts_list = []
        targets_list = []
        categories = []

        dataset_dir = os.path.join(shared_main, dataset_subfolder)
        for entry in sorted(os.listdir(dataset_dir)):
            patient_dir = os.path.join(dataset_dir, entry)
            if not os.path.isdir(patient_dir):
                continue

            user_info_path = os.path.join(patient_dir, "user_information.csv")
            spectra_path = os.path.join(patient_dir, "spectra.csv")
            shifts_path = os.path.join(patient_dir, "raman_shift.csv")

            if not (os.path.exists(user_info_path) and os.path.exists(spectra_path) and os.path.exists(shifts_path)):
                GitHubLoader.logger.warning(f"[!] Skipping patient folder (missing files): {patient_dir}")
                continue

            try:
                ui = pd.read_csv(user_info_path)
            except Exception as e:
                GitHubLoader.logger.warning(f"[!] Failed to read user_information.csv for {patient_dir}: {e}")
                continue

            cat_col = next((c for c in ui.columns if c.lower() == "category"), None)
            if cat_col is None and len(ui.columns) >= 2:
                cat_col = ui.columns[1]
            if cat_col is None:
                cat_col = next((c for c in ui.columns if c.lower() == "label"), None)
            if cat_col is None:
                GitHubLoader.logger.warning(f"[!] No category/label column found in {user_info_path}; skipping")
                continue

            category = str(ui[cat_col].iloc[0])

            if category not in category_filter:
                continue

            categories.append(category)

            try:
                spectra_df = pd.read_csv(spectra_path, header=None)
                shifts = pd.read_csv(shifts_path, header=None).to_numpy().squeeze()
            except Exception as e:
                GitHubLoader.logger.warning(f"[!] Failed to read spectra/shift for {patient_dir}: {e}")
                continue

            for _, row in spectra_df.iterrows():
                row_arr = row.to_numpy(dtype=float)
                spectra_list.append(row_arr)
                raman_shifts_list.append(shifts)

        if len(spectra_list) == 0:
            raise Exception(f"[!] No spectra found in {dataset_dir}")

        unique_categories = sorted(list(set(categories)))
        cat_to_idx = {lab: i for i, lab in enumerate(unique_categories)}

        targets = []
        for entry in sorted(os.listdir(dataset_dir)):
            patient_dir = os.path.join(dataset_dir, entry)
            if not os.path.isdir(patient_dir):
                continue
            user_info_path = os.path.join(patient_dir, "user_information.csv")
            spectra_path = os.path.join(patient_dir, "spectra.csv")
            if not (os.path.exists(user_info_path) and os.path.exists(spectra_path)):
                continue
            try:
                ui = pd.read_csv(user_info_path)
                cat_col = next((c for c in ui.columns if c.lower() == "category"), ui.columns[2] if len(ui.columns) >= 3 else ui.columns[0])
                lab = str(ui[cat_col].iloc[0])
                sf = pd.read_csv(spectra_path, header=None)
                count = len(sf)
                targets.extend([cat_to_idx[lab]] * count)
            except Exception:
                continue

        targets = np.array(targets, dtype=int)

        first_rs = None
        if len(raman_shifts_list) > 0:
            try:
                first_rs = raman_shifts_list[0]
                all_equal = all(np.allclose(first_rs, rs) for rs in raman_shifts_list)
            except Exception:
                all_equal = False
        else:
            all_equal = False

        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra = np.stack([np.array(s, dtype=float) for s in spectra_list])
        else:
            # raman_shifts = [np.array(rs, dtype=float) for rs in raman_shifts_list]
            # spectra = [np.array(s, dtype=float) for s in spectra_list]
            raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)

        class_names = unique_categories

        return spectra, raman_shifts, targets, class_names
