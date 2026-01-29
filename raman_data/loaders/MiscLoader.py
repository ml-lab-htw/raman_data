import glob
import logging
import os
import pickle
from typing import Optional, Tuple, List

import numpy as np
from scipy.io import loadmat
import spectrochempy as scp

import raman_data.loaders.helper.rruff as rruff
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.helper import organic
import pandas as pd
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR


class MiscLoader(BaseLoader):
    """
    Loader for miscellaneous Raman spectroscopy datasets.

    Currently supports datasets from the DeepeR paper (Horgan et al., 2021)
    hosted on OneDrive.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "misc")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Misc)

    DATASETS = {
        "deepr_denoising": DatasetInfo(
            task_type=TASK_TYPE.Denoising,
            id="deepr_denoising",
            name="DeepeR Denoising",
            loader=lambda df: MiscLoader._load_deepr_denoising(df),
            metadata={
                "full_name": "DeepeR Denoising Dataset",
                "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EqZaY-_FrGdImybIGuMCvb8Bo_YD1Bc9ATBxbLxdDIv0RA?e=5%3aHhLp91&fromShare=true&at=9",
                "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
                "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
                "description": "Raman spectral denoising dataset from DeepeR paper. Contains noisy input spectra and corresponding denoised target spectra for training deep learning denoising models.",
                "license": "MIT License"
            }
        ),
        "deepr_super_resolution": DatasetInfo(
            task_type=TASK_TYPE.SuperResolution,
            id="deepr_super_resolution",
            name="DeepeR Super-Resolution Dataset",
            loader=lambda df: MiscLoader._load_deepr_super_resolution(df),
            metadata={
                "full_name": "DeepeR Super-Resolution Dataset",
                "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EuIIZkQGtT5NgQcYO_SOzigB706Q8b0EddSLEDGUN22EbA?e=5%3axGyu4b&fromShare=true&at=9",
                "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
                "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
                "description": "Hyperspectral super-resolution dataset from DeepeR paper. Contains low-resolution input spectra and high-resolution target spectra for training super-resolution models.",
                "license": "MIT License"
            }
        ),
        "rruff_mineral_raw": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="rruff_mineral_raw",
            name="RRUFF Database (Raw)",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="mineral_r", align_output=True),
            metadata={
                "full_name": "RRUFF Database - Raw Spectra",
                "source": "https://rruff.info/",
                "paper": "https://doi.org/10.1515/9783110417104-003",
                "citation": [
                    "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                ],
                "description": "Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm).",
                "license": "See paper"
            }
        ),
        "rruff_mineral_preprocessed": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="rruff_mineral_preprocessed",
            name="RRUFF Database (Preprocessed)",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="mineral_p", align_output=True),
            metadata={
                "full_name": "RRUFF Database - Preprocessed Spectra",
                "source": "https://rruff.info/",
                "paper": "https://doi.org/10.1515/9783110417104-003",
                "citation": [
                    "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                ],
                "description": "Preprocessed Raman spectra for over 1,000 mineral species from the RRUFF Database, resampled to a common high-resolution sampling rate and truncated to their intersecting wavenumber range.",
                "license": "See paper"
            }
        ),
        "knowitall_organics_raw": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="knowitall_organics_raw",
            name="Organic Compounds (Raw)",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="organic_r", align_output=True),
            metadata={
                "full_name": "Organic Compounds Multi-Excitation Dataset - Raw",
                "source": "https://data.dtu.dk/api/files/36144495",
                "paper": "https://doi.org/10.1002/jrs.5750",
                "citation": [
                    "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                ],
                "description": "Raw Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data.",
                "license": "See paper"
            }
        ),
        "knowitall_organics_preprocessed": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="knowitall_organics_preprocessed",
            name="Organic Compounds (Preprocessed)",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="organic_p", align_output=False),
            metadata={
                "full_name": "Organic Compounds Multi-Excitation Dataset - Preprocessed",
                "source": "https://data.dtu.dk/api/files/36144495",
                "paper": "https://doi.org/10.1002/jrs.5750",
                "citation": [
                    "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                ],
                "description": "Preprocessed Raman spectra of organic compounds across multiple excitation sources, evaluating the generalization capabilities of deep neural networks on instrument-specific chemical sets.",
                "license": "See paper"
            }
        ),
        "mind_covid": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="mind_covid",
            name="Saliva COVID-19",
            loader=lambda df: MiscLoader._load_mind_dataset(df, "covid_dataset", ["CTRL", "COV+", "COV-"]),
            metadata={
                "full_name": "Saliva COVID-19 Raman Dataset",
                "source": "https://github.com/dpiazza/Raman-Spectra-Data",
                "description": "Curated for non-invasive SARS-CoV-2 screening. Includes ~25 spectral replicates per subject from 101 patients (positive, negative symptomatic, and healthy controls) acquired from dried saliva drops using a 785 nm spectrometer.",
                "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                "citation": [
                    "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                ],
                "license": "See source"
            }
        ),
        "mind_pd": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="mind_pd",
            name="Saliva Parkinson",
            loader=lambda df: MiscLoader._load_mind_dataset(df, "pd_ad_dataset", ["PD", "CTRL"]),
            metadata={
                "full_name": "Saliva Neurodegenerative Disease Raman Dataset (Parkinson)",
                "source": "https://github.com/dpiazza/Raman-Spectra-Data",
                "description": "Raman spectra from dried saliva drops targeting Parkinson's Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment.",
                "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                "citation": [
                    "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                ],
                "license": "See source"
            }
        ),
        "mind_ad": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="mind_ad",
            name="Saliva Alzheimer",
            loader=lambda df: MiscLoader._load_mind_dataset(df, "pd_ad_dataset", ["AD", "CTRL"]),
            metadata={
                "full_name": "Saliva Neurodegenerative Disease Raman Dataset (Alzheimer)",
                "source": "https://github.com/dpiazza/Raman-Spectra-Data",
                "description": "Raman spectra from dried saliva drops targeting Alzheimer's Disease (AD) vs. healthy controls. Serves as a liquid biopsy benchmark for identifying neurodegenerative pathology.",
                "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                "citation": [
                    "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                ],
                "license": "See source"
            }
        ),
        "csho33_bacteria": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="csho33_bacteria",
            name="Pathogenic Bacteria",
            loader=lambda df: MiscLoader._load_csho33_bacteria(df),
            metadata={
                "full_name": "Pathogenic Bacteria Raman Dataset",
                "source": "https://github.com/csho33/bacteria-ID",
                "paper": "https://doi.org/10.1038/s41467-019-12898-9",
                "citation": [
                    "Ho, C.-S., Jean, N., Hogan, C. A., et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nat Commun 10, 4927 (2019)."
                ],
                "description": "60,000 spectra from 30 clinically relevant bacterial and yeast isolates (including an MRSA/MSSA isogenic pair). Acquired with 633 nm illumination on gold-coated silica substrates with low SNR to simulate rapid clinical acquisition times.",
                "license": "See paper"
            }
        ),
        "rwth_acid_species": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="rwth_acid_species",
            name="Acid Species Concentrations",
            loader=lambda df: MiscLoader._load_rwth_acid_species(df),
            metadata={
                "full_name": "Inline Raman Spectroscopy and Indirect Hard Modeling for Concentration Monitoring of Dissociated Acid Species",
                "source": "https://publications.rwth-aachen.de/record/978266/files/Data_RWTH-2024-01177.zip",
                "paper": [
                    "https://doi.org/10.1177/0003702820973275",
                    "https://publications.rwth-aachen.de/record/978266"
                ],
                "citation": [
                    "Echtermeyer, Alexander Walter Wilhelm; Marks, Caroline; Mitsos, Alexander; Viell, Jörn. Inline Raman Spectroscopy and Indirect Hard Modeling for Concentration Monitoring of Dissociated Acid Species. Applied Spectroscopy, 2021, 75(5):506–519. DOI: 10.1177/0003702820973275."
                ],
                "description": "Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling.",
                "license": "See paper/source."
            }
        )
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def _load_deepr_denoising(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, None]]:
        """
        Load the DeepeR denoising dataset from cache.

        Args:
            cache_path: Path to the cached dataset directory.

        Returns:
            Tuple of (spectra, raman_shifts, targets) or None if files are missing.
        """
        required_files = {
            "train_inputs": "Train_Inputs.mat",
            "train_outputs": "Train_Outputs.mat",
            "test_inputs": "Test_Inputs.mat",
            "test_outputs": "Test_Outputs.mat",
            "axis": "axis.txt",
        }

        # Check all files exist
        for key, fname in required_files.items():
            file_path = os.path.join(cache_path, fname)
            if not os.path.exists(file_path):
                MiscLoader.logger.warning(f"[!] Missing file: {fname}")
                MiscLoader.logger.warning(f"[!] Please download manually from OneDrive and extract to: {cache_path}")
                return None

        try:

            # Load data
            axis = np.loadtxt(os.path.join(cache_path, required_files["axis"]))
            train_inputs = loadmat(os.path.join(cache_path, required_files["train_inputs"]))["Train_Inputs"]
            train_outputs = loadmat(os.path.join(cache_path, required_files["train_outputs"]))["Train_Outputs"]
            test_inputs = loadmat(os.path.join(cache_path, required_files["test_inputs"]))["Test_Inputs"]
            test_outputs = loadmat(os.path.join(cache_path, required_files["test_outputs"]))["Test_Outputs"]

            # Combine train and test sets
            inputs = np.vstack((train_inputs, test_inputs))
            outputs = np.vstack((train_outputs, test_outputs))

            return inputs, axis, outputs, None

        except Exception as e:
            MiscLoader.logger.error(f"[!] Error loading dataset: {e}")
            return None


    @staticmethod
    def _load_deepr_super_resolution(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load the DeepeR super-resolution dataset from cache.

        The super-resolution dataset contains hyperspectral Raman images rather
        than single spectra. It includes low-resolution input images and
        high-resolution target images for training super-resolution models.

        Files are expected to follow naming pattern: Cell-{HR/LR}_Norm_{wavelengths}-{id}.mat
        Example: Cell-HR_Norm_500-01-193-020.mat

        Args:
            cache_path: Path to the cached dataset directory.

        Returns:
            Tuple of (low_res_images, raman_shifts, high_res_images) or None if files are missing.
            Images are in format: (n_samples, height, width, n_wavelengths)
        """

        raise NotImplementedError


    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Download helper for miscellaneous datasets. Implements download for MIND-Lab datasets
        by fetching the GitHub repository zip and extracting it. Returns the dataset folder path
        inside the cache.
        """
        # Only implement downloading for the two MIND datasets here
        mind_map = {
            "mind_covid": "covid_dataset",
            "mind_pd_ad": "pd_ad_dataset",
        }

        if dataset_name not in mind_map:
            MiscLoader.logger.warning(f"[!] download_dataset not implemented for: {dataset_name}")
            return None

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Misc)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            MiscLoader.logger.error("[!] Cache root for MiscLoader is not set")
            return None

        shared_root = os.path.join(cache_root, "mind_shared")
        os.makedirs(shared_root, exist_ok=True)

        zip_name = "Raman-Spectra-Data.zip"
        zip_path = os.path.join(shared_root, zip_name)
        extracted_dir = os.path.join(shared_root, "Raman-Spectra-Data-main")

        # Download repo zip from GitHub (main branch)
        try:
            LoaderTools.download(
                url="https://github.com/MIND-Lab/Raman-Spectra-Data/archive/refs/heads/main.zip",
                out_dir_path=shared_root,
                out_file_name=zip_name
            )
        except Exception as e:
            MiscLoader.logger.error(f"[!] Failed to download MIND repo: {e}")
            return None

        LoaderTools.extract_zip_file_content(zip_path, unzip_target_subdir="Raman-Spectra-Data-main")

        # After extraction, the exact nesting may vary (extraction can create
        # an extra top-level folder). Search recursively under shared_root for
        # a directory matching the dataset_sub name and return its path.
        dataset_sub = mind_map[dataset_name]
        found = None
        for root, dirs, files in os.walk(shared_root):
            if dataset_sub in dirs:
                found = os.path.join(root, dataset_sub)
                break

        if found is None:
            # As a fallback, check the previously assumed path
            dataset_folder = os.path.join(extracted_dir, dataset_sub)
            if os.path.isdir(dataset_folder):
                found = dataset_folder

        if found is None:
            MiscLoader.logger.error(f"[!] Expected dataset folder not found after extraction under: {shared_root}")
            return None

        return found

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        """
        Load a miscellaneous dataset from cache.

        If the dataset is not cached, attempts to download it first.

        Args:
            dataset_name: Name of the dataset to load.
            cache_path: Custom cache directory. If None, uses default location.

        Returns:
            RamanDataset object or None if loading fails.
        """
        if not LoaderTools.is_dataset_available(dataset_name, MiscLoader.DATASETS):
            MiscLoader.logger.warning(f"[!] Cannot load {dataset_name} dataset with Miscellaneous loader")
            return None

        # Get or set cache path
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Misc)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        MiscLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        # Get dataset info and load data
        dataset_info = MiscLoader.DATASETS[dataset_name]

        if load_data:
            result = dataset_info.loader(dataset_cache_path)
            if result is None:
                return None
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
    def _load_dtu_split(cache_path: str, split: str, align_output: bool = True):
        shared_root = os.path.join(os.path.dirname(cache_path), "dtu_raman_shared")
        zip_path = os.path.join(shared_root, "public_dataset.zip")
        extracted_dir = os.path.join(shared_root, "public_dataset")
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
                    MiscLoader.logger.error(f"[!] Failed to remove corrupted zip: {e}")
            LoaderTools.download(
                url="https://data.dtu.dk/api/files/36144495/content",
                out_dir_path=shared_root,
                out_file_name="public_dataset.zip"
            )
            LoaderTools.extract_zip_file_content(
                zip_path,
                unzip_target_subdir="public_dataset"
            )
            if not LoaderTools.is_valid_zip(zip_path):
                raise CorruptedZipFileError(zip_path)

        # Map logical split → folder name in ZIP
        split_dirs = {
            "mineral_r": "mineral_raw",
            "mineral_p": "mineral_preprocess",
            "organic_r": "organic_raw",
            "organic_p": "organic_preprocess",
        }

        if split not in split_dirs:
            MiscLoader.logger.error(f"[!] Unknown DTU split: {split}")
            return None

        split_root = os.path.join(extracted_dir, split_dirs[split])
        if not os.path.isdir(split_root):
            MiscLoader.logger.error(f"[!] Expected directory not found: {split_root}")
            return None

        split_root_raw = split_root.replace("preprocess", "raw")
        if not os.path.isdir(split_root_raw):
            MiscLoader.logger.error(f"[!] Expected directory not found: {split_root_raw}")
            return None

        # Use original repo IO for mineral and organic splits
        if split in ("mineral_r", "mineral_p"):

            if split == "mineral_r":
                data = rruff.give_all_raw(split_root, print_info=True)
            else:
                csv_path = os.path.join(split_root, "excellent_unoriented_unoriented.csv")
                data = rruff.give_subset_of_spectrums(csv_path, None, "preprocess", print_info=True)

            class_names = list(data['name'].unique())
            name_to_idx = {name: i for i, name in enumerate(class_names)} if class_names else {}
            targets = data['name'].map(name_to_idx).to_numpy() if name_to_idx else np.zeros(len(data), dtype=int)

            spectra_path = os.path.join(split_root, 'spectra.obj')
            with open(spectra_path, "rb") as f:
                spectra_data = pickle.load(f)

            raman_shifts_list = [arr[:, 0] for arr in spectra_data]
            spectra_list = [arr[:, 1] for arr in spectra_data]

        elif split == "organic_r":

            spectra_list, raman_shifts_list, targets = organic.get_raw_data(split_root)
            class_names = [str(i) for i in range(len(np.unique(targets)))]  # TODO

        elif split == "organic_p":

            spectra_list, raman_shifts_list, targets = organic.get_preprocessed_data(split_root)
            class_names = [str(i) for i in range(len(np.unique(targets)))] # TODO

        else:
            raise ValueError(f"Unknown DTU split: {split}")

        if align_output:
            raman_shifts, spectra = MiscLoader.align_raman_shifts(raman_shifts_list, spectra_list)
        else:
            raman_shifts = raman_shifts_list
            spectra = spectra_list

        return spectra, raman_shifts, targets, class_names


    @staticmethod
    def align_raman_shifts(raman_shifts_list: list[np.ndarray], spectra_list: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        min_shift = np.max([rs[0] for rs in raman_shifts_list])
        max_shift = np.min([rs[-1] for rs in raman_shifts_list])
        frequency_steps = [rs[1] - rs[0] for rs in raman_shifts_list]
        min_step = min(frequency_steps)
        raman_shifts = np.arange(min_shift, max_shift, min_step)
        new_spectra_list = [np.interp(raman_shifts, rs, spec) for rs, spec in zip(raman_shifts_list, spectra_list)]
        spectra = np.stack(new_spectra_list)
        return raman_shifts, spectra

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
        # If user passed in a cache path that is not the actual dataset folder, try common locations
        # 1) cache_path itself
        # 2) parent/mind_shared/Raman-Spectra-Data-main/<dataset_subfolder>

        shared_root = os.path.join(os.path.dirname(cache_path), "mind_shared")
        shared_main = os.path.join(shared_root, "Raman-Spectra-Data-main", "Raman-Spectra-Data-main")
        if os.path.isdir(shared_main) and os.listdir(shared_main):
            MiscLoader.logger.debug(f"Using existing dataset folder at {shared_main}")
        else:
            zip_file = os.path.join(shared_root, "Raman-Spectra-Data.zip")

            if not os.path.exists(shared_root):
                MiscLoader.logger.debug(f"Attempting to download dataset {dataset_subfolder} to {shared_root}")
                # Construct dataset key matching DATASETS mapping, e.g. 'pd_ad_dataset' -> 'mind_pd_ad'
                dataset_key = f"mind_{dataset_subfolder.replace('_dataset', '')}"
                downloaded = MiscLoader.download_dataset(dataset_key, cache_path=os.path.dirname(cache_path))

                if not downloaded or not os.path.isdir(downloaded):
                    MiscLoader.logger.error(f"[!] Could not locate or download dataset folder for {dataset_subfolder}")
                    return None

            if os.path.exists(zip_file) and LoaderTools.is_valid_zip(zip_file):
                # Extract dataset folder from zip
                LoaderTools.extract_zip_file_content(zip_file, unzip_target_subdir="Raman-Spectra-Data-main")
            else:
                MiscLoader.logger.error(f"[!] Failed to extract dataset folder from zip: {zip_file}")
                return None

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
                MiscLoader.logger.warning(f"[!] Skipping patient folder (missing files): {patient_dir}")
                continue

            try:
                ui = pd.read_csv(user_info_path)
            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to read user_information.csv for {patient_dir}: {e}")
                continue

            # Prefer the 'category' column for class names (case-insensitive).
            # Fallbacks: positional second column, then 'label' column, then first column.
            cat_col = next((c for c in ui.columns if c.lower() == "category"), None)
            if cat_col is None and len(ui.columns) >= 2:
                cat_col = ui.columns[1]
            if cat_col is None:
                # fallback to label-like column
                cat_col = next((c for c in ui.columns if c.lower() == "label"), None)
            if cat_col is None:
                MiscLoader.logger.warning(f"[!] No category/label column found in {user_info_path}; skipping")
                continue

            category = str(ui[cat_col].iloc[0])

            if category not in category_filter:
                continue

            categories.append(category)

            # Read spectra and shifts
            try:
                spectra_df = pd.read_csv(spectra_path, header=None)
                shifts = pd.read_csv(shifts_path, header=None).to_numpy().squeeze()
            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to read spectra/shift for {patient_dir}: {e}")
                continue

            # For each spectrum (row) add to list and set target to label
            for _, row in spectra_df.iterrows():
                row_arr = row.to_numpy(dtype=float)
                spectra_list.append(row_arr)
                raman_shifts_list.append(shifts)
                # targets per spectrum will be assigned later after mapping labels to indices

        if len(spectra_list) == 0:
            MiscLoader.logger.error(f"[!] No spectra found in {dataset_dir}")
            return None

        # Map categories to class indices
        unique_categories = sorted(list(set(categories)))
        cat_to_idx = {lab: i for i, lab in enumerate(unique_categories)}

        # Re-read user info per patient to create targets aligned with spectra_list –
        # simpler: we built labels list per patient order, but need per-spectrum targets.
        # The earlier loop appended spectra in patient order and appended the patient's label once to labels list;
        # to construct targets we will walk base_dir again and collect counts per patient.
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


        # Guard: ensure first_rs is defined for later use in all_equal branch
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
            raman_shifts = [np.array(rs, dtype=float) for rs in raman_shifts_list]
            spectra = [np.array(s, dtype=float) for s in spectra_list]

        class_names = unique_categories

        return spectra, raman_shifts, targets, class_names

    @staticmethod
    def _load_csho33_bacteria(cache_path: str):
        """
        Download and load the csho33 bacteria dataset (Dropbox zip) expected layout:
          <extracted_root>/data/X_finetune.npy
                                 y_finetune.npy
                                 X_test.npy
                                 y_test.npy
                                 X_2018clinical.npy
                                 y_2018clinical.npy
                                 X_2019clinical.npy
                                 y_2019clinical.npy

        Returns spectra, raman_shifts, targets, class_names
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            MiscLoader.logger.error("[!] Cache root for MiscLoader is not set")
            return None

        shared_root = os.path.join(cache_root, "csho33_bacteria")
        os.makedirs(shared_root, exist_ok=True)

        zip_name = "csho33_bacteria.zip"
        zip_path = os.path.join(shared_root, zip_name)
        # Use Dropbox direct-download URL (dl=1)
        dl_url = "https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&dl=1"

        # Files we expect to find somewhere under the extracted tree
        required = [
            "X_finetune.npy", "y_finetune.npy",
            "X_test.npy", "y_test.npy",
            "X_2018clinical.npy", "y_2018clinical.npy",
            "X_2019clinical.npy", "y_2019clinical.npy",
        ]

        def find_dir_with_files(root_dir: str, filenames: list[str]) -> str | None:
            for root, dirs, files in os.walk(root_dir):
                if all(fname in files for fname in filenames):
                    return root
            return None

        # If the extracted folder already exists and contains our files, reuse it.
        extracted_dir = os.path.join(shared_root, "csho33_bacteria")
        data_root = find_dir_with_files(extracted_dir, required) if os.path.isdir(extracted_dir) else None

        # If not found under the extracted dir, try scanning the shared_root and the wider cache_root
        if data_root is None:
            data_root = find_dir_with_files(shared_root, required)
        if data_root is None:
            data_root = find_dir_with_files(cache_root, required)

        # If no data files found yet, look for any existing zip file anywhere under the misc cache root and reuse it.
        existing_zip = None
        if data_root is None:
            for root, dirs, files in os.walk(cache_root):
                for f in files:
                    if f.lower().endswith('.zip'):
                        candidate = os.path.join(root, f)
                        if LoaderTools.is_valid_zip(candidate):
                            existing_zip = candidate
                            MiscLoader.logger.debug(f"Found existing zip at {existing_zip}; will attempt to extract")
                            break
                if existing_zip:
                    break

        # If we have an existing zip, extract it into the canonical extraction folder and search for the data
        if data_root is None and existing_zip is not None:
            try:
                LoaderTools.extract_zip_file_content(existing_zip, unzip_target_subdir="csho33_bacteria")
            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to extract existing zip {existing_zip}: {e}")
            data_root = find_dir_with_files(extracted_dir, required) or find_dir_with_files(shared_root, required) or find_dir_with_files(cache_root, required)

        # If still not found, do a fresh download using the expected filename into shared_root
        if data_root is None:
            try:
                LoaderTools.download(url=dl_url, out_dir_path=shared_root, out_file_name=zip_name)
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to download csho33 dataset: {e}")
                return None

            extracted_dir = LoaderTools.extract_zip_file_content(zip_path, unzip_target_subdir="csho33_bacteria")
            if extracted_dir is None:
                MiscLoader.logger.error("[!] Failed to extract csho33 zip")
                return None

            data_root = find_dir_with_files(extracted_dir, required) or find_dir_with_files(shared_root, required) or find_dir_with_files(cache_root, required)

        if data_root is None:
            MiscLoader.logger.error(f"[!] Could not find the expected files after extraction under: {shared_root} or {cache_root}")
            return None

        try:
            # Load arrays; some archives include extra files like reference or wavenumbers
            # Try to load an optional wavenumbers.npy if present
            X_f = np.load(os.path.join(data_root, "X_finetune.npy"))
            y_f = np.load(os.path.join(data_root, "y_finetune.npy"))
            X_t = np.load(os.path.join(data_root, "X_test.npy"))
            y_t = np.load(os.path.join(data_root, "y_test.npy"))
            X_2018 = np.load(os.path.join(data_root, "X_2018clinical.npy"))
            y_2018 = np.load(os.path.join(data_root, "y_2018clinical.npy"))
            X_2019 = np.load(os.path.join(data_root, "X_2019clinical.npy"))
            y_2019 = np.load(os.path.join(data_root, "y_2019clinical.npy"))

            # optional files
            X_r = None
            y_r = None
            raman_shifts = None
            if os.path.exists(os.path.join(data_root, "X_reference.npy")):
                X_r = np.load(os.path.join(data_root, "X_reference.npy"))
            if os.path.exists(os.path.join(data_root, "y_reference.npy")):
                y_r = np.load(os.path.join(data_root, "y_reference.npy"))
            if os.path.exists(os.path.join(data_root, "wavenumbers.npy")):
                raman_shifts = np.load(os.path.join(data_root, "wavenumbers.npy"))
        except Exception as e:
            MiscLoader.logger.error(f"[!] Failed to load numpy arrays: {e}")
            return None

        # Concatenate datasets in order finetune, (reference if present), test, 2018, 2019
        parts_X = [X_f]
        parts_y = [y_f]
        if X_r is not None and y_r is not None:
            parts_X.append(X_r)
            parts_y.append(y_r)
        parts_X.extend([X_t, X_2018, X_2019])
        parts_y.extend([y_t, y_2018, y_2019])

        X = np.vstack(parts_X)
        y = np.concatenate(parts_y)

        # If raman_shifts missing, create default
        if raman_shifts is None:
            if X.ndim == 2:
                raman_shifts = np.arange(X.shape[1], dtype=float)
            else:
                raman_shifts = [np.arange(x.shape[0], dtype=float) for x in X]

        # Determine class names from unique labels
        unique = sorted(list(map(str, np.unique(y))))

        return X.astype(float), raman_shifts, y.astype(int), unique

    @staticmethod
    def _load_rwth_acid_species(cache_path: str):
        """
        Download and load the RWTH acid species dataset.
        Returns spectra, raman_shifts, targets, class_names (acids).
        """
        dataset_url = "https://publications.rwth-aachen.de/record/978266/files/Data_RWTH-2024-01177.zip?version=1"
        zip_name = "Data_RWTH-2024-01177.zip"
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            MiscLoader.logger.error("[!] Cache root for MiscLoader is not set")
            return None
        shared_root = os.path.join(cache_root, "rwth_acid_species")
        os.makedirs(shared_root, exist_ok=True)
        zip_path = os.path.join(shared_root, zip_name)
        # Download if not present
        if not os.path.exists(zip_path) or not LoaderTools.is_valid_zip(zip_path):
            try:
                LoaderTools.download(url=dataset_url, out_dir_path=shared_root, out_file_name=zip_name)
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to download RWTH acid species dataset: {e}")
                return None
        # Extract if not already extracted
        extracted_dir = os.path.join(shared_root, "Data_RWTH-2024-01177")
        if not os.path.isdir(extracted_dir):
            try:
                LoaderTools.extract_zip_file_content(zip_path)
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to extract RWTH acid species zip: {e}")
                return None
        # Parse subfolders for each acid system
        acid_dirs = [d for d in os.listdir(extracted_dir) if os.path.isdir(os.path.join(extracted_dir, d))]
        spectra_list = []
        raman_shifts_list = []
        targets = []
        class_names = []
        acid_to_idx = {}


        for acid_dir in acid_dirs:
            acid_path = os.path.join(extracted_dir, acid_dir)
            class_names.append(acid_dir)

            spectra_files = glob.glob(os.path.join(acid_path, "*.spc"), recursive=True)

            for file in spectra_files:
                scp_dataset = scp.read_spc(file)
                for spec in scp_dataset:
                    print(f"{spec.name} : {spec.shape}")

            # TODO

        if len(spectra_list) == 0:
            MiscLoader.logger.error(f"[!] No spectra found in {extracted_dir}")
            return None
        # Align raman shifts if possible
        first_rs = raman_shifts_list[0]
        all_equal = all(np.allclose(first_rs, rs) for rs in raman_shifts_list)
        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra = np.stack([np.array(s, dtype=float) for s in spectra_list])
        else:
            raman_shifts, spectra = MiscLoader.align_raman_shifts(raman_shifts_list, spectra_list)
        targets = np.array(targets, dtype=int)
        return spectra, raman_shifts, targets, class_names
