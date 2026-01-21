import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat

import raman_data.loaders.helper.rruff as rruff
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR


class MiscLoader(BaseLoader):
    """
    Loader for miscellaneous Raman spectroscopy datasets.

    Currently supports datasets from the DeepeR paper (Horgan et al., 2021)
    hosted on OneDrive.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "misc")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Misc)

    # Define DATASETS after class definition to avoid NameError
    DATASETS = {
        # "deepr_denoising": DatasetInfo(
        #     task_type=TASK_TYPE.Denoising,
        #     id="deepr_denoising",
        #     loader=lambda df: MiscLoader._load_deepr_denoising(df),
        #     metadata={
        #         "full_name": "DeepeR Denoising Dataset",
        #         "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EqZaY-_FrGdImybIGuMCvb8Bo_YD1Bc9ATBxbLxdDIv0RA?e=5%3aHhLp91&fromShare=true&at=9",
        #         "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
        #         "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
        #         "description": "Raman spectral denoising dataset from DeepeR paper. Contains noisy input spectra and corresponding denoised target spectra for training deep learning denoising models.",
        #         "license": "MIT License"
        #     }
        # ),
        # "deepr_super_resolution": DatasetInfo(
        #     task_type=TASK_TYPE.SuperResolution,
        #     id="deepr_super_resolution",
        #     loader=lambda df: MiscLoader._load_deepr_super_resolution(df),
        #     metadata={
        #         "full_name": "DeepeR Super-Resolution Dataset",
        #         "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EuIIZkQGtT5NgQcYO_SOzigB706Q8b0EddSLEDGUN22EbA?e=5%3axGyu4b&fromShare=true&at=9",
        #         "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
        #         "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
        #         "description": "Hyperspectral super-resolution dataset from DeepeR paper. Contains low-resolution input spectra and high-resolution target spectra for training super-resolution models.",
        #         "license": "MIT License"
        #     }
        # ),
        "rruff_mineral_raw": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="rruff_mineral_raw",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="mineral_r"),
            metadata={
                "full_name": "RRUFF - Mineral (raw)",
                "source": "https://data.dtu.dk/api/files/36144495",
                "paper": [
                    "https://www.rruff.net/wp-content/uploads/2023/04/HMC1-30.pdf",
                    "https://doi.org/10.1039/D2AN00403H"
                ],
                "citation": [
                    "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                ],
                "description": "Mineral (raw) raman spectra subset from RRUFF database",
                "license": "See paper"
            }
        ),
        "rruff_mineral_preprocessed": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="rruff_mineral_preprocessed",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="mineral_p"),
            metadata={
                "full_name": "RRUFF - Mineral (preprocessed)",
                "source": "https://data.dtu.dk/api/files/36144495",
                "paper": [
                    "https://www.rruff.net/wp-content/uploads/2023/04/HMC1-30.pdf",
                    "https://doi.org/10.1039/D2AN00403H"
                ],
                "citation": [
                    "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                ],
                "description": "Mineral (preprocessed) raman spectra subset from RRUFF database",
                "license": "See paper"
            }
        ),
        "tlb_organic_raw": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="tlb_organic_raw",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="organic_r"),
            metadata={
                "full_name": "Transfer-learning-based Raman spectra identification - Organic (raw)",
                "source": "https://data.dtu.dk/api/files/36144495",
                "paper": [
                    "https://doi.org/10.1002/jrs.5750",
                    "https://doi.org/10.1039/D2AN00403H"
                ],
                "citation": [
                    "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                ],
                "description": "Organic (raw) dataset from Transfer-learningbased Raman spectra identification. Organic compounds measured with several excitation sources.",
                "license": "See paper"
            }
        ),
        "tlb_organic_preprocessed": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="tlb_organic_preprocessed",
            loader=lambda df: MiscLoader._load_dtu_split(df, split="organic_p"),
            metadata={
                "full_name": "Transfer-learning-based Raman spectra identification - Organic (preprocessed)",
                "source": "https://data.dtu.dk/api/files/36144495",
                "paper": [
                    "https://doi.org/10.1002/jrs.5750",
                    "https://doi.org/10.1039/D2AN00403H"
                ],
                "citation": [
                    "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                ],
                "description": "Organic (preprocessed) dataset from Transfer-learningbased Raman spectra identification. Organic compounds measured with several excitation sources.",
                "license": "See paper"
            }
        ),
        # "bacteria": DatasetInfo(
        #     task_type=TASK_TYPE.Classification,
        #     id="bacteria",
        #     loader=lambda df: MiscLoader._load_dtu_split(df, split="bacteria"),
        #     metadata={
        #         "full_name": "Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning",
        #         "source": "https://data.dtu.dk/api/files/36144495",
        #         "paper": [
        #             "https://doi.org/10.1038/s41467-019-12898-9",
        #             "http://dx.doi.org/10.1039/D2AN00403H"
        #         ],
        #         "citation": [
        #             "Ho, CS., Jean, N., Hogan, C.A. et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nat Commun 10, 4927 (2019)."
        #         ],
        #         "description": "Bacteria dataset from Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Bacterial Raman spectra described by Ho et al.",
        #         "license": "See paper"
        #     }
        # )
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
        raise NotImplementedError

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
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
        result = dataset_info.loader(dataset_cache_path)

        if result is None:
            return None

        spectra, raman_shifts, targets, class_names = result

        return RamanDataset(
            metadata=dataset_info.metadata,
            name=dataset_name,
            raman_shifts=raman_shifts,
            spectra=spectra,
            targets=targets,
            task_type=dataset_info.task_type,
            target_names=class_names,
        )

    @staticmethod
    def _load_dtu_split(cache_path: str, split: str):
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

        # Use original repo IO for mineral and organic splits
        if split in ("mineral_r", "mineral_p"):
            if split == "mineral_r":
                data = rruff.give_all_raw(split_root, print_info=True)
            else:
                csv_path = os.path.join(split_root, "excellent_unoriented_unoriented.csv")
                data = rruff.give_subset_of_spectrums(csv_path, None, "preprocess", print_info=False)
            minerals_txt = os.path.join(split_root, 'minerals.txt')
            class_names = []
            if os.path.exists(minerals_txt):
                with open(minerals_txt, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
            name_to_idx = {name.lower(): i for i, name in enumerate(class_names)} if class_names else {}
            targets = data['name'].str.lower().map(name_to_idx).to_numpy() if name_to_idx else np.zeros(len(data), dtype=int)
            spectra_path = os.path.join(split_root, 'spectra.obj')
            with open(spectra_path, "rb") as f:
                spectra_data = pickle.load(f)
            raman_shifts_list = [arr[:, 0] for arr in spectra_data]
            spectra_list = [arr[:, 1] for arr in spectra_data]
            lengths = [len(rs) for rs in raman_shifts_list]
            unique_lengths = set(lengths)
            if len(unique_lengths) == 1:
                all_equal = all(np.allclose(raman_shifts_list[0], rs) for rs in raman_shifts_list)
                if all_equal:
                    raman_shifts = raman_shifts_list[0]
                    spectra = np.stack(spectra_list)
                else:
                    raman_shifts = np.stack(raman_shifts_list).astype(np.float32)
                    spectra = np.stack(spectra_list).astype(np.float32)
            else:
                raman_shifts = raman_shifts_list
                spectra = spectra_list

            return spectra, raman_shifts, targets, class_names

        elif split in ("organic_r", "organic_p"):
            import raman_data.loaders.helper.organic as organic
            preprocess = (split == "organic_p")
            [tr_spectra, tr_label], _ = organic.get_target_data_with_randomleaveone(split_root, preprocess, random_leave_one_out=False)
            org_txt = os.path.join(split_root, 'organic.txt')
            class_names = []
            if os.path.exists(org_txt):
                with open(org_txt, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
            else:
                class_names = sorted(list(set(tr_label)))
            name_to_idx = {str(name).lower(): i for i, name in enumerate(class_names)} if class_names else {}
            targets = np.array([name_to_idx.get(str(lbl).lower(), 0) for lbl in tr_label])
            raman_shifts_list = [arr[:, 0] for arr in tr_spectra]
            spectra_list = [arr[:, 1] for arr in tr_spectra]
            lengths = [len(rs) for rs in raman_shifts_list]
            unique_lengths = set(lengths)
            if len(unique_lengths) == 1:
                all_equal = all(np.allclose(raman_shifts_list[0], rs) for rs in raman_shifts_list)
                if all_equal:
                    raman_shifts = raman_shifts_list[0]
                    spectra = np.stack(spectra_list)
                else:
                    raman_shifts = np.stack(raman_shifts_list).astype(np.float32)
                    spectra = np.stack(spectra_list).astype(np.float32)
            else:
                raman_shifts = np.empty(len(raman_shifts_list), dtype=object)
                spectra = np.empty(len(spectra_list), dtype=object)
                for i, (rs, sp) in enumerate(zip(raman_shifts_list, spectra_list)):
                    raman_shifts[i] = rs
                    spectra[i] = sp
            return spectra, raman_shifts, targets, class_names

