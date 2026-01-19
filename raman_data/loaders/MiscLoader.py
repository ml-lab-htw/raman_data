from typing import Optional, Tuple
import os
import numpy as np
import logging

from scipy.io import loadmat

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR
from raman_data.loaders.LoaderTools import LoaderTools


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
        "deepr_denoising": DatasetInfo(
            task_type=TASK_TYPE.Denoising,
            id="deepr_denoising",
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
            loader=lambda df: MiscLoader._load_deepr_super_resolution(df),
            metadata={
                "full_name": "DeepeR Super-Resolution Dataset",
                "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EuIIZkQGtT5NgQcYO_SOzigB706Q8b0EddSLEDGUN22EbA?e=5%3axGyu4b&fromShare=true&at=9",
                "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
                "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
                "description": "Hyperspectral super-resolution dataset from DeepeR paper. Contains low-resolution input spectra and high-resolution target spectra for training super-resolution models.",
                "license": "MIT License"
            }
        )
    }

    logger = logging.getLogger(__name__)

    @staticmethod
    def _load_deepr_denoising(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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

            return inputs, axis, outputs

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

        # Try to open if not present
        if not os.path.exists(dataset_cache_path) or not os.listdir(dataset_cache_path):
            os.makedirs(dataset_cache_path, exist_ok=True)
            MiscLoader.logger.warning(f"[!] Dataset not found in cache. Automatic donload is currently not supported for OneDrive datasets.")
            MiscLoader.logger.warning(f"[!] Please download the dataset manually from the provided link.")
            MiscLoader.logger.warning(f"[!] {dataset_name} dataset is available at {MiscLoader.DATASETS[dataset_name].metadata.get('source', 'No link provided')}")
            MiscLoader.logger.warning(f"[!] Please download the dataset folder manually from the provided link and extract it to: {dataset_cache_path}")
            return None

        MiscLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        # Get dataset info and load data
        dataset_info = MiscLoader.DATASETS[dataset_name]
        result = dataset_info.loader(dataset_cache_path)

        if result is None:
            return None

        spectra, raman_shifts, targets = result

        return RamanDataset(
            metadata=dataset_info.metadata,
            name=dataset_name,
            raman_shifts=raman_shifts,
            spectra=spectra,
            targets=targets,
            task_type=dataset_info.task_type
        )
