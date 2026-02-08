import glob
import logging
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE


class MiscLoader(BaseLoader):
    """
    Loader for miscellaneous Raman spectroscopy datasets from various sources
    that don't fit into a dedicated loader (Dropbox, SharePoint, email-only, etc.).
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "misc")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Misc)

    DATASETS = {
        # "deepr_denoising": DatasetInfo(
        #     task_type=TASK_TYPE.Denoising,
        #     id="deepr_denoising",
        #     name="DeepeR Denoising",
        #     loader=lambda cache_path: MiscLoader._load_deepr_denoising(cache_path),
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
        #     name="DeepeR Super-Resolution Dataset",
        #     loader=lambda cache_path: MiscLoader._load_deepr_super_resolution(cache_path),
        #     metadata={
        #         "full_name": "DeepeR Super-Resolution Dataset",
        #         "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EuIIZkQGtT5NgQcYO_SOzigB706Q8b0EddSLEDGUN22EbA?e=5%3axGyu4b&fromShare=true&at=9",
        #         "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
        #         "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
        #         "description": "Hyperspectral super-resolution dataset from DeepeR paper. Contains low-resolution input spectra and high-resolution target spectra for training super-resolution models.",
        #         "license": "MIT License"
        #     }
        # ),
        "covid19_serum": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="covid19_serum",
            name="COVID-19 Human Serum",
            loader=lambda cache_path: MiscLoader._load_covid(cache_path),
            metadata={
                "full_name": "COVID-19 in human serum using Raman spectroscopy",
                "source": "Only via E-Mail",
                "description": "This study proposed the diagnosis of COVID-19 by means of Raman spectroscopy. Samples of blood serum from 10 patients positive and 10 patients negative for COVID-19 by RT-PCR RNA and ELISA tests were analyzed.",
                "paper": "https://doi.org/10.1007/s10103-021-03488-7",
                "citation": [
                    "Goulart, Ana Cristina Castro, et al. 'Diagnosing COVID-19 in human serum using Raman spectroscopy.' Lasers in Medical Science 37.4 (2022): 2217-2226."
                ],
            }
        ),
        "csho33_bacteria": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="csho33_bacteria",
            name="Pathogenic Bacteria",
            loader=lambda cache_path: MiscLoader._load_csho33_bacteria(cache_path),
            metadata={
                "full_name": "Pathogenic Bacteria Raman Dataset",
                "source": "https://github.com/csho33/bacteria-ID",
                "paper": "https://doi.org/10.1038/s41467-019-12898-9",
                "citation": [
                    "Ho, C.-S., Jean, N., Hogan, C. A., et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nat Commun 10, 4927 (2019)."
                ],
                "description": "60,000 spectra from 30 clinically relevant bacterial and yeast isolates (including an MRSA/MSSA isogenic pair). Acquired with 633 nm illumination on gold-coated silica substrates with low SNR to simulate rapid clinical acquisition times.",
            }
        ),
        **{
            f"sop_spectral_library_{process.lower().replace(' ', '_')}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.MaterialScience,
                id=f"sop_spectral_library_{process.lower().replace(' ', '_')}",
                name=f"SOP Spectral Library ({process})",
                loader=lambda cache_path, p=process.lower().replace(' ', '_'): MiscLoader._load_sop_spectral_library(cache_path,
                                                                                variant=p),
                metadata={
                    "full_name": f"Synthetic Organic Pigments Raman Spectral Library - {process}",
                    "source": "https://kikirpa-my.sharepoint.com/:u:/g/personal/wim_fremout_kikirpa_be/ES5_J9PpBatLvbTe6VlFyIoBc6fFRli0YHl2qjnLxn6I8Q?download=1",
                    "paper": "https://doi.org/10.1002/jrs.4054",
                    "citation": [
                        'Fremout, Wim, and Steven Saverwyns. "Identification of synthetic organic pigments: the role of a comprehensive digital Raman spectral library." Journal of Raman Spectroscopy 43.11 (2012): 1536-1544.'
                    ],
                    "description": f"{process} Raman spectral library comprising nearly 300 reference spectra of synthetic organic pigments (SOPs). Designed for spectral matching and identification of pigments in modern and contemporary art conservation.",
                }
            )
            for process in ["Raw", "Baseline Corrected"]
        },
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
                raise FileNotFoundError(f"Required file '{fname}' not found in cache path: {cache_path}")

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
            raise e


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
        raise NotImplementedError("Cannot download datasets from Miscellaneous loader")

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
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

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
            raise ValueError("Cache root for Misc loader is not set. Cannot load csho33_bacteria dataset.")

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

        def find_dir_with_files(root_dir: str, filenames: list[str]) -> str:
            for root, dirs, files in os.walk(root_dir):
                if all(fname in files for fname in filenames):
                    return root
            raise FileNotFoundError(f"Could not find directory containing all required files under: {root_dir}")

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
            LoaderTools.download(url=dl_url, out_dir_path=shared_root, out_file_name=zip_name)
            extracted_dir = LoaderTools.extract_zip_file_content(zip_path, unzip_target_subdir="csho33_bacteria")
            if extracted_dir is None:
                raise RuntimeError(f"[!] Failed to extract csho33_bacteria zip after download")

            data_root = find_dir_with_files(extracted_dir, required) or find_dir_with_files(shared_root, required) or find_dir_with_files(cache_root, required)

        if data_root is None:
            raise FileNotFoundError(f"[!] Could not find required data files for csho33_bacteria dataset after download and extraction. Please check the cache directory: {cache_root}")

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
            raise RuntimeError(f"[!] Failed to load data arrays for csho33_bacteria dataset: {e}")

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
    def _load_covid(cache_path):

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        dataset_cache = os.path.join(
            cache_root,
            "covid19",
        )

        if not os.path.exists(dataset_cache):
            raise Exception("Dataset not found. Please contact the authors.")


        spectra_path = os.path.join(dataset_cache, "covid19.xls")

        xls = pd.ExcelFile(spectra_path)

        # data = pd.read_excel(xls, 'RAW')
        # data = pd.read_excel(xls, 'Poly7»RAW')
        data = pd.read_excel(xls, 'N1»Poly7»RAW')

        raman_shifts = data.iloc[:, 0].values.astype(float)
        spectra = data.iloc[:, 1:].values.astype(float)

        targets = data.keys()[1:].to_list()
        encoded_targets = ["covid" in label for label in targets]
        target_names = ["control", "covid"]

        return spectra.T, raman_shifts, encoded_targets, target_names

    @staticmethod
    def _load_sop_spectral_library(cache_path: str, variant: str = "baseline_corrected"):
        """
        Load the SOP (Synthetic Organic Pigments) Spectral Library.

        The library contains ~300 Raman reference spectra of synthetic organic pigments.
        Two variants are available: 'baseline_corrected' and 'raw'.

        The data is downloaded from SharePoint as a ZIP archive containing one
        TXT file per spectrum. Each TXT file has two columns (wavenumber, intensity)
        and the pigment name is encoded in the filename.

        Args:
            cache_path: Path to the cached dataset directory.
            variant: Either 'baseline_corrected' or 'raw'.

        Returns:
            Tuple of (spectra, raman_shifts, targets, class_names) or None if loading fails.
        """
        urls = {
            "baseline_corrected": "https://kikirpa-my.sharepoint.com/:u:/g/personal/wim_fremout_kikirpa_be/ES5_J9PpBatLvbTe6VlFyIoBc6fFRli0YHl2qjnLxn6I8Q?download=1",
            "raw": "https://kikirpa-my.sharepoint.com/:u:/g/personal/wim_fremout_kikirpa_be/EdVQqPRI0l1FgNy1lemrYwwBaNcH2Jan0ZuKa04UnBqAWA?download=1",
        }

        if variant not in urls:
            raise Exception(f"Unknown variant: {variant}")

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        shared_root = os.path.join(cache_root, "sop_spectral_library")
        os.makedirs(shared_root, exist_ok=True)

        zip_name = f"sop_{variant}.zip"
        zip_path = os.path.join(shared_root, zip_name)
        extracted_dir = os.path.join(shared_root, f"sop_{variant}")

        # Check if already extracted
        if not os.path.isdir(extracted_dir) or not os.listdir(extracted_dir):
            # Try to find an existing zip if the expected one doesn't exist
            if not os.path.exists(zip_path):
                existing_zips = glob.glob(os.path.join(shared_root, f"*{variant}*.zip"))
                if existing_zips:
                    zip_path = existing_zips[0]
                    MiscLoader.logger.debug(f"Found existing SOP zip: {zip_path}")

            # Download if not present
            if not os.path.exists(zip_path):
                try:
                    LoaderTools.download(
                        url=urls[variant],
                        out_dir_path=shared_root,
                        out_file_name=zip_name,
                    )
                except Exception as e:
                    MiscLoader.logger.error(
                        f"[!] Failed to download SOP Spectral Library ({variant}): {e}\n"
                        f"    Please download manually from:\n"
                        f"    {urls[variant]}\n"
                        f"    and place the ZIP at: {zip_path}"
                    )
                    raise e

            if not os.path.exists(zip_path):
                raise Exception(f"Failed to obtain SOP zip file for variant '{variant}'. Expected at: {zip_path}")

            # Extract
            LoaderTools.extract_zip_file_content(
                zip_path,
                unzip_target_subdir=f"sop_{variant}"
            )

        if not os.path.isdir(extracted_dir):
            raise FileNotFoundError(f"Failed to extract SOP Spectral Library ({variant})")

        # Collect all TXT files recursively
        txt_files = glob.glob(os.path.join(extracted_dir, "**", "*.txt"), recursive=True)
        if not txt_files:
            raise FileNotFoundError(f"Failed to extract SOP Spectral Library ({variant})")

        spectra_list = []
        raman_shifts_list = []
        pigment_labels = []

        for txt_file in sorted(txt_files):
            try:
                # Try common delimiters for two-column data
                data = None
                for sep in ["\t", ",", ";", r"\s+"]:
                    try:
                        candidate = pd.read_csv(
                            txt_file, sep=sep, header=None, comment="#",
                            engine="python", skip_blank_lines=True
                        )
                        if candidate.shape[1] >= 2:
                            data = candidate
                            break
                    except Exception:
                        continue

                if data is None or data.shape[1] < 2:
                    MiscLoader.logger.warning(f"[!] Skipping unparseable file: {txt_file}")
                    continue

                wavenumbers = data.iloc[:, 0].values.astype(float)
                intensities = data.iloc[:, 1].values.astype(float)

                # Sort by ascending wavenumber
                sort_idx = np.argsort(wavenumbers)
                wavenumbers = wavenumbers[sort_idx]
                intensities = intensities[sort_idx]

                # Extract pigment name from filename, dropping institution
                # suffix. E.g. "PO14_A_785_kikirpa (original).txt" -> "PO14_A_785"
                pigment_name = os.path.splitext(os.path.basename(txt_file))[0]
                pigment_name = pigment_name.split("_kikirpa")[0].strip("_ ")

                spectra_list.append(intensities)
                raman_shifts_list.append(wavenumbers)
                pigment_labels.append(pigment_name)

            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to parse {txt_file}: {e}")
                continue

        if len(spectra_list) == 0:
            raise FileNotFoundError(f"Failed to parse {txt_file}")

        # Check if all spectra share the same wavenumber axis
        first_rs = raman_shifts_list[0]
        all_equal = (
            all(len(first_rs) == len(rs) for rs in raman_shifts_list)
            and all(np.allclose(first_rs, rs) for rs in raman_shifts_list)
        )

        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra = np.stack(spectra_list)
        else:
            # Interpolate all spectra to a common wavenumber grid
            raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)

        encoded_targets, class_names = encode_labels(pd.Series(pigment_labels))

        MiscLoader.logger.debug(
            f"Loaded SOP library ({variant}): {spectra.shape[0]} spectra, "
            f"{spectra.shape[1]} wavenumber points, "
            f"{len(class_names)} unique pigments"
        )

        return spectra, raman_shifts, encoded_targets, list(class_names)
