import glob
import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import spectrochempy as scp
from scipy.io import loadmat

import raman_data.loaders.helper.rruff as rruff
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.helper import organic
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, HASH_TYPE


class MiscLoader(BaseLoader):
    """
    Loader for miscellaneous Raman spectroscopy datasets.

    Currently supports datasets from the DeepeR paper (Horgan et al., 2021)
    hosted on OneDrive.
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
        "flow_synthesis": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="flow_synthesis",
            name="Flow Synthesis",
            loader=lambda cache_path: MiscLoader._load_flow_synthesis(cache_path),
            metadata={
                "full_name": "Data-driven product-process optimization of N-isopropylacrylamide microgel flow-synthesis",
                "source": "https://publications.rwth-aachen.de/record/959050/files/Raman_Spectroscopy_Measurements.zip?version=1",
                "paper": "https://doi.org/10.1016/j.cej.2023.147567",
                "citation": [
                    "Kaven, Luise F and Schweidtmann, Artur M and Keil, Jan and Israel, Jana and Wolter, Nadja and Mitsos, Alexander. Data-driven product-process optimization of N-isopropylacrylamide microgel flow-synthesis. Chemical Engineering Journal, 479, 147567, 2024, Elsevier"
                ],
                "description": "This data set contains in-line Raman spectroscopy measurements and predicted microgel sizes from Dynamic Light Scattering (DLS).The Raman spectroscopy measurements were conducted inside a customized measurement cell for monitoring in a tubular flow reactor.Inside the flow reactor, the microgel synthesis based on the monomer N-Isopropylacrylamid and the crosslinker N, N' Methylenebis(acrylamide) takes place.",
                "license": "CC BY 4.0"
            }
        ),
        **{
            f"rruff_mineral_{processed.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                id=f"rruff_mineral_{processed.lower()}",
                name=f"RRUFF Database ({processed})",
                loader=lambda cache_path, p=processed: MiscLoader._load_dtu_split(cache_path, split=f"mineral_{p.lower()}", align_output=True),
                metadata={
                    "full_name": f"RRUFF Database - {processed} Spectra",
                    "source": "https://rruff.info/",
                    "paper": "https://doi.org/10.1515/9783110417104-003",
                    "citation": [
                        "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                    ],
                    "description": "Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm).",
                    "license": "See paper"
                }
            )
            for processed in ["Raw", "Preprocess"]
        },
        "active_pharmaceutical_ingredients": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="active_pharmaceutical_ingredients",
            name="Active Pharmaceutical Ingredients",
            loader=lambda cache_path: MiscLoader._load_api(cache_path),
            metadata={
                "full_name": "Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development",
                "source": "https://springernature.figshare.com/ndownloader/articles/27931131/versions/1",
                "paper": "https://doi.org/10.1038/s41597-025-04848-6",
                "citation": [
                    "Flanagan, A.R., Glavin, F.G. Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development. Sci Data 12, 498 (2025)."
                ],
                "description": "A Raman spectral dataset comprising 3,510 spectra from 32 chemical substances. This dataset includes organic solvents and reagents commonly used in API development, along with information regarding the products in the XLSX, and code to visualise and perform technical validation on the data.",
                "license": "See paper"
            }
        ),
        **{
            f"knowitall_organics_{processed.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                id=f"knowitall_organics_{processed.lower()}",
                name=f"Organic Compounds ({processed})",
                loader=lambda cache_path, p=processed, a=(processed=="Raw"): MiscLoader._load_dtu_split(cache_path,
                                                                                  split=f"organic_{p.lower()}",
                                                                                  align_output=a),
                metadata={
                    "full_name": f"Organic Compounds Multi-Excitation Dataset - {processed}",
                    "source": "https://data.dtu.dk/api/files/36144495",
                    "paper": "https://doi.org/10.1002/jrs.5750",
                    "citation": [
                        "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                    ],
                    "description": f"{processed} Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data.",
                    "license": "See paper"
                }
            )
            for processed in ["Raw", "Preprocess"]
        },
        "covid19_serum": DatasetInfo(
            task_type=TASK_TYPE.Classification,
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
                "license": "See source"
            }
        ),
        "mind_covid": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="mind_covid",
            name="Saliva COVID-19",
            loader=lambda cache_path: MiscLoader._load_mind_dataset(cache_path, "covid_dataset", ["CTRL", "COV+", "COV-"]),
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
        **{
            f"mind_{disease.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                id=f"mind_{disease.lower()}",
                name=f"Saliva {disease}",
                loader=lambda cache_path, c=disease[0]: MiscLoader._load_mind_dataset(cache_path, "pd_ad_dataset", [f"{c}D", "CTRL"]),
                metadata={
                    "full_name": f"Saliva Neurodegenerative Disease Raman Dataset ({disease})",
                    "source": "https://github.com/dpiazza/Raman-Spectra-Data",
                    "description": f"Raman spectra from dried saliva drops targeting {disease}'s Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment.",
                    "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                    "citation": [
                        "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                    ],
                    "license": "See source"
                }
            )
            for disease in ["Parkinson", "Alzheimer"]
        },
        "csho33_bacteria": DatasetInfo(
            task_type=TASK_TYPE.Classification,
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
                "license": "See paper"
            }
        ),
        **{
            f"acid_species_{acid.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                id=f"acid_species_{acid.lower()}",
                name=f"Acid Species Concentrations ({acid})",
                loader=lambda cache_path, a=acid: MiscLoader._load_acid_species(cache_path, a),
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
            for acid in ["Succinic", "Levulinic", "Formic", "Citric", "Itaconic", "Acetic"] # TODO Oxalic: load scp gets None back
        },
        **{
            f"sop_spectral_library_{process.lower().replace(' ', '_')}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
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
                    "license": "See paper"
                }
            )
            for process in ["Raw", "Baseline Corrected"]
        },
        **{
            f"microgel_size_{pretreatment.lower()}_{spectral_range.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                id=f"microgel_size_{pretreatment.lower()}_{spectral_range.lower()}",
                name=f"Microgel Size ({pretreatment}, {spectral_range})",
                loader=lambda cache_path, _p=pretreatment, _r=spectral_range: MiscLoader._load_microgel_size(cache_path, _p, _r),
                metadata={
                    "full_name": "Nonlinear Manifold Learning Determines Microgel Size from Raman Spectroscopy",
                    "source": "https://publications.rwth-aachen.de/record/959137/files/Data_RWTH-2023-05604.zip?version=1",
                    "paper": "https://doi.org/10.1002/smll.202311920",
                    "citation": [
                        "Koronaki, E. D., Scholz, J. G. T., Modelska, M. M., Nayak, P. K., Viell, J., Mitsos, A., & Barkley, S. (2024). Nonlinear Manifold Learning Determines Microgel Size from Raman Spectroscopy. Small, 20(23), 2311920."
                    ],
                    "description": f"Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: {pretreatment}, spectral range: {spectral_range}. Task: predict particle diameter from Raman spectrum.",
                    "license": "See paper/source."
                }
            )
            for pretreatment in ["Raw", "LinearFit", "RubberBand", "MinMax_LinearFit", "MinMax_RubberBand", "SNV_LinearFit", "SNV_RubberBand"]
            for spectral_range in ["Global", "FingerPrint"]
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


        split_root = os.path.join(extracted_dir, split)
        if not os.path.isdir(split_root):
            MiscLoader.logger.error(f"[!] Expected directory not found: {split_root}")
            return None

        split_root_raw = split_root.replace("preprocess", "raw")
        if not os.path.isdir(split_root_raw):
            MiscLoader.logger.error(f"[!] Expected directory not found: {split_root_raw}")
            return None

        # Use original repo IO for mineral and organic splits
        if split in ("mineral_raw", "mineral_preprocess"):

            if split == "mineral_raw":
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

        elif split == "organic_raw":

            spectra_list, raman_shifts_list, targets = organic.get_raw_data(split_root)
            class_names = [str(i) for i in range(len(np.unique(targets)))]  # TODO

        elif split == "organic_preprocess":

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
        shared_main = os.path.join(shared_root, "Raman-Spectra-Data-main")
        if os.path.isdir(shared_main) and os.listdir(shared_main):
            MiscLoader.logger.debug(f"Using existing dataset folder at {shared_main}")
        else:
            zip_name = "Raman-Spectra-Data.zip"
            zip_file = os.path.join(shared_root, zip_name)

            if not os.path.exists(shared_root):
                MiscLoader.logger.debug(f"Attempting to download dataset {dataset_subfolder} to {shared_root}")
                os.makedirs(shared_root, exist_ok=True)

                if not os.path.exists(zip_file):
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
    def _load_acid_species(cache_path: str, subtype: str = "Succinic"):
        """
        Download and load the RWTH acid species dataset.
        Returns spectra, raman_shifts, targets, class_names (acids).
        """

        if subtype not in ["Succinic", "Levulinic", "Formic", "Citric", "Oxalic", "Itaconic", "Acetic"]:
            MiscLoader.logger.error(f"[!] Unknown RWTH acid species subtype: {subtype}")
            return None

        sub_folder = f"{subtype} acid titration"

        dataset_url = "https://publications.rwth-aachen.de/record/978266/files/Data_RWTH-2024-01177.zip?version=1"
        zip_name = "Data_RWTH-2024-01177.zip"

        cache_parent = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        shared_root = os.path.join(cache_parent, "rwth_acid_species")
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
        acid_path = os.path.join(extracted_dir, sub_folder)
        if "Succinic" in sub_folder:
            spectra_files = (glob.glob(os.path.join(acid_path, "20221019_V489", "*.spc"), recursive=True) +
                             glob.glob(os.path.join(acid_path, "20221104_V490", "*.spc"), recursive=True))
        else:
            spectra_files = glob.glob(os.path.join(acid_path, "*.spc"), recursive=True)

        if not spectra_files:
            raise Exception(f"[!] No spectra files found in {acid_path}")

        # here we have two folders for each acid subtype: 20221019_V489 and 20221104_V490
        # TODO: how to handle this?
        if subtype == "Succinic":
            concentration_file_1 = os.path.join(acid_path, "20221019_V489", f"Data Table {subtype} Acid Titration 1.xlsx")
            if not os.path.exists(concentration_file_1):
                raise Exception(f"[!] No concentration file found in {acid_path}")

            concentration_file_2 = os.path.join(acid_path, "20221104_V490", f"Data Table {subtype} Acid Titration 2.xlsx")
            if not os.path.exists(concentration_file_2):
                raise Exception(f"[!] No concentration file found in {acid_path}")

            concentration_df_1 = pd.read_excel(concentration_file_1, skiprows=28, index_col=0)
            concentration_df_1 = concentration_df_1.iloc[1:]
            concentration_names = concentration_df_1.keys()[1:3].to_list() # TODO are these two really of interest?
            concentration_df_1 = concentration_df_1[concentration_names]
            concentration_df_1.index = [file.replace("#1", "") for file in concentration_df_1.index]

            concentration_df_2 = pd.read_excel(concentration_file_2, skiprows=28, index_col=0)
            concentration_df_2 = concentration_df_2.iloc[1:]
            concentration_names = concentration_df_2.keys()[1:3].to_list() # TODO are these two really of interest?
            concentration_df_2 = concentration_df_2[concentration_names]
            concentration_df_2.index = [file.replace("#1", "") for file in concentration_df_2.index]

            concentration_df = pd.concat([concentration_df_1, concentration_df_2], axis=0)
            concentration_names = ["pH", "Mass of NaOH"] # overwrite with pretty names

        else:

            # xlsx file containing concentration values for each sample
            concentration_file = os.path.join(acid_path, f"Data Table {subtype} Acid Titration.xlsx")
            if not os.path.exists(concentration_file):
                raise Exception(f"[!] No concentration file found in {acid_path}")

            if subtype == "Levulinic":
                concentration_df = pd.read_excel(concentration_file, skiprows=28, index_col=0)
                concentration_df = concentration_df.iloc[1:]
                concentration_names = concentration_df.keys()[1:3].to_list() # TODO are these two really of interest?
                concentration_df = concentration_df[concentration_names]
                concentration_names = ["pH", "Mass of NaOH"] # overwrite with pretty names
                concentration_df.index = [file.replace("#1", "") for file in concentration_df.index]
            elif subtype == "Formic":
                concentration_df = pd.read_excel(concentration_file, skiprows=25, index_col=0)
                concentration_df = concentration_df.iloc[1:-7]
                concentration_names = concentration_df.keys()[1:5].to_list()
                concentration_df = concentration_df[concentration_names]
            elif subtype == "Citric":
                concentration_df = pd.read_excel(concentration_file, skiprows=25, index_col=0)
                concentration_df = concentration_df.iloc[1:-6]
                concentration_names = concentration_df.keys()[1:3].to_list()
                concentration_df = concentration_df[concentration_names]
            elif subtype == "Oxalic":
                concentration_df = pd.read_excel(concentration_file, skiprows=28, index_col=0)
                concentration_df = concentration_df.iloc[1:]
                concentration_names = concentration_df.keys()[1:3].to_list() # TODO are these two really of interest?
                concentration_df = concentration_df[concentration_names]
                concentration_names = ["pH", "Mass of NaOH"] # overwrite with pretty names
                concentration_df.index = [file.replace("#1", "") for file in concentration_df.index]
            elif subtype == "Itaconic":
                concentration_df = pd.read_excel(concentration_file, skiprows=25, index_col=0)
                concentration_df = concentration_df.iloc[1:-36]
                concentration_names = concentration_df.keys()[1:4].to_list()
                concentration_df = concentration_df[concentration_names]
            elif subtype == "Acetic":
                concentration_df = pd.read_excel(concentration_file, skiprows=25, index_col=0)
                concentration_df = concentration_df.iloc[1:-8]
                concentration_names = concentration_df.keys()[1:3].to_list()
                concentration_df = concentration_df[concentration_names]
                concentration_df.index = [file.replace(".mat#1", "") for file in concentration_df.index]
            else:
                raise Exception(f"Unknown subtype: {subtype}")

        # filter file containing "Water_pure" since it has different raman shifts # TODO: what to do with it? subtract?
        spectra_files = [file for file in spectra_files if "Water_pure" not in file]

        spectra_list = []
        raman_shifts_list = []
        concentrations_list = []

        for file in spectra_files:
            scp_dataset = scp.read_spc(file)
            if scp_dataset is None:
                continue

            idx = np.where([idx in file for idx in concentration_df.index])[0]
            if not len(idx):
                MiscLoader.logger.warning(f"[!] No concentration found for {file}") # TODO: what to do with it?
                continue
            elif len(idx) > 1:
                MiscLoader.logger.warning(f"[!] Multiple concentrations found for {file}: {concentration_df.iloc[idx]}") # TODO: what to do with it?
                continue

            current_concentrations = concentration_df.iloc[idx[0]].to_numpy(dtype=float).flatten()
            if current_concentrations.shape != (len(concentration_names),):
                raise Exception(f"Concentration shape mismatch for {file}")

            concentrations_list.append(current_concentrations)

            for spec in scp_dataset:
                spectra_list.append(spec.data.flatten())
                raman_shifts_list.append(np.array(spec.x.values))


        if len(spectra_list) == 0:
            MiscLoader.logger.error(f"[!] No spectra found in {extracted_dir}")
            return None

        # Align raman shifts if possible
        first_rs = raman_shifts_list[0]
        all_equal = (all(len(first_rs) == len(rs) for rs in raman_shifts_list) and
                     all(np.allclose(first_rs, rs) for rs in raman_shifts_list))
        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
        else:
            raise Exception("Ramachandran shifts are not aligned")

        spectra = np.stack(spectra_list)
        concentrations = np.stack(concentrations_list)

        return spectra, raman_shifts, concentrations, concentration_names


    @staticmethod
    def fetch_figshare_metadata(article_id: int) -> dict:
        r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}")
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _load_api(cache_path):

        metadata = MiscLoader.fetch_figshare_metadata(27931131)
        files = metadata["files"]

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            MiscLoader.logger.error("[!] Cache root for MiscLoader is not set")
            return None

        dataset_cache = os.path.join(
            cache_root,
            "active_pharmaceutical_ingredient",
        )
        os.makedirs(dataset_cache, exist_ok=True)

        for f in files:
            file_url = f["download_url"]
            file_name = f["name"]
            file_md5 = f.get("computed_md5")

            out_path = os.path.join(dataset_cache, file_name)

            # download only if missing
            if not os.path.exists(out_path):
                try:
                    LoaderTools.download(
                        url=file_url,
                        out_dir_path=dataset_cache,
                        out_file_name=file_name,
                        hash_target=file_md5,
                        hash_type=HASH_TYPE.md5,
                        referer="https://figshare.com/",
                    )
                except Exception as e:
                    MiscLoader.logger.error(
                        f"[!] Failed to download Figshare file {file_name}: {e}"
                    )
                    return None

        spectra_path = os.path.join(dataset_cache, "raman_spectra_api_compounds.csv")
        product_info_path = os.path.join(dataset_cache, "API_Product_Information.xlsx")

        spectra_df = pd.read_csv(spectra_path)
        targets = spectra_df["label"]
        raman_shifts = np.array(spectra_df.keys()[: -1].astype(float))
        spectra = spectra_df.values[:, :-1].astype(float)
        # product_info_df = pd.read_excel(product_info_path)

        encoded_targets, target_names = encode_labels(targets)
        return spectra, raman_shifts, encoded_targets, target_names


    @staticmethod
    def _load_covid(cache_path):

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            MiscLoader.logger.error("[!] Cache root for MiscLoader is not set")
            return None

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
            MiscLoader.logger.error(f"[!] Unknown SOP variant: {variant}. Use 'baseline_corrected' or 'raw'.")
            return None

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            MiscLoader.logger.error("[!] Cache root for MiscLoader is not set")
            return None

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
                    return None

            if not os.path.exists(zip_path):
                MiscLoader.logger.error(
                    f"[!] SOP ZIP file not found at: {zip_path}\n"
                    f"    Please download from: {urls[variant]}\n"
                    f"    and save to: {zip_path}"
                )
                return None

            # Extract
            LoaderTools.extract_zip_file_content(
                zip_path,
                unzip_target_subdir=f"sop_{variant}"
            )

        if not os.path.isdir(extracted_dir):
            MiscLoader.logger.error(f"[!] SOP extracted directory not found: {extracted_dir}")
            return None

        # Collect all TXT files recursively
        txt_files = glob.glob(os.path.join(extracted_dir, "**", "*.txt"), recursive=True)
        if not txt_files:
            MiscLoader.logger.error(f"[!] No TXT files found in {extracted_dir}")
            return None

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
            MiscLoader.logger.error(f"[!] No spectra could be loaded from {extracted_dir}")
            return None

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
            raman_shifts, spectra = MiscLoader.align_raman_shifts(raman_shifts_list, spectra_list)

        encoded_targets, class_names = encode_labels(pd.Series(pigment_labels))

        MiscLoader.logger.debug(
            f"Loaded SOP library ({variant}): {spectra.shape[0]} spectra, "
            f"{spectra.shape[1]} wavenumber points, "
            f"{len(class_names)} unique pigments"
        )

        return spectra, raman_shifts, encoded_targets, list(class_names)

    @staticmethod
    def _load_microgel_size(cache_path: str, pretreatment: str = "Raw", spectral_range: str = "Global"):
        """
        Download and load the RWTH microgel size dataset.

        The data contains Raman spectra of 235 microgel samples with DLS-measured
        particle diameters (208–483 nm). 14 variants are available: 7 pretreatments
        × 2 spectral ranges (Global, FingerPrint).

        Args:
            cache_path: Path to the cached dataset directory.
            pretreatment: One of Raw, LinearFit, RubberBand, MinMax_LinearFit,
                          MinMax_RubberBand, SNV_LinearFit, SNV_RubberBand.
            spectral_range: Either "Global" or "FingerPrint".

        Returns:
            Tuple of (spectra, raman_shifts, diameters, target_names) or None.
        """
        valid_pretreatments = [
            "Raw", "LinearFit", "RubberBand",
            "MinMax_LinearFit", "MinMax_RubberBand",
            "SNV_LinearFit", "SNV_RubberBand"
        ]
        valid_ranges = ["Global", "FingerPrint"]

        if pretreatment not in valid_pretreatments:
            MiscLoader.logger.error(f"[!] Unknown pretreatment: {pretreatment}")
            return None
        if spectral_range not in valid_ranges:
            MiscLoader.logger.error(f"[!] Unknown spectral range: {spectral_range}")
            return None

        dataset_url = "https://publications.rwth-aachen.de/record/959137/files/Data_RWTH-2023-05604.zip?version=1"
        zip_name = "Data_RWTH-2023-05604.zip"
        nested_zip_name = "Pretreated Raman intensity datasets with according diameter from DLS.zip"

        cache_parent = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        shared_root = os.path.join(cache_parent, "rwth_microgel_size")
        os.makedirs(shared_root, exist_ok=True)

        zip_path = os.path.join(shared_root, zip_name)

        # Download main zip if not present
        if not os.path.exists(zip_path) or not LoaderTools.is_valid_zip(zip_path):
            try:
                LoaderTools.download(url=dataset_url, out_dir_path=shared_root, out_file_name=zip_name)
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to download microgel size dataset: {e}")
                return None

        # Extract main zip if not already done
        main_extracted_dir = os.path.join(shared_root, "Data_RWTH-2023-05604")
        if not os.path.isdir(main_extracted_dir):
            try:
                LoaderTools.extract_zip_file_content(zip_path)
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to extract main zip: {e}")
                return None

        # Find and extract the nested zip containing pretreated datasets
        pretreated_dir = os.path.join(shared_root, "PretreatedDatasets")
        if not os.path.isdir(pretreated_dir) or not os.listdir(pretreated_dir):
            # Look for the nested zip in the extracted main dir
            nested_zip_path = os.path.join(main_extracted_dir, nested_zip_name)
            if not os.path.exists(nested_zip_path):
                # Search recursively for the nested zip
                import zipfile
                candidates = glob.glob(os.path.join(main_extracted_dir, "**", "*.zip"), recursive=True)
                nested_zip_path = None
                for candidate in candidates:
                    if "Pretreated" in os.path.basename(candidate):
                        nested_zip_path = candidate
                        break
                if nested_zip_path is None:
                    # Try extracting directly from the main zip
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for name in zf.namelist():
                            if "Pretreated" in name and name.endswith(".zip"):
                                nested_zip_path = os.path.join(shared_root, os.path.basename(name))
                                with open(nested_zip_path, 'wb') as f:
                                    f.write(zf.read(name))
                                break

                if nested_zip_path is None:
                    MiscLoader.logger.error(f"[!] Could not find nested zip '{nested_zip_name}' in {main_extracted_dir}")
                    return None

            try:
                LoaderTools.extract_zip_file_content(nested_zip_path, unzip_target_subdir="PretreatedDatasets")
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to extract nested pretreated zip: {e}")
                return None

        # Find the specific MAT file
        mat_filename = f"{pretreatment}_{spectral_range}.mat"
        mat_path = None

        # Search in the pretreated directory and its subdirectories
        candidates = glob.glob(os.path.join(pretreated_dir, "**", mat_filename), recursive=True)
        if candidates:
            mat_path = candidates[0]
        else:
            # Also search in the main extracted dir
            candidates = glob.glob(os.path.join(shared_root, "**", mat_filename), recursive=True)
            if candidates:
                mat_path = candidates[0]

        if mat_path is None:
            MiscLoader.logger.error(f"[!] MAT file not found: {mat_filename}")
            return None

        # Load the MAT file
        mat_data = LoaderTools.read_mat_file(mat_path)
        if mat_data is None:
            MiscLoader.logger.error(f"[!] Failed to read MAT file: {mat_path}")
            return None

        if "X_merged" not in mat_data:
            MiscLoader.logger.error(f"[!] 'X_merged' key not found in {mat_path}. Keys: {list(mat_data.keys())}")
            return None

        X = mat_data["X_merged"]

        # Parse the matrix:
        # Column 0: wavenumber axis (rows 7 to N-2)
        # Columns 1+: samples
        # Rows 0-6: metadata (part, dilution, reactor settings, monomer prediction)
        # Rows 7 to N-2: Raman spectral intensities
        # Row N-1 (last): DLS diameter (nm) - the regression target
        raman_shifts = X[7:-1, 0].astype(float)
        spectra = X[7:-1, 1:].T.astype(float)  # transpose to (n_samples, n_wavenumbers)
        diameters = X[-1, 1:].astype(float)

        MiscLoader.logger.debug(
            f"Loaded microgel size ({pretreatment}, {spectral_range}): "
            f"{spectra.shape[0]} samples, {spectra.shape[1]} wavenumber points, "
            f"diameter range: {diameters.min():.0f}–{diameters.max():.0f} nm"
        )

        return spectra, raman_shifts, diameters, ["DLS diameter (nm)"]


    @staticmethod
    def _load_flow_synthesis(cache_path: str):
        zip_path = os.path.join(cache_path, "Raman_Spectroscopy_Measurements.zip")
        if not os.path.exists(zip_path):
            # Download if not present
            try:
                LoaderTools.download(
                    url="https://publications.rwth-aachen.de/record/959050/files/Raman_Spectroscopy_Measurements.zip?version=1",
                    out_dir_path=cache_path,
                    out_file_name="Raman_Spectroscopy_Measurements.zip"
                )
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to download Flow Synthesis dataset: {e}")
                return None

        extracted_dir = os.path.join(cache_path, "extracted")
        if not os.path.isdir(extracted_dir):
            try:
                LoaderTools.extract_zip_file_content(zip_path, extracted_dir)
            except Exception as e:
                MiscLoader.logger.error(f"[!] Failed to extract Flow Synthesis zip: {e}")
                return None

        # The user will implement this part for loading data from `extracted_dir`
        pass


