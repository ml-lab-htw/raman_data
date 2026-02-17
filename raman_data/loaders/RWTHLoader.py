import glob
import logging
import os
from typing import Optional, Tuple, List, Callable

import numpy as np
import pandas as pd
import spectrochempy as scp

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE


class RWTHLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets from RWTH Aachen University publications.

    Datasets are hosted at publications.rwth-aachen.de and include flow synthesis,
    microgel synthesis, acid species, and microgel size datasets.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "rwth")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.RWTH)

    DATASETS = {
        "flow_microgel_synthesis": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="flow_microgel_synthesis",
            name="Microgel Synthesis in Flow",
            loader=lambda cache_path: RWTHLoader._load_flow_microgel_synthesis(cache_path),
            metadata={
                "full_name": "Data-driven product-process optimization of N-isopropylacrylamide microgel flow-synthesis",
                "source": "https://publications.rwth-aachen.de/record/959050/files/Raman_Spectroscopy_Measurements.zip?version=1",
                "paper": [
                    "https://doi.org/10.1016/j.cej.2023.147567",
                    "https://doi.org/10.18154/RWTH-2023-05551"
                ],
                "citation": [
                    "Kaven, Luise F and Schweidtmann, Artur M and Keil, Jan and Israel, Jana and Wolter, Nadja and Mitsos, Alexander. Data-driven product-process optimization of N-isopropylacrylamide microgel flow-synthesis. Chemical Engineering Journal, 479, 147567, 2024, Elsevier"
                ],
                "description": "This data set contains in-line Raman spectroscopy measurements and predicted microgel sizes from Dynamic Light Scattering (DLS).The Raman spectroscopy measurements were conducted inside a customized measurement cell for monitoring in a tubular flow reactor.Inside the flow reactor, the microgel synthesis based on the monomer N-Isopropylacrylamid and the crosslinker N, N' Methylenebis(acrylamide) takes place.",
                "license": "CC BY 4.0"
            }
        ),
        "microgel_synthesis": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="microgel_synthesis",
            name="Microgel Synthesis Flow vs. Batch",
            loader=lambda cache_path: RWTHLoader._load_microgel_synthesis(cache_path),
            metadata={
                "full_name": "In-line Monitoring of Microgel Synthesis: Flow versus Batch Reactor",
                "source": "https://publications.rwth-aachen.de/record/834113/files/Raman_spectra_and_Indirect_Hard_Models.zip?version=1",
                "paper": "https://doi.org/10.18154/RWTH-2021-09666",
                "citation": [
                    "Kaven, Luise F., et al. 'In-line monitoring of microgel synthesis: flow versus batch reactor.' Organic Process Research & Development 25.9 (2021): 2039-2051."
                ],
                "description": "This data set contains in-line Raman spectroscopy measurements inside a customized measurement cell for monitoring in a tubular flow reactor. The setup aims at monitoring the microgel synthesis in a flow reactor while aiming at a high measurement precision. The measurements include a systematic accuracy analysis, where different aspects of the flowing analyte are considered: solvent flow, flowing monomer solution, and flowing microgel solution. In addition, measurements for different calibration strategies are included. Lastly, this data set contains measurements of the microgel synthesis at varying residence times inside the tubular flow reactor.",
                "license": "CC BY 4.0"
            }
        ),
        **{
            f"acid_species_{acid.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.Chemical,
                id=f"acid_species_{acid.lower()}",
                name=f"Acid Species Concentrations ({acid})",
                loader=lambda cache_path, a=acid: RWTHLoader._load_acid_species(cache_path, a),
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
                }
            )
            for acid in ["Succinic", "Levulinic", "Formic", "Citric", "Itaconic", "Acetic"] # TODO Oxalic: load scp gets None back
        },
        **{
            f"microgel_size_{short_key}_{spectral_range.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.Chemical,
                id=f"microgel_size_{short_key}_{spectral_range.lower()}",
                name=f"Microgel Size ({label}, {spectral_range})",
                loader=lambda cache_path, _f=file_key, _r=spectral_range: RWTHLoader._load_microgel_size(cache_path, _f, _r),
                metadata={
                    "full_name": "Nonlinear Manifold Learning Determines Microgel Size from Raman Spectroscopy",
                    "source": "https://publications.rwth-aachen.de/record/959137/files/Data_RWTH-2023-05604.zip?version=1",
                    "paper": "https://doi.org/10.1002/smll.202311920",
                    "citation": [
                        "Koronaki, E. D., Scholz, J. G. T., Modelska, M. M., Nayak, P. K., Viell, J., Mitsos, A., & Barkley, S. (2024). Nonlinear Manifold Learning Determines Microgel Size from Raman Spectroscopy. Small, 20(23), 2311920."
                    ],
                    "description": f"Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: {label}, spectral range: {spectral_range}. Task: predict particle diameter from Raman spectrum.",
                }
            )
            for short_key, file_key, label in [
                ("raw",    "Raw",                "Raw"),
                ("lf",     "LinearFit",           "Linear Fit"),
                ("rb",     "RubberBand",          "Rubber Band"),
                ("mm_lf",  "MinMax_LinearFit",    "MinMax + Linear Fit"),
                ("mm_rb",  "MinMax_RubberBand",   "MinMax + Rubber Band"),
                ("snv_lf", "SNV_LinearFit",       "SNV + Linear Fit"),
                ("snv_rb", "SNV_RubberBand",      "SNV + Rubber Band"),
            ]
            for spectral_range in ["Global", "FingerPrint"]
        },
        **{ # TODO implement loading function
            f"hmf_separation_{todo.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.Chemical,
                id=f"hmf_separation_{todo.lower()}",
                name=f"HMF Separation ({todo})",
                loader=lambda cache_path, todo=todo: RWTHLoader._load_hmf_dataset(cache_path, todo),
                metadata={
                    "full_name": f"Liquid–Liquid Equilibrium of 2-MTHF/Water/5-HMF with Sulfate Electrolytes ({todo})",
                    "source": "https://doi.org/10.18154/RWTH-2024-01176",
                    "description": (
                        "Experimental liquid–liquid equilibrium dataset investigating the phase separation behavior "
                        "of 2-methyltetrahydrofuran (2-MTHF), water, and 5-hydroxymethylfurfural (5-HMF) in the presence "
                        "of sulfate salts and sulfuric acid. The dataset includes mid-infrared (MIR) spectra of organic "
                        "and aqueous phases, calibration compositions, and equilibrium phase compositions measured "
                        "between 293 K and 333 K at atmospheric pressure. Spectral data are analyzed using Indirect Hard "
                        "Modeling and support thermodynamic modeling with ePC-SAFT."
                    ),
                    "paper": "https://doi.org/10.1021/acs.jced.2c00698",
                    "citation": [
                        "Roth, D. M., Haas, M., Echtermeyer, A. W. W., Kaminski, S., Viell, J., and Jupke, A. (2023). "
                        "The Effect of Sulfate Electrolytes on the Liquid–Liquid Equilibrium of 2-MTHF/Water/5-HMF: "
                        "Experimental Study and Thermodynamic Modeling. Journal of Chemical & Engineering Data, 68(6), 1397–1410."
                    ],
                }
            )
            for todo in [""]
        },
    }
    logger = logging.getLogger(__name__)

    # Maps short dataset key fragments to the MAT filename pretreatment component
    _MICROGEL_PRETREATMENTS = {
        "Raw": "Raw",
        "LinearFit": "LinearFit",
        "RubberBand": "RubberBand",
        "MinMax_LinearFit": "MinMax_LinearFit",
        "MinMax_RubberBand": "MinMax_RubberBand",
        "SNV_LinearFit": "SNV_LinearFit",
        "SNV_RubberBand": "SNV_RubberBand",
    }

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from RWTH loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, RWTHLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.RWTH)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.RWTH)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        RWTHLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = RWTHLoader.DATASETS[dataset_name]

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
    def _download_and_extract(cache_path: str, url: str, zip_name: str, extracted_dir: str):
        zip_path = os.path.join(cache_path, zip_name)
        if not os.path.exists(zip_path):
            LoaderTools.download(url=url, out_dir_path=cache_path, out_file_name=zip_name)
        if not os.path.isdir(extracted_dir):
            LoaderTools.extract_zip_file_content(zip_path)

    @staticmethod
    def _load_spc_spectra(
            spc_dir: str,
            target_extractor: Callable[[str], str],
            dataset_label: str,
    ):
        spc_files = glob.glob(os.path.join(spc_dir, "*.spc"), recursive=True)
        if not spc_files:
            raise FileNotFoundError(f"[!] No .spc files found in {spc_dir}")

        spectra_list = []
        raman_shifts_list = []
        targets_list = []

        for spc_file in spc_files:
            try:
                dataset = scp.read_spc(spc_file)
                if dataset is None or len(dataset) == 0:
                    RWTHLoader.logger.warning(f"[!] No spectra found in {spc_file}")
                    continue

                target = target_extractor(spc_file)
                for spec in dataset:
                    spectra_list.append(spec.data.flatten())
                    raman_shifts_list.append(np.array(spec.x.values))
                    targets_list.append(target)

            except Exception as e:
                RWTHLoader.logger.warning(f"[!] Failed to read {spc_file}: {e}")
                continue

        if len(spectra_list) == 0:
            raise ValueError(f"[!] No spectra could be loaded from .spc files in {spc_dir}")

        first_rs = raman_shifts_list[0]
        all_equal = (
            all(len(first_rs) == len(rs) for rs in raman_shifts_list)
            and all(np.allclose(first_rs, rs) for rs in raman_shifts_list))
        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra = np.stack(spectra_list)
        else:
            raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)

        encoded_targets, target_names = encode_labels(targets_list)

        RWTHLoader.logger.debug(
            f"Loaded {dataset_label}: {spectra.shape[0]} spectra, "
            f"{spectra.shape[1]} wavenumber points, "
            f"{len(target_names)} unique targets"
        )
        return spectra, raman_shifts, encoded_targets, list(target_names), targets_list

    @staticmethod
    def _load_flow_microgel_synthesis(cache_path: str):
        extracted_dir = os.path.join(cache_path, "RamanSpectroscopy")
        RWTHLoader._download_and_extract(
            cache_path,
            url="https://publications.rwth-aachen.de/record/959050/files/Raman_Spectroscopy_Measurements.zip?version=1",
            zip_name="Raman_Spectroscopy_Measurements.zip",
            extracted_dir=extracted_dir,
        )

        # TODO how to get the real target? from filename? somewhere else?
        def extract_target(spc_file: str) -> str:
            return os.path.splitext(os.path.basename(spc_file))[0][18:23]

        spectra, raman_shifts, encoded_targets, target_names, targets_list = RWTHLoader._load_spc_spectra(
            extracted_dir, extract_target, "Flow Microgel Synthesis"
        )

        xlsx_files = {
            20: "Dynamic_Light_Scattering_size_predictions_at_20%C2%B0C.xlsx",
            50: "Dynamic_Light_Scattering_size_predictions_at_50%C2%B0C.xlsx"
        }

        # TODO: which one to use? or both? how to match with the spectra?
        degree = 20
        xlsx_file = xlsx_files[degree]

        xlsx_path = os.path.join(cache_path, xlsx_file)
        if not os.path.exists(xlsx_path):
            xlsx_download_path = f"https://publications.rwth-aachen.de/record/959050/files/{xlsx_file}?version=1"
            LoaderTools.download(
                url=xlsx_download_path,
                out_dir_path=cache_path,
                out_file_name=xlsx_file
            )

        xlsx_data = pd.read_excel(xlsx_path, index_col=0)

        hydraulic_radius = xlsx_data["hydrodynamic radius [nm]"]

        # TODO: how to use this?
        hydraulic_radius_values = [hydraulic_radius.get(target.replace("-", "."), np.nan) for target in targets_list]

        return spectra, raman_shifts, encoded_targets, target_names

    @staticmethod
    def _load_microgel_synthesis(cache_path: str):
        extracted_dir = os.path.join(cache_path, "Raman spectra and Indirect Hard Models")
        RWTHLoader._download_and_extract(
            cache_path,
            url="https://publications.rwth-aachen.de/record/834113/files/Raman_spectra_and_Indirect_Hard_Models.zip?version=1?version=1",
            zip_name="Raman_spectra_and_Indirect_Hard_Models.zip",
            extracted_dir=extracted_dir,
        )

        data_folder = os.path.join(extracted_dir, "Data_pub")
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"[!] Expected data folder not found: {data_folder}")

        # TODO: which subfolder to use?
        sub_folder = os.path.join(data_folder, "3_Synthesis experiments", "20190606_70°C_Exp3")

        # TODO how to get the real target? from filename? somewhere else?
        def extract_target(spc_file: str) -> str:
            return os.path.splitext(os.path.basename(spc_file))[0][19:24].replace("_", "")

        spectra, raman_shifts, encoded_targets, target_names, _ = RWTHLoader._load_spc_spectra(
            sub_folder, extract_target, "Microgel Synthesis"
        )
        return spectra, raman_shifts, encoded_targets, target_names

    @staticmethod
    def _load_acid_species(cache_path: str, subtype: str = "Succinic"):
        if subtype not in ["Succinic", "Levulinic", "Formic", "Citric", "Oxalic", "Itaconic", "Acetic"]:
            raise ValueError(f"Unknown acid subtype: {subtype}. Expected one of: Succinic, Levulinic, Formic, Citric, Oxalic, Itaconic, Acetic")

        sub_folder = f"{subtype} acid titration"

        dataset_url = "https://publications.rwth-aachen.de/record/978266/files/Data_RWTH-2024-01177.zip?version=1"
        zip_name = "Data_RWTH-2024-01177.zip"

        cache_parent = LoaderTools.get_cache_root(CACHE_DIR.RWTH)
        shared_root = os.path.join(cache_parent, "rwth_acid_species")
        os.makedirs(shared_root, exist_ok=True)
        zip_path = os.path.join(shared_root, zip_name)
        if not os.path.exists(zip_path) or not LoaderTools.is_valid_zip(zip_path):
            LoaderTools.download(url=dataset_url, out_dir_path=shared_root, out_file_name=zip_name)

        extracted_dir = os.path.join(shared_root, "Data_RWTH-2024-01177")
        if not os.path.isdir(extracted_dir):
            LoaderTools.extract_zip_file_content(zip_path)

        acid_path = os.path.join(extracted_dir, sub_folder)
        if "Succinic" in sub_folder:
            spectra_files = (glob.glob(os.path.join(acid_path, "20221019_V489", "*.spc"), recursive=True) +
                             glob.glob(os.path.join(acid_path, "20221104_V490", "*.spc"), recursive=True))
        else:
            spectra_files = glob.glob(os.path.join(acid_path, "*.spc"), recursive=True)

        if not spectra_files:
            raise Exception(f"[!] No spectra files found in {acid_path}")

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
                RWTHLoader.logger.warning(f"[!] No concentration found for {file}") # TODO: what to do with it?
                continue
            elif len(idx) > 1:
                RWTHLoader.logger.warning(f"[!] Multiple concentrations found for {file}: {concentration_df.iloc[idx]}") # TODO: what to do with it?
                continue

            current_concentrations = concentration_df.iloc[idx[0]].to_numpy(dtype=float).flatten()
            if current_concentrations.shape != (len(concentration_names),):
                raise Exception(f"Concentration shape mismatch for {file}")

            concentrations_list.append(current_concentrations)

            for spec in scp_dataset:
                spectra_list.append(spec.data.flatten())
                raman_shifts_list.append(np.array(spec.x.values))


        if len(spectra_list) == 0:
            raise Exception(f"No spectra in {file}")

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
    def _load_hmf_dataset(cache_path: str, subtype: str = "Succinic"):
        dataset_url = "https://publications.rwth-aachen.de/record/978265/files/Data_RWTH-2024-01176.zip?version=1"
        zip_name = "Data_RWTH-2024-01176.zip"

        cache_parent = LoaderTools.get_cache_root(CACHE_DIR.RWTH)
        shared_root = os.path.join(cache_parent, "rwth_hmf_separation")
        os.makedirs(shared_root, exist_ok=True)
        zip_path = os.path.join(shared_root, zip_name)
        if not os.path.exists(zip_path) or not LoaderTools.is_valid_zip(zip_path):
            LoaderTools.download(url=dataset_url, out_dir_path=shared_root, out_file_name=zip_name)

        extracted_dir = os.path.join(shared_root, "Data_RWTH-2024-01176")
        if not os.path.isdir(extracted_dir):
            LoaderTools.extract_zip_file_content(zip_path)

        raise NotImplementedError("HMF dataset loading not implemented yet")

    @staticmethod
    def _load_microgel_size(cache_path: str, pretreatment: str = "Raw", spectral_range: str = "Global"):
        if pretreatment not in RWTHLoader._MICROGEL_PRETREATMENTS:
            raise NotImplementedError(f"Unknown pretreatment {pretreatment}")
        if spectral_range not in ("Global", "FingerPrint"):
            raise NotImplementedError(f"Unknown spectral range {spectral_range}")

        dataset_url = "https://publications.rwth-aachen.de/record/959137/files/Data_RWTH-2023-05604.zip?version=1"
        zip_name = "Data_RWTH-2023-05604.zip"
        nested_zip_name = "Pretreated Raman intensity datasets with according diameter from DLS.zip"

        cache_parent = LoaderTools.get_cache_root(CACHE_DIR.RWTH)
        shared_root = os.path.join(cache_parent, "rwth_microgel_size")
        os.makedirs(shared_root, exist_ok=True)

        zip_path = os.path.join(shared_root, zip_name)

        if not os.path.exists(zip_path) or not LoaderTools.is_valid_zip(zip_path):
            LoaderTools.download(url=dataset_url, out_dir_path=shared_root, out_file_name=zip_name)

        main_extracted_dir = os.path.join(shared_root, "Data_RWTH-2023-05604")
        if not os.path.isdir(main_extracted_dir):
            LoaderTools.extract_zip_file_content(zip_path)

        pretreated_dir = os.path.join(shared_root, "PretreatedDatasets")
        if not os.path.isdir(pretreated_dir) or not os.listdir(pretreated_dir):
            nested_zip_path = os.path.join(main_extracted_dir, nested_zip_name)
            if not os.path.exists(nested_zip_path):
                import zipfile
                candidates = glob.glob(os.path.join(main_extracted_dir, "**", "*.zip"), recursive=True)
                nested_zip_path = None
                for candidate in candidates:
                    if "Pretreated" in os.path.basename(candidate):
                        nested_zip_path = candidate
                        break
                if nested_zip_path is None:
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for name in zf.namelist():
                            if "Pretreated" in name and name.endswith(".zip"):
                                nested_zip_path = os.path.join(shared_root, os.path.basename(name))
                                with open(nested_zip_path, 'wb') as f:
                                    f.write(zf.read(name))
                                break

                if nested_zip_path is None:
                    raise FileNotFoundError(f"Could not find zip file {nested_zip_path}")

            try:
                LoaderTools.extract_zip_file_content(nested_zip_path, unzip_target_subdir="PretreatedDatasets")
            except Exception as e:
                raise Exception(f"Failed to extract nested zip {nested_zip_path}: {e}")

        mat_filename = f"{pretreatment}_{spectral_range}.mat"
        mat_path = None

        candidates = glob.glob(os.path.join(pretreated_dir, "**", mat_filename), recursive=True)
        if candidates:
            mat_path = candidates[0]
        else:
            candidates = glob.glob(os.path.join(shared_root, "**", mat_filename), recursive=True)
            if candidates:
                mat_path = candidates[0]

        if mat_path is None:
            raise FileNotFoundError(f"Could not find mat file {mat_filename}")

        mat_data = LoaderTools.read_mat_file(mat_path)
        if mat_data is None:
            raise FileNotFoundError(f"Could not find mat file {mat_path}")

        if "X_merged" not in mat_data:
            raise FileNotFoundError(f"Could not find X_merged file {mat_path}")

        X = mat_data["X_merged"]

        raman_shifts = X[7:-1, 0].astype(float)
        spectra = X[7:-1, 1:].T.astype(float)
        diameters = X[-1, 1:].astype(float)

        RWTHLoader.logger.debug(
            f"Loaded microgel size ({pretreatment}, {spectral_range}): "
            f"{spectra.shape[0]} samples, {spectra.shape[1]} wavenumber points, "
            f"diameter range: {diameters.min():.0f}–{diameters.max():.0f} nm"
        )

        return spectra, raman_shifts, diameters, ["DLS diameter (nm)"]
