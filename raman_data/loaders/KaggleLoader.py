import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from kagglehub import dataset_load, dataset_download

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels, LOG_FORMAT
from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE, APPLICATION_TYPE

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class KaggleLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on Kaggle.

    This loader provides access to datasets stored on Kaggle, handling
    download, caching, and formatting of the data into RamanDataset objects.
    Requires Kaggle API credentials to be configured.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import KaggleLoader
        >>> dataset = KaggleLoader.load_dataset("codina_diabetes_AGEs")
        >>> KaggleLoader.list_datasets()

    Note:
        Kaggle API credentials must be set up before using this loader.
        See: https://www.kaggle.com/docs/api
    """

    # Note: __load_cancer_cells was removed as mathiascharconnet/cancer-cells-sers-spectra
    # requires special Kaggle consent. The same data is available via
    # andriitrelin_cells-raman-spectra which is loaded by __load_andriitrelin above.

    DATASETS = {
        **{
            f"diabetes_skin_{position.lower().replace(' ', '_')}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Medical,
                id=f"diabetes_skin_{position.lower().replace(' ', '_')}",
                name=f"Diabetes Skin ({position})",
                short_name=f"Diab. Skin ({position})",
                license="Optica Open Access Publishing Agreement",
                loader=lambda position=position: KaggleLoader.__load_diabetes(position),
                metadata={
                    "full_name": "codina_raman-spectroscopy-of-diabetes",
                    "source": "https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes",
                    "paper": "https://doi.org/10.1364/BOE.9.004998",
                    "bibtex": "@article{Guevara_2018, title={Use of Raman spectroscopy to screen diabetes mellitus with machine learning tools}, volume={9}, ISSN={2156-7085}, url={http://dx.doi.org/10.1364/BOE.9.004998}, DOI={10.1364/BOE.9.004998}, number={10}, journal={Biomedical Optics Express}, publisher={Optica Publishing Group}, author={Guevara, Edgar and Torres-Galvan, Juan Carlos and Ramirez-Elias, Miguel G. and Luevano-Contreras, Claudia and Gonzalez, Francisco Javier}, year={2018}, month=sep, pages={4998}}",
                    "description": (
                        f"In vivo portable Raman spectra of human skin at the {position} site for DM2 screening. "
                        "11 type 2 diabetes patients (7F, 49.5±6.7 y) and 9 healthy controls (7F, 33.2±4.9 y), "
                        "University of Guanajuato, Mexico, IRB approved. PEK-785 spectrometer (Agiltron), 785 nm, "
                        "90 mW, 200 µm spot, 12 cm⁻¹ resolution, 5 scans (~15 s, ANSI Z136.1). "
                        "Preprocessed: Vancouver Raman Algorithm (VRA) fluorescence removal, cropped 800–1800 cm⁻¹, "
                        "area-normalised, zero-mean. Published ANN accuracy 88.9–90.9%, SVM 76.0–82.5% (site-dependent), "
                        "10-fold CV. HbA1c reference by boronic acid affinity chromatography (LabonaCheck MH-200). "
                        + ("AGEs subset explores correlation with advanced glycation end-product signatures "
                           "(GOLD, MG-H2, pentosidine) and precursors (3-deoxyglucosone, glyoxal, methylglyoxal). "
                           if position == "AGEs" else "")
                        + "Target: DM2 / healthy control classification."
                    )
                }
            )
            for position in ["AGEs", "Ear Lobe", "Inner Arm", "Thumbnail", "Vein"]
        },
        **{
            f"amino_acids_{substrate.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.Chemical,
                id=f"amino_acids_{substrate.lower()}",
                name=f"Amino Acid LC ({substrate})",
                short_name=f"Amino Acids ({substrate})",
                license="unknown",
                loader=lambda idx=idx: KaggleLoader.__load_sergioalejandrod(str(idx+1)),
                metadata={
                    "full_name": f"Amino Acid LC ({substrate})",
                    "source": "https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy",
                    "paper": "https://arxiv.org/abs/2011.07470",
                    "bibtex": "@misc{Rini_2020, title={An efficient label-free analyte detection algorithm for time-resolved spectroscopy}, author={Rini, Stefano and Hiramatsu, Hirotsugu}, year={2020}, eprint={2011.07470}, archivePrefix={arXiv}, primaryClass={eess.SP}}",
                    "description": (
                        f"Time-resolved LC-Raman spectra for {substrate} elution using the vertical flow method "
                        f"(Lo, Hiramatsu lab, NCTU Taiwan). {substrate} injected at "
                        + ("100 mM" if substrate in ["Glycine", "Leucine"] else "55 mM")
                        + " into an HPLC system (hydrophobic resin column, H₂O→ACE gradient, 7 mL/min, 50 µL "
                        "injection). Time-resolved acquisition at 0.2 s/frame. Preprocessing: solvent background "
                        "removal (linear regression), fluorescence removal (polynomial fitting), Savitzky-Golay "
                        "smoothing. Designed to benchmark unsupervised label-free analyte detection algorithms. "
                        "Target: elution concentration profile (regression)."
                    )
                }
            )
            for idx, substrate in enumerate(["Glycine", "Leucine", "Phenylalanine", "Tryptophan"])
        },
        **{
            f"cancer_cell_{element.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Biological,
                id=f"cancer_cell_{element.lower()}",
                name=f"Cancer Cell Metabolite ({element})",
                short_name=f"Cancer Cells ({element})",
                license="CC BY-NC-SA 4.0",
                loader=lambda element=element: KaggleLoader.__load_andriitrelin(element),
                metadata={
                    "full_name": "andriitrelin_cells-raman-spectra",
                    "source": "https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra",
                    "paper": "https://doi.org/10.1016/j.snb.2020.127660",
                    "bibtex": "@article{Erzina_2020, title={Precise cancer detection via the combination of functionalized SERS surfaces and convolutional neural network with independent inputs}, volume={308}, ISSN={0925-4005}, url={http://dx.doi.org/10.1016/j.snb.2020.127660}, DOI={10.1016/j.snb.2020.127660}, journal={Sensors and Actuators B: Chemical}, publisher={Elsevier BV}, author={Erzina, M. and Trelin, A. and Guselnikova, O. and Dvorankova, B. and Strnadova, K. and Perminova, A. and Ulbrich, P. and Mares, D. and Jerabek, V. and Elashnikov, R. and Svorcik, V. and Lyutakov, O.}, year={2020}, month=apr, pages={127660}}",
                    "description": (
                        f"SERS spectra of conditioned cell culture media metabolites entrapped on gold multibranched "
                        f"nanoparticles (AuMs, ~50 nm) functionalized with {element}. 12 sample categories from "
                        "Table 1 (Erzina et al. 2020): A2058/G361 melanoma cells, HPM melanocytes, HF skin "
                        "fibroblasts, ZAM tumour-associated fibroblasts, DMEM control — each at 0% and 10% FBS. "
                        "19,056 SERS spectra total (all three functionalization variants). ProRaman-L spectrometer "
                        "(Ondax), 785 nm, 33 mW, 30 spectra/sample, 3 s accumulation. Preprocessing: ALSS "
                        "background removal (λ=1e5, p=0.01), MinMax normalisation [0,1]. Published multi-input CNN "
                        "(Keras/TF, 400 epochs, 75:25 split) achieved 100% validation accuracy. Ethics approved by "
                        "Local Ethics Committee, General University Hospital Prague (Helsinki Declaration). "
                        f"The {element} functionalization provides selectivity toward "
                        + ("lipids and amino acids." if element == "NH2"
                           else "RNA and proteins." if element == "COOH"
                           else "lipids and amino acids.")
                    )
                }
            )
            for element in ["COOH", "NH2", "(COOH)2"]
        },
    }

    @staticmethod
    def __load_diabetes(
        dataset: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse and extract data from the diabetes Raman spectroscopy dataset.

        Args:
            dataset_id: The specific sub-dataset identifier (e.g., "AGEs", "earLobe", "innerArm").

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        file_handle = "codina/raman-spectroscopy-of-diabetes"

        key_mapping = {
            "AGEs": "AGEs",
            "Ear Lobe": "earLobe",
            "Inner Arm": "innerArm",
            "Thumbnail": "thumbNail",
            "Vein": "vein"
        }
        dataset_id = key_mapping.get(dataset, dataset)  # Use the provided dataset string directly if not in mapping

        try:
            df = dataset_load(
                adapter=KaggleDatasetAdapter.PANDAS,
                handle=file_handle,
                path=f"{dataset_id}.csv"
            )
        except Exception as e:
            raise Exception(f"Failed to load diabetes dataset '{dataset_id}': {str(e)}") from e

        if dataset_id == "AGEs":
            # AGEs dataset has a different structure
            spectra = df.loc[1:, "Var802":].to_numpy(dtype=float)
            raman_shifts = df.loc[0, "Var802":].to_numpy(dtype=float)
            targets = df.loc[1:, "AGEsID"].to_numpy()
            encoded_targets, target_names = encode_labels(targets)
        else:
            # Standard diabetes datasets
            spectra = df.loc[1:, "Var2":].to_numpy(dtype=float)
            raman_shifts = df.loc[0, "Var2":].to_numpy(dtype=float)
            targets = df.loc[1:, "has_DM2"].to_numpy().astype(int)
            target_names = ["No Diabetes", "Diabetes Type 2"]
            encoded_targets = targets

        return spectra, raman_shifts, encoded_targets, target_names


    @staticmethod
    def __load_sergioalejandrod(
        sheet_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse and extract data from the amino acids Raman spectroscopy dataset.

        Args:
            sheet_id: The sheet number identifier (1-4) corresponding to different amino acids:
                     1 = Glycine, 2 = Leucine, 3 = Phenylalanine, 4 = Tryptophan

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        file_handle = "sergioalejandrod/raman-spectroscopy"
        amino_acid_headers = {
            "1": "Gly, 40 mM",
            "2": "Leu, 40 mM",
            "3": "Phe, 40 mM",
            "4": "Trp, 40 mM"
        }

        header = amino_acid_headers[sheet_id]

        df = dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=file_handle,
            path="AminoAcids_40mM.xlsx",
            pandas_kwargs={"sheet_name": f"Sheet{sheet_id}"}
        )

        # Transpose spectra: rows become columns and vice versa
        spectra = df.loc[1:, 4.5:].to_numpy(dtype=float).T
        raman_shifts = df.loc[1:, header].to_numpy(dtype=float)
        concentrations = np.array(df.columns.values[2:], dtype=float)
        concentration_name = header[(int(sheet_id) - 1)]

        return spectra, raman_shifts, concentrations, concentration_name



    @staticmethod
    def __load_andriitrelin(
        functionalization: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse and extract data from the cells Raman spectra dataset.

        This dataset contains SERS spectra of various cell types (melanoma cells,
        melanocytes, fibroblasts) collected on gold nanourchins with different
        surface functionalizations.

        Args:
            functionalization: The surface functionalization type ("COOH", "NH2", or "(COOH)2").

        Returns:
            A tuple of (spectra, raman_shifts, labels) arrays,
            or None if parsing fails.
        """
        file_handle = "andriitrelin/cells-raman-spectra"

        # Cell type labels (folder names in the dataset)
        cell_types = [
            "A", "A-S", "G", "G-S", "HF", "HF-S",
            "ZAM", "ZAM-S", "MEL", "MEL-S", "DMEM", "DMEM-S"
        ]

        # Download the dataset first
        cache_path = dataset_download(file_handle)

        all_spectra = []
        all_labels = []
        files_loaded = 0
        files_skipped = 0

        for cell_type in cell_types:
            # Data is in the dataset_i subfolder
            file_path = os.path.join(cache_path, "dataset_i", cell_type, f"{functionalization}.csv")

            if not os.path.exists(file_path):
                logger.debug(f"Skipping missing file: {file_path}")
                files_skipped += 1
                continue

            try:
                df = pd.read_csv(file_path)
                spectra = df.values.astype(float)

                all_spectra.append(spectra)
                all_labels.extend([cell_type] * spectra.shape[0])
                files_loaded += 1

            except Exception as e:
                logger.warning(f"Could not load {file_path}: {str(e)}")
                files_skipped += 1
                continue

        if not all_spectra:
            raise Exception(f"No spectra loaded for functionalization '{functionalization}'")

        logger.info(
            f"Loaded {files_loaded} cell type files for '{functionalization}' "
            f"({files_skipped} skipped)"
        )

        # Combine all spectra
        spectra = np.vstack(all_spectra)

        # Ensure spectra are in the correct orientation (samples × features)
        # Heuristic: assume more samples than features is the correct orientation
        if spectra.shape[1] < spectra.shape[0]:
            spectra = spectra.T
            logger.debug(f"Transposed spectra from {spectra.T.shape} to {spectra.shape}")

        labels = np.array(all_labels)
        encoded_labels, label_names = encode_labels(labels)

        # Raman shifts are fixed from the README
        raman_shifts = np.linspace(100, 4278, 2090)

        return spectra, raman_shifts, encoded_labels, label_names



    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
    ) -> str:
        """
        Download a Kaggle dataset to the local cache.

        Args:
            dataset_name: The name of the dataset to download (e.g., "codina_diabetes_AGEs").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        Kaggle cache directory (~/.cache/kagglehub).

        Returns:
            str | None: The path where the dataset was downloaded, or None if the
                        dataset is not available through this loader.
        """
        if not LoaderTools.is_dataset_available(dataset_name, KaggleLoader.DATASETS):
            raise ValueError(
                f"Dataset '{dataset_name}' is not available through KaggleLoader. "
                f"Use list_datasets() to see available datasets."
            )

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Kaggle)

        logger.info(f"Downloading Kaggle dataset: {dataset_name}")

        path = dataset_download(handle=dataset_name, path=cache_path)
        logger.info(f"Dataset downloaded to: {path}")

        return path


    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
        load_data: bool = True,
    ) -> RamanDataset | None:
        """
        Load a Kaggle dataset as a RamanDataset object.

        Downloads the dataset if not already cached, then parses it into
        a standardized RamanDataset format.

        Args:
            dataset_name: The name of the dataset to load (e.g., "codina_diabetes_AGEs").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        Kaggle cache directory (~/.cache/kagglehub).

        Returns:
            RamanDataset | None: A RamanDataset object containing the spectral data,
                                 target values, and metadata, or None if loading fails.
        """
        if not LoaderTools.is_dataset_available(dataset_name, KaggleLoader.DATASETS):
            raise ValueError(
                f"Dataset '{dataset_name}' is not available through KaggleLoader. "
                f"Use list_datasets() to see available datasets."
            )

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Kaggle)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Kaggle)

        logger.info(
            f"Loading Kaggle dataset '{dataset_name}' from "
            f"{cache_path if cache_path else 'default folder (~/.cache/kagglehub)'}"
        )

        if load_data:
            data = KaggleLoader.DATASETS[dataset_name].loader()
        else:
            data = None, None, None, None

        if data is not None:
            spectra, raman_shifts, targets, target_names = data
            return RamanDataset(
                info=KaggleLoader.DATASETS[dataset_name],
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=targets,
                target_names=target_names,
            )

        return data

    @staticmethod
    def list_datasets() -> None:
        """
        Prints formatted list of datasets provided by this loader.
        """
        LoaderTools.list_datasets(KaggleLoader)
