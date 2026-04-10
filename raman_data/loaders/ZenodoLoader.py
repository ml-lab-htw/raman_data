import glob
from typing import Optional, Tuple, List
import logging

import os, requests
import pandas as pd
import numpy as np
from zenodo_get import download

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE, APPLICATION_TYPE
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import  LoaderTools
from raman_data.loaders.utils import is_wavenumber, encode_labels, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ZenodoLoader(BaseLoader):
    """
    A static class for loading Raman spectroscopy datasets hosted on Zenodo.

    This loader provides access to datasets stored on the Zenodo research
    data repository, handling download, caching, and formatting of the data
    into RamanDataset objects.

    Attributes:
        DATASETS (dict): A dictionary mapping dataset names to their DatasetInfo objects.

    Example:
        >>> from raman_data.loaders import ZenodoLoader
        >>> dataset = ZenodoLoader.load_dataset("sugar_mixtures")
        >>> ZenodoLoader.list_datasets()
    """

    @staticmethod
    def __load_10779223(
        cache_path: str,
        snr: str = "Low SNR"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
        """
        Parse and extract data from the sugar_mixtures Raman dataset (Zenodo ID: 10779223).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """

        if snr not in ['Low SNR', 'High SNR']:
            raise ValueError(f"SNR {snr} not available in dataset sugar_mixtures")

        data_folder_parent = os.path.join(
            cache_path,
            "10779223",
            "Raw data",
            "Raw data",
            "Experimental data from sugar mixtures",
            "Raw datasets for analyses"
        )

        if not os.path.isdir(data_folder_parent) or not os.listdir(data_folder_parent):
            zip_filename = "Raw data.zip"
            try:
                extracted_dir = LoaderTools.extract_zip_file_content(
                    os.path.join(cache_path, "10779223", zip_filename),
                    zip_filename.split(".")[0]
                )
            except CorruptedZipFileError as e:
                logger.error(
                    f"There seems to be an issue with dataset '10779223/sugar_mixtures'. \n"
                )
                return None

        # load the data file
        data_folder = os.path.join(data_folder_parent, snr)

        # read spectra intensity data with pandas
        data_path = os.path.join(data_folder, "data.pkl")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find data.pkl in {data_path}")

        spectra = pd.read_pickle(data_path)

        # read raman shifts (wavenumbers) with pandas
        raman_shifts_path = os.path.join(data_folder, "spectral_axis.pkl")
        if not os.path.isfile(raman_shifts_path):
            raise FileNotFoundError(
                f"Could not find spectral_axis.pkl in {raman_shifts_path}"
            )

        raman_shifts = pd.read_pickle(raman_shifts_path)

        # read gt with pandas
        gt_path = os.path.join(data_folder, "gt_endmembers.pkl")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Could not find gt_endmembers.pkl in {gt_path}")

        meta_data_csv_path = os.path.join(data_folder, "metadata.csv")
        if not os.path.isfile(meta_data_csv_path):
            raise FileNotFoundError(f"Could not find meta_data.csv in {meta_data_csv_path}")

        meta_data = pd.read_csv(meta_data_csv_path)

        # take the last 6 columns of the meta_data dataframe
        concentrations = meta_data.iloc[:, -6:-1]

        # take their column names as target names
        target_names = concentrations.keys().to_list()

        return np.array(spectra), np.array(raman_shifts), np.array(concentrations), target_names


    @staticmethod
    def __load_256329(cache_path: str) -> np.ndarray | None:
        """
        Parse and extract data from the volumetric cells Raman dataset (Zenodo ID: 256329).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.

        Note:
            This method is not yet implemented.
        """
        raise NotImplementedError

    @staticmethod
    def __load_7644521(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
        """
        Parse and extract data from the wheat_lines Raman dataset (Zenodo ID: 7644521).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        # data field names in the mat file
        data_keys = ["COM", "COM_125mM", "ML1_125mM", "ML2_125mM"]

        # load data file
        data_path = os.path.join(cache_path, "7644521", "Data.mat")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find Data.mat in {data_path}")

        # read content
        file_content = LoaderTools.read_mat_file(data_path)
        if file_content is None:
            logger.error(
                f"There was an error while reading the dataset '7644521/wheat_lines'.\n"
            )
            return None

        # raman shifts (wavenumbers/x-axis)
        raman_shifts = file_content["Calx"].squeeze()
        spectra_list = []
        concentrations = []

        # spectra intensity data
        for idx, key in enumerate(data_keys):
            data_row = file_content[key]
            spectra_list.append(data_row)
            concentrations.append(np.repeat(idx, data_row.shape[0]))

        spectra = np.concatenate(spectra_list)
        concentrations = np.concatenate(concentrations)

        return spectra, raman_shifts, concentrations, data_keys


    @staticmethod
    def __load_3572359(
        cache_path: str, type: str, material: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
        """
        Parse and extract data from the adenine SERS dataset (Zenodo ID: 3572359).

        Args:
            cache_path: The base path where the dataset files are cached.

        Returns:
            A tuple of (spectra, raman_shifts, concentrations) arrays,
            or None if parsing fails.
        """
        # load data file
        data_path = os.path.join(cache_path, "3572359", "ILSdata.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find ILSdata.csv in {data_path}")

        substrate = type[0].lower()
        if material == "Gold":
            substrate += "Au"
        elif material == "Silver":
            substrate += "Ag"
        else:
            raise ValueError(f"Unknown material {material}")

        if substrate not in ['cAg', 'sAg', 'cAu', 'sAu']:
            raise ValueError(f"Substrate {substrate} not available in dataset adenine")

        df = pd.read_csv(data_path)
        raman_shifts = np.array(df.columns.values[9:], dtype=int)

        # substrates = df["substrate"].unique().tolist()
        substrates = [substrate]
        df_substrate = df[df["substrate"] == substrate]
        spectra = df_substrate.iloc[:, 9:].to_numpy()
        concentrations = df_substrate["conc"].to_numpy()

        return spectra, raman_shifts, concentrations, substrates


    @staticmethod
    def __load_18881751(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
        """
        Parse and extract data from the hair_dyes_sers dataset (Zenodo ID: 18881751).

        Returns a tuple of (spectra, raman_shifts, targets, class_names).
        """
        data_path = os.path.join(cache_path, "18881751", "All_Spectra_Raw.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find All_Spectra_Raw.csv in {data_path}")

        df = pd.read_csv(data_path)
        wn_cols = [c for c in df.columns if is_wavenumber(c)]
        meta_cols = [c for c in df.columns if not is_wavenumber(c)]

        spectra = df[wn_cols].to_numpy(dtype=float)
        raman_shifts = np.array([float(c) for c in wn_cols])

        brand_col = next((c for c in meta_cols if c.lower() == "brand"), None)
        label_col = brand_col if brand_col else meta_cols[0]
        targets, class_names = encode_labels(df[label_col])

        return spectra, raman_shifts, targets, list(class_names)

    @staticmethod
    def __load_11229959(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
        """
        Parse and extract data from the glioma_subtyping dataset (Zenodo ID: 11229959).

        Returns a tuple of (spectra, raman_shifts, targets, class_names).
        """
        data_path = os.path.join(cache_path, "11229959", "Spectra Data.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find 'Spectra Data.csv' in {data_path}")

        df = pd.read_csv(data_path)
        wn_cols = [c for c in df.columns if is_wavenumber(c)]
        meta_cols = [c for c in df.columns if not is_wavenumber(c)]

        if not wn_cols:
            raise FileNotFoundError(f"No wavenumber columns found in {data_path}")

        spectra = df[wn_cols].to_numpy(dtype=float)
        raman_shifts = np.array([float(c) for c in wn_cols])

        label_candidates = {"class", "label", "subtype", "grade", "diagnosis", "type", "group", "category"}
        label_col = next((c for c in meta_cols if c.lower() in label_candidates), None)
        if label_col is None and meta_cols:
            label_col = meta_cols[0]

        if label_col is None:
            raise FileNotFoundError(
                f"The Zenodo deposit for glioma_subtyping (11229959) does not include class labels. "
                f"'Spectra Data.csv' contains only spectral intensities with no label column."
            )

        targets, class_names = encode_labels(df[label_col])

        return spectra, raman_shifts, targets, list(class_names)

    @staticmethod
    def __load_7044324(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
        """
        Parse and extract data from the head_neck_cancer dataset (Zenodo ID: 7044324).

        Returns a tuple of (spectra, raman_shifts, targets, class_names).
        """
        dataset_dir = os.path.join(cache_path, "7044324")
        extracted_dir = os.path.join(dataset_dir, "Raw data")

        if not os.path.isdir(extracted_dir) or not os.listdir(extracted_dir):
            zip_path = os.path.join(dataset_dir, "Raw data.zip")
            if not os.path.isfile(zip_path):
                raise FileNotFoundError(f"Could not find 'Raw data.zip' in {dataset_dir}")
            try:
                LoaderTools.extract_zip_file_content(zip_path, "Raw data")
            except CorruptedZipFileError as e:
                logger.error(f"There seems to be an issue with dataset '7044324/head_neck_cancer'.")
                return None

        # Each .txt file is a two-column (wavenumber, intensity) file with no header.
        # Cancer spectra: Raw data/Data/Fig 3/{Plasma,Saliva}/Cancer spectra/
        # Non-cancer spectra: Raw data/Data/S3/{Plasma,Saliva}/Non-cancer spectra/
        label_dirs = [
            (os.path.join(extracted_dir, "Data", "Fig 3", "Plasma", "Cancer spectra"), "Plasma", "Cancer"),
            (os.path.join(extracted_dir, "Data", "S3", "Plasma", "Non-cancer spectra"), "Plasma", "Non-cancer"),
            (os.path.join(extracted_dir, "Data", "Fig 3", "Saliva", "Cancer spectra"), "Saliva", "Cancer"),
            (os.path.join(extracted_dir, "Data", "S3", "Saliva", "Non-cancer spectra"), "Saliva", "Non-cancer"),
        ]

        spectra_list = []
        labels_list = []
        raman_shifts_list = []

        for dir_path, tissue, diagnosis in label_dirs:
            if not os.path.isdir(dir_path):
                logger.warning(f"Expected directory not found: {dir_path}")
                continue
            label = f"{tissue} {diagnosis}"
            for fname in sorted(os.listdir(dir_path)):
                if not fname.lower().endswith(".txt"):
                    continue
                fpath = os.path.join(dir_path, fname)
                try:
                    arr = np.loadtxt(fpath, delimiter=",")
                    if arr.ndim != 2 or arr.shape[1] != 2:
                        continue
                    raman_shifts_list.append(arr[:, 0])
                    spectra_list.append(arr[:, 1])
                    labels_list.append(label)
                except Exception as e:
                    logger.warning(f"Failed to read {fpath}: {e}")

        if not spectra_list:
            raise FileNotFoundError(f"No spectral .txt files found under {extracted_dir}/Data/")

        raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)
        targets, class_names = encode_labels(pd.Series(labels_list))

        return spectra, raman_shifts, targets, list(class_names)

    __BASE_URL = "https://zenodo.org/api/records/ID/files-archive"
    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "zenodo")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Zenodo)

    DATASETS = {
        **{
            f"sugar_mixtures_{snr.lower()}_snr": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.Chemical,
                id="10779223",
                name=f"Sugar Mixtures ({snr} SNR)",
                short_name=f"Sugar Mix. ({snr} SNR)",
                file_typ="*.zip",
                license="CC BY 4.0",
                loader=lambda cache_path, snr=snr: ZenodoLoader.__load_10779223(cache_path, f"{snr} SNR"),
                metadata={
                    "full_name": f"Sugar Mixtures Raman Dataset ({snr} SNR)",
                    "source": "https://doi.org/10.5281/zenodo.10779223",
                    "paper": "https://doi.org/10.1073/pnas.2407439121",
                    "bibtex": "@article{Georgiev_2024, title={Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders}, volume={121}, ISSN={1091-6490}, url={http://dx.doi.org/10.1073/pnas.2407439121}, DOI={10.1073/pnas.2407439121}, number={45}, journal={Proceedings of the National Academy of Sciences}, publisher={Proceedings of the National Academy of Sciences}, author={Georgiev, Dimitar and Fernandez-Galiana, Alvaro and Vilms Pedersen, Simon and Papadopoulos, Georgios and Xie, Ruoxiao and Stevens, Molly M. and Barahona, Mauricio}, year={2024}, month=oct}",
                    "description": f"The {snr.lower()} signal-to-noise ratio subset of the Sugar Mixtures benchmark (7,680 measurements at 0.5 s integration). Used for evaluating the noise-robustness of hyperspectral unmixing and quantification algorithms."
                }
            )
            for snr in ["Low", "High"]
        },
        "wheat_lines": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="7644521",
            name="Mutant Wheat",
            short_name="Mutant Wheat",
            file_typ="*.mat",
            license="CC BY 4.0",
            loader=lambda cache_path: ZenodoLoader.__load_7644521(cache_path),
            metadata={
                "full_name": "Mutant Wheat Raman Dataset",
                "source": "https://doi.org/10.5281/zenodo.7644521",
                "paper": "https://doi.org/10.3389/fpls.2023.1116876",
                "bibtex": "@article{Sen_2023, title={Differentiation of advanced generation mutant wheat lines: Conventional techniques versus Raman spectroscopy}, volume={14}, ISSN={1664-462X}, url={http://dx.doi.org/10.3389/fpls.2023.1116876}, DOI={10.3389/fpls.2023.1116876}, journal={Frontiers in Plant Science}, publisher={Frontiers Media SA}, author={Sen, Ayse and Kecoglu, Ibrahim and Ahmed, Muhammad and Parlatan, Ugur and Unlu, Mehmet Burcin}, year={2023}, month=feb}",
                "description": "Raman spectra from the 7th generation of salt-stress-tolerant wheat mutant lines and their commercial cultivars. Features 785 nm excitation and tracks biochemical shifts in carotenoids and protein-related bands for agricultural phenotyping."
            }
        ),
        **{
            f"adenine_{type.lower()}_{material.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Regression,
                application_type=APPLICATION_TYPE.Chemical,
                id="3572359",
                name=f"Adenine ({type} {material})",
                short_name=f"Adenine ({type[:3]}. {material})",
                file_typ="*.csv",
                license="CC BY 4.0",
                loader=lambda cache_path, t=type, m=material: ZenodoLoader.__load_3572359(cache_path, t, m),
                metadata={
                    "full_name": f"SERS Interlaboratory Adenine Dataset ({type} {material})",
                    "source": "https://doi.org/10.5281/zenodo.3572359",
                    "paper": "https://doi.org/10.1021/acs.analchem.9b05658",
                    "bibtex": "@article{Fornasaro_2020, title={Surface Enhanced Raman Spectroscopy for Quantitative Analysis: Results of a Large-Scale European Multi-Instrument Interlaboratory Study}, volume={92}, ISSN={1520-6882}, url={http://dx.doi.org/10.1021/acs.analchem.9b05658}, DOI={10.1021/acs.analchem.9b05658}, number={5}, journal={Analytical Chemistry}, publisher={American Chemical Society (ACS)}, author={Fornasaro, Stefano and Alsamad, Fatima and Baia, Monica and Batista de Carvalho, Luis A. E. and Beleites, Claudia and Byrne, Hugh J. and Chiado, Alessandro and Chis, Mihaela and Chisanga, Malama and Daniel, Amuthachelvi and Dybas, Jakub and Eppe, Gauthier and Falgayrac, Guillaume and Faulds, Karen and Gebavi, Hrvoje and Giorgis, Fabrizio and Goodacre, Royston and Graham, Duncan and La Manna, Pietro and Laing, Stacey and Litti, Lucio and Lyng, Fiona M. and Malek, Kamilla and Malherbe, Cedric and Marques, Maria P. M. and Meneghetti, Moreno and Mitri, Elisa and Mohacek-Grosev, Vlasta and Morasso, Carlo and Muhamadali, Howbeer and Musto, Pellegrino and Novara, Chiara and Pannico, Marianna and Penel, Guillaume and Piot, Olivier and Rindzevicius, Tomas and Rusu, Elena A. and Schmidt, Michael S. and Sergo, Valter and Sockalingum, Ganesh D. and Untereiner, Valerie and Vanna, Renzo and Wiercigroch, Ewelina and Bonifacio, Alois}, year={2020}, month=feb, pages={4053--4064}}",
                    "description": f"Quantitative SERS spectra of adenine measured using {type.lower()} {material.lower()} substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability."
                }
            )
            for material in ["Gold", "Silver"]
            for type in ["Colloidal", "Solid"]
        },
        "hair_dyes_sers": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Chemical,
            id="18881751",
            name="Hair Dyes SERS",
            short_name="Hair Dyes (SERS)",
            file_typ="*.csv",
            license="CC BY 4.0",
            loader=lambda cache_path: ZenodoLoader.__load_18881751(cache_path),
            metadata={
                "full_name": "SERS Spectra of Hair Dyes Dataset",
                "source": "https://doi.org/10.5281/zen  odo.18881751",
                "paper": "https://doi.org/10.1016/j.talanta.2022.123762",
                "bibtex": "@article{Higgins_2023, title={Surface-enhanced Raman spectroscopy enables highly accurate identification of different brands, types and colors of hair dyes}, volume={251}, ISSN={0039-9140}, url={http://dx.doi.org/10.1016/j.talanta.2022.123762}, DOI={10.1016/j.talanta.2022.123762}, journal={Talanta}, publisher={Elsevier BV}, author={Higgins, Samantha and Kurouski, Dmitry}, year={2023}, month=jan, pages={123762}}",
                "description": (
                    "SERS spectra of commercial hair dye products acquired with a portable Raman spectrometer. "
                    "Each spectrum is labelled by brand, permanence (permanent/semi-permanent/temporary), and colour. "
                    "Target: brand identity (classification)."
                ),
            },
        ),
        # "glioma_subtyping": DatasetInfo(
        #     task_type=TASK_TYPE.Classification,
        #     application_type=APPLICATION_TYPE.Medical,
        #     id="11229959",
        #     name="Glioma Subtyping",
        #     file_typ="*.csv",
        #     loader=lambda cache_path: ZenodoLoader.__load_11229959(cache_path),
        #     metadata={
        #         "full_name": "Label-free Raman Spectroscopy Glioma Subtyping Dataset",
        #         "source": "https://doi.org/10.5281/zenodo.11229959",
        #         "description": (
        #             "Raman spectra of glioma tissue samples for molecular subtyping without fluorescent dyes. "
        #             "Targets correspond to WHO-grade molecular subgroups (e.g. IDH-wildtype vs IDH-mutant). "
        #             "Task: multiclass classification of glioma subtypes."
        #         ),
        #     },
        # ),
        "head_neck_cancer": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="7044324",
            name="Head & Neck Cancer",
            short_name="Head & Neck Cancer",
            file_typ="*.zip",
            license="CC BY 4.0",
            loader=lambda cache_path: ZenodoLoader.__load_7044324(cache_path),
            metadata={
                "full_name": "Head and Neck Cancer Raman Spectroscopy Dataset",
                "source": "https://doi.org/10.5281/zenodo.7044324",
                "paper": "https://doi.org/10.1038/s41598-022-22197-x",
                "bibtex": "@article{koster2022fused, title={Fused Raman spectroscopic analysis of blood and saliva delivers high accuracy for head and neck cancer diagnostics}, author={Koster, Hanna J and Guillen-Perez, Antonio and Gomez-Diaz, Juan Sebastian and Navas-Moreno, Maria and Birkeland, Andrew C and Carney, Randy P}, journal={Scientific Reports}, volume={12}, number={1}, pages={18464}, year={2022}, publisher={Nature Publishing Group UK London}}",
                "description": (
                    "Raman spectra of blood plasma and saliva samples from head and neck cancer patients and "
                    "healthy controls. Acquired for non-invasive liquid biopsy screening. "
                    "Target: cancer vs. control (binary classification)."
                ),
            },
        ),
    }

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
    ) -> str | None:
        """
        Download a Zenodo dataset to the local cache.

        Args:
            dataset_name: The name of the dataset to download (e.g., "sugar_mixtures").
            cache_path: Custom directory to save the dataset. If None, uses the default
                        Zenodo cache directory (~/.cache/zenodo).

        Returns:
            str | None: The path where the dataset was downloaded, or None if the
                        dataset is not available or download fails.

        Raises:
            requests.HTTPError: If the HTTP request to Zenodo fails.
        """
        cache_path = ZenodoLoader.set_cache_dir(cache_path, dataset_name)
        dataset_id = ZenodoLoader.DATASETS[dataset_name].id
        file_typ = ZenodoLoader.DATASETS[dataset_name].file_typ
        dataset_cache_path = os.path.join(cache_path, dataset_id)
        os.makedirs(dataset_cache_path, exist_ok=True)

        try:
            dataset_id = ZenodoLoader.DATASETS[dataset_name].id
            download(
                record_or_doi=dataset_id, 
                output_dir=dataset_cache_path, 
                file_glob=file_typ,
                # file_glob="*.csv",
            )

        except requests.HTTPError as e:
            logger.error(f"Could not download requested dataset")
            return None
        except OSError as e:
            logger.error(f"Failed to save dataset due to filesystem error: {e}")
            return None

        return cache_path

    @staticmethod
    def set_cache_dir(cache_path: str | None, dataset_name: str) -> str | None:
        if not LoaderTools.is_dataset_available(dataset_name, ZenodoLoader.DATASETS):
            logger.error(f"[!] Cannot download {dataset_name} dataset with ZenodoLoader")
            raise Exception(f"Dataset {dataset_name} not available")

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        return cache_path

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> RamanDataset | None:
        """
        Load a Zenodo dataset as a RamanDataset object.

        Downloads the dataset if not already cached, then parses it into
        a standardized RamanDataset format. Automatically retries download
        up to 3 times if the file appears corrupted.

        Args:
            dataset_name: The name of the dataset to load (e.g., "sugar_mixtures").
            cache_path: Custom directory to load/save the dataset. If None, uses the default
                        Zenodo cache directory (~/.cache/zenodo).
            load_data: If True, loads the actual spectral data. If False, returns metadata only.

        Returns:
            RamanDataset | None: A RamanDataset object containing the spectral data,
                                 target values, and metadata, or None if loading fails.

        Raises:
            Exception: If the file download fails after maximum retry attempts.
        """
        cache_path = ZenodoLoader.set_cache_dir(cache_path, dataset_name)
        dataset_id = ZenodoLoader.DATASETS[dataset_name].id

        if load_data:
            dataset_cache_path = os.path.join(cache_path, dataset_id)

            if not os.path.isdir(dataset_cache_path) or not glob.glob(os.path.join(dataset_cache_path, "*")):
                ZenodoLoader.download_dataset(dataset_name, cache_path)

            data = ZenodoLoader.DATASETS[dataset_name].loader(cache_path)
        else:
            data = None, None, None, None

        if data is not None:
            spectra, raman_shifts, concentrations, target_names = data
            return RamanDataset(
                info=ZenodoLoader.DATASETS[dataset_name],
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=concentrations,
                target_names=target_names,
            )

        return data
