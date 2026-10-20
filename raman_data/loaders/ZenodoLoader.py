import glob
from collections import Counter
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

        # take the last 5 volume columns of the meta_data dataframe
        volumes = meta_data.iloc[:, -6:-1]

        # convert volumes to concentrations (fraction of total volume per sample)  and remove water
        concentrations = volumes.div(volumes.sum(axis=1), axis=0).values[:, :-1]

        # strip unit suffix from column names (e.g. "Sucrose [ul]" -> "Sucrose") and remove water
        target_names = [col.split(" [")[0] for col in volumes.columns.tolist()][:-1]

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

    @staticmethod
    def __load_19653302(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray] | None:
        """
        Parse and extract data from the locust_phase_hemolymph dataset (Zenodo ID: 19653302).

        Uses the "CvI" (Crowded vs. Isolated) file: resonance Raman spectra of desert
        locust hemolymph labelled by rearing density (phase state). The `Sample` column
        identifies which of the 200 physical animals each of the 5,000 spectra came
        from (~25 replicate spectra per animal) -- returned as an explicit group id
        rather than inferred, per the dataset's own real measurement metadata.

        Returns a tuple of (spectra, raman_shifts, targets, class_names, group_ids).
        """
        data_path = os.path.join(cache_path, "19653302", "CvI_Raw.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find CvI_Raw.csv in {data_path}")

        df = pd.read_csv(data_path)
        wn_cols = [c for c in df.columns if is_wavenumber(c)]
        if not wn_cols:
            raise FileNotFoundError(f"No wavenumber columns found in {data_path}")

        spectra = df[wn_cols].to_numpy(dtype=float)
        raman_shifts = np.array([float(c) for c in wn_cols])

        env = df["Env"].map({"C": "crowded", "I": "isolated"})
        targets, class_names = encode_labels(env)
        group_ids = pd.factorize(df["Sample"])[0]

        return spectra, raman_shifts, targets, list(class_names), group_ids

    @staticmethod
    def __load_5644790(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray] | None:
        """
        Parse and extract data from the cspp_serum_metabolites dataset (Zenodo ID: 5644790).

        Uses the "Figure-7_all-spectra-and-metadata.csv" file only (not the other four
        Figure-X CSVs in the same deposit, nor Spectra.zip): SERS spectra of human serum
        on Centrifugal Silver Plasmonic Paper (CSPP), either unspiked or spiked with one
        of two known metabolites at a single fixed concentration each (confirmed against
        the paper, https://doi.org/10.3390/bios11110467 -- Ergothioneine 25 uM,
        Hypoxanthine 50 uM; `conc` in this deposit is 1:1 confounded with `metabolite`,
        so this is a 3-class identity classification, not a concentration regression).
        The other four figures in this deposit were checked and excluded: Figure 2's
        'processing' label is an explicit null-result comparison (paper: spectra
        "practically the same" between processed/unprocessed), and Figures 4/5's numeric
        targets (`Y`, `ratio`) are intra-spectrum peak-ratio metrics computed from the
        very same spectrum row, not independent ground truth.

        `rep` identifies the physical sample replicate each spectrum was measured from
        (10 spectra per replicate) -- combined with `metabolite` into an explicit group
        id, since a given (metabolite, rep) pair is one physical prep measured 10 times.

        Returns a tuple of (spectra, raman_shifts, targets, class_names, group_ids).
        """
        data_path = os.path.join(cache_path, "5644790", "Figure-7_all-spectra-and-metadata.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Could not find Figure-7_all-spectra-and-metadata.csv in {data_path}"
            )

        df = pd.read_csv(data_path)
        wn_cols = [c for c in df.columns if is_wavenumber(c)]
        if not wn_cols:
            raise FileNotFoundError(f"No wavenumber columns found in {data_path}")

        spectra = df[wn_cols].to_numpy(dtype=float)
        raman_shifts = np.array([float(c) for c in wn_cols])

        label_map = {
            "Bkg": "serum (unspiked)",
            "Erg": "serum + ergothioneine (25 uM)",
            "Hyp": "serum + hypoxanthine (50 uM)",
        }
        labels = df["metabolite"].map(label_map)
        targets, class_names = encode_labels(labels)

        group_key = df["metabolite"].astype(str) + "_" + df["rep"].astype(str)
        group_ids = pd.factorize(group_key)[0]

        return spectra, raman_shifts, targets, list(class_names), group_ids

    # Stroke urine SERS (Zenodo ID: 19369604). Both laser variants share the
    # same data.rar; each variant reads one of the two SERSspectra folders
    # (deposit folder names: '532，0.5%，1s' and '632，5%，1s' -- the '632'
    # folder is the 633 nm line, the percent is the laser power setting).
    _STROKE_URI_RAR = "data.rar"
    _STROKE_URI_FOLDER_PREFIX = {532: "532", 633: "632"}
    # The native WiRE axes of both variants (slightly non-uniform, per file
    # ~484.9-1638.5 cm^-1) are resampled row-wise onto this common 1 cm^-1 grid.
    _STROKE_URI_GRID = np.arange(485, 1639, 1.0)
    # Patient (map file stem) -> diagnosis for the 101 participants of the
    # SERS cohort. Taken from the authors' final labelled tables in the same
    # record ('results/SERSspectra/{532,633,concat}/data.csv', `label2`/
    # `labels` columns -- identical across all three); matches the paper's
    # cohort of 55 healthy controls and 46 patients with confirmed ischemic
    # stroke.
    _STROKE_URI_PATIENT_LABELS = {
        "1": "IS", "2": "HC", "3": "HC", "4": "IS", "5": "IS", "6": "IS",
        "7": "IS", "8": "IS", "9": "IS", "10": "IS", "11": "IS", "12": "IS",
        "13": "IS", "14": "IS", "15": "IS", "16": "IS", "17": "IS", "18": "IS",
        "19": "IS", "20": "IS", "21": "IS", "22": "HC", "23": "HC", "24": "HC",
        "25": "HC", "26": "HC", "27": "HC", "28": "HC", "29": "HC", "30": "HC",
        "31": "HC", "32": "HC", "33": "IS", "34": "IS", "35": "HC", "36": "HC",
        "37": "HC", "38": "HC", "39": "HC", "40": "HC", "41": "HC", "42": "HC",
        "43": "HC", "44": "HC", "45": "HC", "46": "HC", "47": "HC", "48": "HC",
        "49": "HC", "50": "HC", "51": "HC", "52": "HC", "53": "HC", "54": "HC",
        "55": "HC", "56": "HC", "57": "IS", "58": "IS", "59": "IS", "60": "IS",
        "61": "IS", "62": "IS", "63": "HC", "64": "HC", "65": "HC", "66": "HC",
        "67": "HC", "68": "HC", "69": "HC", "70": "HC", "71": "HC", "72": "HC",
        "73": "HC", "74": "HC", "75": "HC", "76": "IS", "77": "IS", "78": "IS",
        "79": "IS", "80": "IS", "81": "IS", "82": "IS", "83": "IS", "84": "IS",
        "85": "IS", "86": "IS", "87": "IS", "88": "IS", "89": "IS", "90": "IS",
        "91": "IS", "92": "IS", "93": "IS", "94": "IS", "95": "HC", "96": "HC",
        "97": "HC", "98": "HC", "99": "HC", "100": "HC", "101": "HC",
    }

    @staticmethod
    def __load_19369604(
        cache_path: str,
        laser_nm: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray] | None:
        """
        Parse and extract data from the stroke urine SERS dataset (Zenodo ID: 19369604).

        Urine SERS spectra for ischemic-stroke vs. healthy-control
        classification, deposited by Yu et al. 2026 (Sci. Adv. 12:eaee3487).
        `laser_nm` selects one of the two excitation variants (532 or 633 nm);
        both variants are stored in the same 'data.rar' archive.

        Layout of data.rar (after extraction):
            SERSspectra/532，0.5%，1s/*.wdf   532 nm, 100 mW (0.5% power), 1 s
            SERSspectra/632，5%，1s/*.wdf     633 nm, 35 mW (5% power), 1 s
        Each .wdf is a Renishaw WiRE 10x10 map (100 spectra) of one
        participant's dried urine spot; files 1..101 are the 101 participants,
        files 31.1-34.1 are four repeated collections.

        Returns a tuple of (spectra, raman_shifts, targets, class_names,
        group_ids) with group_id = participant id (repeated collections merged
        into the parent participant).
        """
        if laser_nm not in (532, 633):
            raise ValueError(f"laser_nm must be 532 or 633, got {laser_nm}")

        # Lazy import: renishawWiRE is only needed for this one dataset; a
        # module-level import would break the whole package when the
        # community parser is missing.
        try:
            from renishawWiRE import WDFReader
        except ImportError as e:
            raise RuntimeError(
                "[!] The stroke urine SERS loader needs the community package "
                "'renishawWiRE' (Renishaw WiRE .wdf parser) to be installed: "
                "pip install renishawWiRE, then retry."
            ) from e

        dataset_dir = os.path.join(cache_path, "19369604")
        extracted_dir = os.path.join(dataset_dir, "data")

        if not os.path.isdir(extracted_dir) or not os.listdir(extracted_dir):
            rar_path = os.path.join(dataset_dir, ZenodoLoader._STROKE_URI_RAR)
            if not os.path.isfile(rar_path):
                raise FileNotFoundError(
                    f"Could not find {ZenodoLoader._STROKE_URI_RAR} in {dataset_dir}"
                )
            if LoaderTools.extract_rar_file_content(rar_path, "data") is None:
                raise RuntimeError(
                    "[!] Could not extract data.rar (Zenodo 19369604): no working "
                    "RAR extraction tool found on this system. Install bsdtar "
                    "(libarchive; ships with macOS), unrar, unar, or 7-Zip and retry."
                )

        sers_dir = os.path.join(extracted_dir, "SERSspectra")
        if not os.path.isdir(sers_dir):
            raise FileNotFoundError(f"No SERSspectra/ directory found under {extracted_dir}")

        prefix = ZenodoLoader._STROKE_URI_FOLDER_PREFIX[laser_nm]
        folders = [
            f for f in sorted(os.listdir(sers_dir))
            if f.startswith(prefix) and os.path.isdir(os.path.join(sers_dir, f))
        ]
        if len(folders) != 1:
            raise RuntimeError(
                f"[!] stroke_urine_sers_{laser_nm}: expected exactly one SERSspectra "
                f"folder starting with {prefix!r}, found: {folders}"
            )
        wdf_dir = os.path.join(sers_dir, folders[0])

        wdf_files = sorted(
            (f for f in os.listdir(wdf_dir) if f.endswith(".wdf")),
            key=lambda f: [float(t) for t in f[:-4].split(".")],
        )
        stems = [f[:-4] for f in wdf_files]
        expected_stems = {str(i) for i in range(1, 102)} | {"31.1", "32.1", "33.1", "34.1"}
        if len(stems) != 105 or set(stems) != expected_stems:
            raise RuntimeError(
                f"[!] stroke_urine_sers_{laser_nm}: expected the 105 map files "
                "1..101 plus repeat collections 31.1-34.1, got "
                f"{len(stems)} files: {stems[:8]}..."
            )

        grid = ZenodoLoader._STROKE_URI_GRID
        spectra_list, labels_list, groups_list = [], [], []
        for fname in wdf_files:
            stem = fname[:-4]
            patient = stem.split(".")[0]
            reader = WDFReader(os.path.join(wdf_dir, fname))
            x = np.asarray(reader.xdata, dtype=float)
            if x.ndim != 1 or x.shape[0] < 50:
                raise RuntimeError(
                    f"[!] stroke_urine_sers_{laser_nm}: {fname} has no usable wavenumber axis"
                )
            order = np.argsort(x)
            sp = np.asarray(reader.spectra, dtype=float).reshape(-1, x.shape[0])
            if sp.shape[0] != 100:
                raise RuntimeError(
                    f"[!] stroke_urine_sers_{laser_nm}: {fname} holds {sp.shape[0]} "
                    "spectra, expected 100 (10x10 map)"
                )
            resampled = np.empty((sp.shape[0], grid.shape[0]), dtype=float)
            for i in range(sp.shape[0]):
                resampled[i] = np.interp(grid, x[order], sp[i][order])
            spectra_list.append(resampled)
            labels_list.extend([ZenodoLoader._STROKE_URI_PATIENT_LABELS[patient]] * sp.shape[0])
            groups_list.extend([patient] * sp.shape[0])

        spectra = np.concatenate(spectra_list)
        if np.isnan(spectra).any():
            raise RuntimeError(f"[!] stroke_urine_sers_{laser_nm}: NaN values after resampling")

        targets, class_names = encode_labels(pd.Series(labels_list))
        group_ids = np.array(groups_list, dtype=str)
        return spectra, grid, targets, list(class_names), group_ids

    @staticmethod
    def __load_20042934(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray] | None:
        """
        Parse and extract data from the breast_cancer_subtypes dataset (Zenodo ID: 20042934).

        Ex vivo breast tissue Raman spectra for histological subtype
        classification (Ezzat et al. 2026, Sci. Rep.,
        doi:10.1038/s41598-026-54071-5; preprint
        doi:10.21203/rs.3.rs-7714586/v1): normal vs. IDC vs. ILC vs. DCIS.

        DATASET.xlsx / Sheet1: one row per spectrum; `histopathology` is the
        class label, `sample_id` follows SXXX_MX_RX (tissue sample /
        measurement point / repeat), then one column per wavenumber
        (500-2998 cm^-1 in 2 cm^-1 steps). The deposited wavenumber header
        repeats the final two values (2996, 2998) -- pandas/Excel renamed the
        duplicates '2996.1'/'2998.1'; only the 500..2998 step-2 columns are
        used. One sample id (S076_M4_R1) appears twice with distinct spectra
        (two measurements); both rows are kept.

        Returns a tuple of (spectra, raman_shifts, targets, class_names,
        group_ids) with group_id = tissue sample id.
        """
        data_path = os.path.join(cache_path, "20042934", "DATASET.xlsx")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find DATASET.xlsx in {data_path}")

        df = pd.read_excel(data_path, sheet_name="Sheet1")
        for col in ("histopathology", "sample_id"):
            if col not in df.columns:
                raise RuntimeError(f"[!] breast_cancer_subtypes: missing column '{col}'.")

        # Keep exactly the 500..2998 step-2 wavenumber columns (this drops the
        # duplicated '2996.1'/'2998.1' header artifact of the deposit).
        expected = {str(500 + 2 * i) for i in range(1250)}
        wn_cols = [c for c in df.columns if str(c) in expected]
        if len(wn_cols) != 1250:
            raise RuntimeError(
                f"[!] breast_cancer_subtypes: expected the 500-2998 cm^-1 step-2 "
                f"wavenumber columns, found {len(wn_cols)}."
            )
        raman_shifts = np.array([float(c) for c in wn_cols])
        if not np.all(np.diff(raman_shifts) == 2.0):
            raise RuntimeError(
                "[!] breast_cancer_subtypes: wavenumber axis is not uniform (step 2 cm^-1)."
            )

        spectra = df[wn_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError("[!] breast_cancer_subtypes: missing intensity values.")

        if len(df) != 641:
            raise RuntimeError(
                f"[!] breast_cancer_subtypes: expected 641 spectra, got {len(df)}."
            )
        labels = df["histopathology"].astype(str).str.strip()
        expected_classes = {"normal", "dcis", "idc", "ilc"}
        if set(labels) != expected_classes:
            raise RuntimeError(
                f"[!] breast_cancer_subtypes: unexpected class labels: {sorted(set(labels))}."
            )

        targets, class_names = encode_labels(labels)
        # Group by tissue sample (the SXXX prefix of sample_id SXXX_MX_RX):
        # every measurement point and repeat on one tissue sample is the same
        # physical specimen, so they must not be split across train/test.
        group_ids = (
            df["sample_id"].astype(str).str.strip().str.extract(r"^(S\d+)")[0].to_numpy()
        )
        return spectra, raman_shifts, targets, list(class_names), group_ids

    # yeast_batch_effects (Zenodo 19226627, Cheveralls et al. 2026)
    _YEAST_BATCH_ZIP = "raman-batch-effects-yeast-v1.0.zip"
    _YEAST_BATCH_REPO_PREFIX = "Arcadia-Science-raman-batch-effects-yeast-"
    # day -> (spectra subdirectory, platemap file name)
    _YEAST_BATCH_DAYS = {
        1: ("2025-08-06", "08-06_yeast-mutants.csv"),
        2: ("2025-08-07", "08-07_yeast-mutants.csv"),
        3: ("2025-08-12", "08-12_yeast-mutants.csv"),
    }

    @staticmethod
    def __load_19226627(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray] | None:
        """
        Parse and extract data from the yeast_batch_effects dataset
        (Zenodo ID: 19226627).

        Raman spectra of 9 yeast strains (3 S. cerevisiae, 6 S. pombe)
        desiccated onto three replicate plates, used to expose plate-level
        batch effects (Cheveralls et al. 2026,
        doi:10.57844/arcadia-xdmk-yq0w).

        The Zenodo file is a GitHub repo snapshot with a nested data.zip:
            august-2025-platemaps/<day>_yeast-mutants.csv  well -> strain
            august-2025-spectra/<date>/yeast_mut/<well>-Site_<n>.csv
                5x5 grid of site spectra per well (2,048 points; axis in
                the 'Pixel' column, 'Intensity' for the signal)
            august-2025-spectra/<date>/dark/Default.csv    per-plate dark
        Platemap well ids are zero-padded ('D08'); file names are not
        ('D8'). Each plate's 10th well (E12, E12, C12) is an empty-well
        control without a strain label and is excluded.

        Returns a tuple of (spectra, raman_shifts, targets, class_names,
        group_ids) with group_id = the desiccation plate (the date
        subdirectory, e.g. '2025-08-06'; 225 spectra = 9 strain wells x
        25 co-located site spectra). The plate is the paper's batch unit
        -- leave-one-plate-out is its protocol -- so the well level is
        deliberately not exposed as the group.
        """
        dataset_dir = os.path.join(cache_path, "19226627")
        # zenodo_get writes the file out under its API 'key', which carries
        # an 'Arcadia-Science/' directory prefix for this record.
        zip_path = os.path.join(dataset_dir, "Arcadia-Science", ZenodoLoader._YEAST_BATCH_ZIP)
        if not os.path.isfile(zip_path):
            zip_path = os.path.join(dataset_dir, ZenodoLoader._YEAST_BATCH_ZIP)
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(
                f"Could not find {ZenodoLoader._YEAST_BATCH_ZIP} in {dataset_dir}"
            )
        # The outer zip is extracted next to it, into a 'repo' subdirectory.
        repo_root = os.path.join(os.path.dirname(zip_path), "repo")

        data_dir = next(
            (
                d
                for d in glob.glob(
                    os.path.join(repo_root, f"{ZenodoLoader._YEAST_BATCH_REPO_PREFIX}*", "data")
                )
                if os.path.isdir(d)
            ),
            None,
        )
        if data_dir is None:
            if LoaderTools.extract_zip_file_content(zip_path, "repo") is None:
                raise RuntimeError(
                    "[!] yeast_batch_effects: could not extract the repo archive."
                )
            repo_dirs = [
                d
                for d in glob.glob(os.path.join(repo_root, f"{ZenodoLoader._YEAST_BATCH_REPO_PREFIX}*"))
                if os.path.isdir(d)
            ]
            if len(repo_dirs) != 1:
                raise RuntimeError(
                    f"[!] yeast_batch_effects: expected exactly one repo directory, found {repo_dirs}."
                )
            inner_zip = os.path.join(repo_dirs[0], "data.zip")
            if not os.path.isfile(inner_zip):
                raise FileNotFoundError(f"Could not find data.zip in {repo_dirs[0]}")
            if LoaderTools.extract_zip_file_content(inner_zip, "data") is None:
                raise RuntimeError("[!] yeast_batch_effects: could not extract data.zip.")
            data_dir = os.path.join(repo_dirs[0], "data")

        spectra_list: List[np.ndarray] = []
        raman_shifts_ref: Optional[np.ndarray] = None
        labels: List[str] = []
        group_ids: List[str] = []
        well_ids: List[str] = []

        for day, (subdir, platemap_name) in ZenodoLoader._YEAST_BATCH_DAYS.items():
            platemap = pd.read_csv(os.path.join(data_dir, "august-2025-platemaps", platemap_name))
            for col in ("well_id", "strain"):
                if col not in platemap.columns:
                    raise RuntimeError(
                        f"[!] yeast_batch_effects: missing column '{col}' in {platemap_name}."
                    )
            # Platemap well ids are zero-padded ('D08'); file names are not ('D8').
            well_to_strain = {
                f"{w[0]}{int(w[1:])}": s for w, s in zip(platemap["well_id"], platemap["strain"])
            }
            site_dir = os.path.join(data_dir, "august-2025-spectra", subdir, "yeast_mut")
            files = sorted(glob.glob(os.path.join(site_dir, "*.csv")))
            if not files:
                raise FileNotFoundError(f"[!] yeast_batch_effects: no site spectra in {site_dir}.")
            for fpath in files:
                well = os.path.basename(fpath).split("-")[0]
                strain = well_to_strain.get(well)
                if strain is None:
                    continue  # empty-well control
                df = pd.read_csv(fpath).dropna()
                axis_col = "Pixel" if "Pixel" in df.columns else "Wavenumber"
                if axis_col not in df.columns or "Intensity" not in df.columns:
                    raise RuntimeError(
                        f"[!] yeast_batch_effects: unexpected columns in {fpath}: {list(df.columns)}."
                    )
                axis = df[axis_col].to_numpy(dtype=float)
                if raman_shifts_ref is None:
                    raman_shifts_ref = axis
                elif axis.shape != raman_shifts_ref.shape or not np.allclose(axis, raman_shifts_ref):
                    raise RuntimeError(f"[!] yeast_batch_effects: spectral axis differs in {fpath}.")
                spectra_list.append(df["Intensity"].to_numpy(dtype=float))
                labels.append(strain)
                # Group = the desiccation plate (date), not the (plate, well)
                # cell: the paper's batch effect acts at plate level, and
                # every plate carries all 9 strains, so grouped k-fold with
                # the plates as groups is exactly leave-one-plate-out CV.
                # (Well positions repeat across plates, so the integrity
                # counter below keys on (plate, well), not the well alone.)
                group_ids.append(subdir)
                well_ids.append(f"{subdir}/{well}")

        if raman_shifts_ref is None or not spectra_list:
            raise RuntimeError("[!] yeast_batch_effects: no spectra parsed.")
        if raman_shifts_ref.shape != (2048,):
            raise RuntimeError(
                f"[!] yeast_batch_effects: expected 2,048 spectral points, got {raman_shifts_ref.shape[0]}."
            )

        spectra = np.asarray(spectra_list, dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError("[!] yeast_batch_effects: missing intensity values.")

        per_well = Counter(well_ids)
        per_plate = Counter(group_ids)
        per_strain = Counter(labels)
        if len(per_well) != 27 or any(v != 25 for v in per_well.values()):
            raise RuntimeError(
                f"[!] yeast_batch_effects: expected 27 (plate, well) cells with "
                f"25 site spectra each, got {len(per_well)} cells "
                f"(sizes {sorted(per_well.values())})."
            )
        if len(per_plate) != 3 or any(v != 225 for v in per_plate.values()):
            raise RuntimeError(
                f"[!] yeast_batch_effects: expected 3 plate groups with "
                f"225 spectra each, got {len(per_plate)} groups "
                f"(sizes {sorted(per_plate.values())})."
            )
        if len(per_strain) != 9 or any(v != 75 for v in per_strain.values()):
            raise RuntimeError(
                f"[!] yeast_batch_effects: expected 9 strains with 75 spectra each, "
                f"got {len(per_strain)} strains (sizes {sorted(per_strain.values())})."
            )

        targets, class_names = encode_labels(labels)
        ZenodoLoader.logger.debug(
            f"Loaded yeast_batch_effects: {spectra.shape[0]} spectra x {spectra.shape[1]} points, "
            f"{len(class_names)} strains, {len(per_well)} (plate, well) groups"
        )
        return spectra, raman_shifts_ref, targets, list(class_names), np.array(group_ids)

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
                },
                # Checked: raman_bench.splitting.infer_group_ids_from_targets finds
                # real replicate structure in both SNR subsets (repeated measurements
                # of the same mixture composition).
                is_grouped=True,
                # Checked: no missing (NaN) target values in either SNR subset.
                has_missing_labels=False,
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
            },
            # Plant-level structure: ~60 wheat lines with 53,134 spectra total (~900 per line)
            is_grouped=True,
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
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
                },
                # Checked: raman_bench.splitting.infer_group_ids_from_targets finds
                # real replicate structure in all four Colloidal/Solid x Gold/Silver
                # variants (repeated concentration-series measurements).
                is_grouped=True,
                # Checked: no missing (NaN) target values in any of the four variants.
                has_missing_labels=False,
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
                "source": "https://doi.org/10.5281/zenodo.18881751",
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
            # Individual-level structure: ~150 individuals with 2 biofluids each (blood plasma, saliva)
            is_grouped=True,
            # Source preprocessing (from source paper): penalized least-squares baseline,
            # smoothing, unit normalization
            embedded_preprocessing="penalized least-squares baseline correction, smoothing, unit normalization (source-applied)",
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        "locust_phase_hemolymph": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="19653302",
            name="Locust Phase State (Hemolymph)",
            short_name="Locust Phase State",
            file_typ="*.csv",
            license="CC BY 4.0",
            loader=lambda cache_path: ZenodoLoader.__load_19653302(cache_path),
            metadata={
                "full_name": "Desert Locust Density-Dependent Phase State (Resonance Raman of Hemolymph)",
                "source": "https://doi.org/10.5281/zenodo.19653302",
                "paper": "https://doi.org/10.5281/zenodo.19653302",
                "bibtex": (
                    "@dataset{Holman_2026, title={Datasets for: Prediction of density-dependent phase "
                    "state in the Desert Locust, Schistocerca gregaria, using resonance Raman spectroscopy "
                    "of hemolymph and machine learning}, url={https://doi.org/10.5281/zenodo.19653302}, "
                    "DOI={10.5281/zenodo.19653302}, publisher={Zenodo}, "
                    "author={Holman, Aidan P. and Mechti, Audélia and Miller, Andie C. and Techer, Maeva A. "
                    "and Goff, Nicolas K. and Song, Hojun and Behmer, Spencer T. and Kurouski, Dmitry and "
                    "Sword, Gregory A.}, year={2026}}"
                ),
                "citation": [
                    "Holman, A.P., Mechti, A., Miller, A.C., Techer, M.A., Goff, N.K., Song, H., Behmer, S.T., "
                    "Kurouski, D. and Sword, G.A., 2026. Datasets for: Prediction of density-dependent phase "
                    "state in the Desert Locust, Schistocerca gregaria, using resonance Raman spectroscopy of "
                    "hemolymph and machine learning. Zenodo."
                ],
                "description": (
                    "Resonance Raman spectra of hemolymph from desert locust (Schistocerca gregaria) "
                    "juveniles reared in crowded vs. isolated conditions -- the classic density-dependent "
                    "phase polyphenism studied in locust biology. 5,000 spectra from 200 individual animals "
                    "(~25 replicate spectra per animal). Target: rearing density/phase state, crowded vs. "
                    "isolated (binary classification). The dataset's own `Sample` column identifies which "
                    "animal each spectrum came from and is used as an explicit group id."
                ),
            },
            # Explicit group ids from the dataset's own `Sample` column (~25 replicate
            # spectra per of 200 physical animals) -- not inferred, a real measurement id.
            is_grouped=True,
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        "cspp_serum_metabolites": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="5644790",
            name="CSPP Serum Metabolites",
            short_name="CSPP Serum Metab.",
            file_typ="Figure-7_all-spectra-and-metadata.csv",
            license="CC BY 4.0",
            loader=lambda cache_path: ZenodoLoader.__load_5644790(cache_path),
            metadata={
                "full_name": "Label-free SERS on Centrifugal Silver Plasmonic Paper (CSPP): Serum Metabolite Spike-In",
                "source": "https://doi.org/10.5281/zenodo.5644790",
                "paper": "https://doi.org/10.3390/bios11110467",
                "bibtex": "@article{Esposito_2021, title={Label-free Surface Enhanced Raman Scattering (SERS) on Centrifugal Silver Plasmonic Paper (CSPP): A Novel Methodology for Unprocessed Biofluids Sampling and Analysis}, volume={11}, ISSN={2079-6374}, url={http://dx.doi.org/10.3390/bios11110467}, DOI={10.3390/bios11110467}, number={11}, journal={Biosensors}, publisher={MDPI AG}, author={Esposito, Alessandro and Bonifacio, Alois and Sergo, Valter and Fornasaro, Stefano}, year={2021}, month=nov, pages={467}}",
                "description": (
                    "SERS spectra of human serum on Centrifugal Silver Plasmonic Paper (CSPP), a paper-based "
                    "silver-nanoparticle substrate for label-free biofluid analysis, either unspiked or spiked "
                    "with a known metabolite at a single fixed concentration (ergothioneine 25 uM or "
                    "hypoxanthine 50 uM), 50 spectra per class (150 total). Target: metabolite identity -- "
                    "unspiked serum vs. ergothioneine-spiked vs. hypoxanthine-spiked (3-class classification), "
                    "matching the paper's own framing (PCA-based group separation, not absolute quantification). "
                    "Derived from the 'Figure-7_all-spectra-and-metadata.csv' file of the Zenodo deposit, one "
                    "of five figure-specific CSVs; the other four (substrate fabrication DoE, reproducibility, "
                    "shelf-life, and a serum-processing null-result comparison) were evaluated and excluded as "
                    "not providing an independent, non-derived ground-truth label suitable for benchmarking."
                ),
            },
            # Checked: `rep` identifies the physical sample replicate (10 spectra per
            # (metabolite, rep) pair) -- real replicate structure, returned as an
            # explicit group id rather than inferred.
            is_grouped=True,
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        **{
            f"stroke_urine_sers_{nm}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Medical,
                id="19369604",
                name=f"Stroke Urine SERS ({nm} nm, Yu et al. 2026)",
                short_name=f"Stroke Urine SERS {nm}",
                file_typ="data.rar",
                license="CC BY 4.0",
                loader=lambda cache_path, nm=nm: ZenodoLoader.__load_19369604(cache_path, nm),
                metadata={
                    "full_name": f"Ischemic Stroke Urine SERS, {nm} nm laser (Yu et al. 2026)",
                    "source": "https://doi.org/10.5281/zenodo.19369604",
                    "paper": "https://doi.org/10.1126/sciadv.aee3487",
                    "bibtex": (
                        "@article{Yu_2026, title={Dual-mode analysis of ischemic stroke "
                        "based on urine SERS spectra and carotid B-ultrasound}, volume={12}, "
                        "ISSN={2375-2541}, url={http://dx.doi.org/10.1126/sciadv.aee3487}, "
                        "DOI={10.1126/sciadv.aee3487}, number={27}, journal={Science Advances}, "
                        "publisher={American Association for the Advancement of Science}, "
                        "author={Yu, Wenrou and Chen, Zhaoxian and Liu, Yao and Du, Yan and "
                        "Wang, Hui and Tang, Fengjie and Xie, Xin and Huang, Yingzhou}, "
                        "year={2026}, month=Jul, pages={eaee3487}}"
                    ),
                    "citation": [
                        "Yu, W., Chen, Z., Liu, Y., Du, Y., Wang, H., Tang, F., Xie, X., and "
                        "Huang, Y. (2026). Dual-mode analysis of ischemic stroke based on urine "
                        "SERS spectra and carotid B-ultrasound. Science Advances, 12(27), eaee3487."
                    ],
                    "description": (
                        "Surface-enhanced Raman (SERS) spectra of urine from 101 participants "
                        "(55 healthy controls, 46 patients with confirmed ischemic stroke) on "
                        "colloidal Ag-nanoparticle SERS filter-paper chips (R6G enhancement "
                        f"factor 1.62e6), Renishaw inVia Qontor, 50x objective, 1 s acquisition, "
                        f"{nm} nm excitation "
                        f"({'100 mW (0.5% power setting)' if nm == 532 else '35 mW (5% power setting)'}). "
                        "Each participant's dried urine spot was scanned as a 10x10 map "
                        "(100 spectra); the deposit additionally contains four repeated "
                        "collections (files 31.1-34.1), so each variant holds 10,500 spectra "
                        "from 105 map files (the paper's headline 10,100 counts only the 101 "
                        "primary maps). Native WiRE axes (667-668 points at 532 nm, 1,015 points "
                        "at 633 nm; ~485-1,638 cm^-1; slightly non-uniform) resampled row-wise "
                        "onto a common 1 cm^-1 grid (1,154 points). group_ids = the 101 "
                        "participants (repeat collections merged into the parent participant), "
                        "so all spectra of one group share one class. Raw WiRE exports as "
                        "deposited: the authors' baseline correction, smoothing, normalization, "
                        "and outlier removal (which dropped ~8% of the map points, incl. all-zero "
                        "scans) live in their analysis code/results, not in the deposited spectra."
                    ),
                },
                # Checked: 101 participants, each with exactly one diagnosis
                # (55 HC / 46 IS); the four repeat collections (31.1-34.1) are
                # merged into their parent participant, so groups are label-pure.
                is_grouped=True,
                # Checked: all 101 patients carry a HC/IS label; no missing labels.
                has_missing_labels=False,
                # Raw WiRE map exports as deposited; the authors' preprocessing
                # (baseline correction, smoothing, normalization, outlier
                # removal) is in their analysis code (code.rar) / results
                # (results.rar), not in the deposited spectra.
                embedded_preprocessing=None,
            )
            for nm in (532, 633)
        },
        "breast_cancer_subtypes": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="20042934",
            name="Breast Cancer Subtypes (Ex Vivo Tissue)",
            short_name="Breast Cancer Subtypes",
            file_typ="DATASET.xlsx",
            license="CC BY 4.0",
            loader=lambda cache_path: ZenodoLoader.__load_20042934(cache_path),
            metadata={
                "full_name": "Ex Vivo Breast Tissue Histological Subtype Classification (Ezzat et al. 2026)",
                "source": "https://doi.org/10.5281/zenodo.20042934",
                "paper": "https://doi.org/10.1038/s41598-026-54071-5",
                "bibtex": (
                    "@article{Ezzat_2026, title={Label-free classification of breast "
                    "cancer subtypes in ex vivo human tissues using Raman spectroscopy and "
                    "machine learning}, url={http://dx.doi.org/10.1038/s41598-026-54071-5}, "
                    "DOI={10.1038/s41598-026-54071-5}, journal={Scientific Reports}, "
                    "publisher={Springer Nature}, "
                    "author={Ezzat, Ahmed and Zhu, Zhiyu and Roddan, Alfie and Silvanto, Anna "
                    "and Hou, Yingwei and Mandal, Nilanjan and Xu, Jiacheng and Zhang, Ziqi "
                    "and Zhou, Mengyi and Darzi, Ara and Dryden, Simon and Leff, Daniel R. "
                    "and Thompson, Alex J.}, year={2026}, month=jun}"
                ),
                "citation": [
                    "Ezzat, A., Zhu, Z., Roddan, A., Silvanto, A., Hou, Y., Mandal, N., Xu, J., "
                    "Zhang, Z., Zhou, M., Darzi, A., Dryden, S., Leff, D. R., and Thompson, A. J. "
                    "(2026). Label-free classification of breast cancer subtypes in ex vivo human "
                    "tissues using Raman spectroscopy and machine learning. Scientific Reports."
                ],
                "description": (
                    "Label-free Raman spectra of ex vivo human breast tissue (80 frozen tissue "
                    "samples -- 46 normal, 34 cancerous -- from consenting donors via the Imperial "
                    "College Healthcare Tissue Bank) for histological subtype classification: "
                    "normal vs. invasive ductal carcinoma (idc) vs. invasive lobular carcinoma "
                    "(ilc) vs. ductal carcinoma in situ (dcis). 641 spectra (177 normal, 221 dcis, "
                    "114 idc, 129 ilc) on a confocal Raman microscope (DXR3xi, Thermo Fisher "
                    "Scientific), 785 nm excitation, 500-2,998 cm^-1 at a 2 cm^-1 grid. Tumor "
                    "samples were measured at 3-6 spatial points, normal tissue at a single "
                    "point, with 2-3 spectra per point. group_ids are the 80 tissue sample ids "
                    "(sample_id format SXXX_MX_RX: sample / measurement point / repeat): the "
                    "paper reports the 80 samples derive from 71 individuals (9 donors "
                    "contributed two samples), but the deposit carries no sample-to-individual "
                    "mapping, so sample level is the finest available group. One sample id "
                    "(S076_M4_R1) appears twice with distinct spectra (two measurements); both "
                    "rows are kept as deposited. The authors' analysis used an 800-2,000 cm^-1 "
                    "region of interest with z-score normalization (see embedded_preprocessing); "
                    "the deposit spans the full acquired range."
                ),
            },
            # Checked: all spectra of one tissue sample share its histological
            # label (class is assigned per sample, never per spectrum). Note:
            # per the paper, 9 of the 80 samples come from the same 71
            # individuals, but that mapping is not in the deposit.
            is_grouped=True,
            # Checked: every row carries one of the four histological labels.
            has_missing_labels=False,
            # The deposit is the authors' pipeline output minus ROI selection
            # and normalization: baseline-corrected (iterative 5th-order
            # polynomial, 5% relative residual-std convergence; Savitzky-Golay
            # smoothing, window 11, order 3) but NOT z-score normalized
            # (row means span 46-2,771) and over the full 500-2,998 cm^-1
            # range (the paper's 800-2,000 cm^-1 ROI not applied). ~8% of
            # cells are negative -- baseline subtraction residuals.
            embedded_preprocessing=(
                "Baseline-corrected (iterative 5th-order polynomial; "
                "Savitzky-Golay smoothing, window 11, order 3) per the "
                "authors' pipeline; full 500-2,998 cm^-1 range, not "
                "z-score normalized (paper additionally applied an "
                "800-2,000 cm^-1 region of interest and z-score "
                "normalization in its analysis)."
            ),
        ),
        "yeast_batch_effects": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="19226627",
            name="Yeast Strain Classification with Batch Effects (Cheveralls et al. 2026)",
            short_name="Yeast Batch Effects",
            file_typ="*/raman-batch-effects-yeast-v1.0.zip",
            license="MIT",
            loader=lambda cache_path: ZenodoLoader.__load_19226627(cache_path),
            metadata={
                "full_name": (
                    "Yeast Strain Classification with Plate Batch Effects "
                    "(Cheveralls et al. 2026)"
                ),
                "source": "https://doi.org/10.5281/zenodo.19226627",
                "paper": "https://doi.org/10.57844/arcadia-xdmk-yq0w",
                "bibtex": (
                    "@article{Cheveralls_2026, title={Leave-one-batch-out "
                    "cross-validation reveals strong batch effects in Raman "
                    "spectroscopy of yeast cultures}, url={http://dx.doi.org/"
                    "10.57844/arcadia-xdmk-yq0w}, DOI={10.57844/"
                    "arcadia-xdmk-yq0w}, journal={The Stacks}, publisher="
                    "{Arcadia Science}, author={Cheveralls, Keith and Mets, "
                    "David G. and Kolb, Ilya}, year={2026}, month=mar}"
                ),
                "citation": [
                    "Cheveralls, K., Mets, D. G., and Kolb, I. (2026). "
                    "Leave-one-batch-out cross-validation reveals strong batch "
                    "effects in Raman spectroscopy of yeast cultures. The Stacks. "
                    "https://doi.org/10.57844/arcadia-xdmk-yq0w"
                ],
                "description": (
                    "Spontaneous Raman spectra of desiccated yeast cells for strain "
                    "classification under strong plate-level batch effects: 9 strains -- "
                    "3 Saccharomyces cerevisiae (BY4741 WT, YNL141/ADA1, YGL058/UBE2A) and "
                    "6 Schizosaccharomyces pombe (SP286 WT, ED666 WT, PDF1/PPT1, RAD6/UBE2A, "
                    "DEA2/ADA1, ARSG) -- grown to OD600 of 5, pelleted, washed, smeared onto "
                    "a stainless-steel plate stamped with a 96-well layout and desiccated for "
                    "~3 h. 785 nm excitation, Wasatch Photonics WP-785X-F13-R-IL fiber probe "
                    "(93 mW at the sample plane), 5 s integration with 10 averages, 5x5 "
                    "position grid per well, across 3 independent end-to-end replicate plates "
                    "(2025-08-06, 2025-08-07, 2025-08-12): 75 spectra per strain, 675 in "
                    "total (the 75 empty-well control spectra are excluded). 235.7-3,653.6 "
                    "cm^-1 at 2,048 points (non-uniform grid; the paper's analysis used a "
                    "300-1,800 cm^-1 fingerprint region). group_ids are the 3 desiccation "
                    "plates (the dates above; 225 spectra each = 9 strain wells x 25 "
                    "co-located site spectra) -- the paper's batch unit, so a grouped "
                    "3-fold split is exactly leave-one-plate-out CV, its protocol. The "
                    "finer (plate, well) structure is documented here but not exposed as "
                    "groups: well-level grouping would let spectra from the held-out "
                    "plate's other wells into training and hide the batch effect. "
                    "Spectra are raw as deposited -- no dark subtraction, baseline "
                    "correction, or normalization; the per-plate dark spectra are in the "
                    "archive (dark/Default.csv) for anyone replicating the authors' pipeline "
                    "(ramanspy: Whitaker-Hayes despiking, consensus dark subtraction, "
                    "low-intensity QC at mean < 1000, Savitzky-Golay, ModPoly order 5, AUC "
                    "normalization, per-well outlier QC; leaves 24-33 spectra per strain). "
                    "The paper shows standard 5-fold CV overestimates strain accuracy (median "
                    "MCC 0.79) while leave-one-plate-out CV exposes the batch effect (median "
                    "MCC 0.32), making this a purpose-built probe of group-aware "
                    "cross-validation."
                ),
            },
            # Checked: the 25 site spectra of one well on one plate are
            # co-located measurements of the same desiccated smear, and the
            # same strain on other plates is an independent preparation.
            # Groups are the 3 plates (the batch-effect unit of the paper),
            # not the 27 (plate, well) cells; every plate carries all 9
            # strains, so leave-one-plate-out keeps all classes in training.
            is_grouped=True,
            # Checked: every strain spectrum carries a platemap strain label;
            # the 75 empty-well control spectra (no strain) are excluded.
            has_missing_labels=False,
            # Raw spectra as deposited (Pixel/Intensity exports of the
            # Wasatch probe); all of the authors' processing lives in their
            # analysis code, not in the deposited files.
            embedded_preprocessing=None,
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
            if len(data) == 5:
                spectra, raman_shifts, concentrations, target_names, group_ids = data
            else:
                spectra, raman_shifts, concentrations, target_names = data
                group_ids = None
            return RamanDataset(
                info=ZenodoLoader.DATASETS[dataset_name],
                raman_shifts=raman_shifts,
                spectra=spectra,
                targets=concentrations,
                target_names=target_names,
                group_ids=group_ids,
            )

        return data
