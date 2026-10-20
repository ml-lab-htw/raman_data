import logging
import os
import re
from collections import Counter
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE


_MENDELEY_DATASET_ID = "y4md8znppn"
_MENDELEY_VERSION = 1

_CITATION = (
    "Rizzo, S., Weesepoel, Y., Erasmus, S., Sinkeldam, J., Piccinelli, A. L., van Ruth, S. (2023). "
    "Dataset of Raman and Surface-enhanced Raman Spectroscopy spectra of illicit adulterants added to dietary "
    "supplements. Wageningen University & Research. https://doi.org/10.17632/y4md8znppn/1"
)


class MendeleyLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on Mendeley Data.

    Downloads dataset files via the Mendeley Data public API and caches them locally.

    File formats in this dataset:
    - ``*.0``      — plain-text two-column CSV (wavenumber, intensity), FT-Raman spectra.
    - ``*_MISA.spc`` — Galactic SPC binary format, SERS spectra from the portable MISA analyzer.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "mendeley")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Mendeley)

    DATASETS = {
        "illicit_adulterants_ft_raman": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="illicit_adulterants_ft_raman",
            name="Illicit Adulterants in Dietary Supplements (FT-Raman)",
            short_name="Illicit Adult. (FT)",
            license="CC BY 4.0",
            loader=lambda cache_path: MendeleyLoader._load_adulterants(cache_path, instrument="ft_raman"),
            metadata={
                "full_name": "Dataset of Raman and SERS spectra of illicit adulterants added to dietary supplements",
                "source": "https://data.mendeley.com/datasets/y4md8znppn/1",
                "doi": "10.17632/y4md8znppn/1",
                "paper": "https://doi.org/10.1016/j.heliyon.2023.e18509",
                "bibtex": "@article{Rizzo_2023, title={A multi-analyte screening method for the rapid detection of illicit adulterants in dietary supplements using a portable SERS analyzer}, volume={9}, ISSN={2405-8440}, url={http://dx.doi.org/10.1016/j.heliyon.2023.e18509}, DOI={10.1016/j.heliyon.2023.e18509}, number={8}, journal={Heliyon}, publisher={Elsevier BV}, author={Rizzo, Serena and Weesepoel, Yannick and Erasmus, Sara and Sinkeldam, Joost and Piccinelli, Anna Lisa and van Ruth, Saskia}, year={2023}, month=aug, pages={e18509}}",
                "citation": _CITATION,
                "description": (
                    "FT-Raman reference spectra (1064 nm, 50–3600 cm⁻¹, 4 cm⁻¹ resolution, 32 scans) of 11 "
                    "SERS-active pharmaceutically active adulterants found in dietary supplements, acquired with a "
                    "benchtop Bruker RAM II FT-IR Raman module (OPUS v7.2, 300 mW). Compounds selected from RASFF "
                    "portal EU alerts. Library spectra at 1 mg/mL in ethanol; three replicates per compound. "
                    "Companion to the portable SERS dataset (illicit_adulterants_sers). "
                    "Target: compound identity (classification)."
                ),
            },
        ),
        "illicit_adulterants_sers": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="illicit_adulterants_sers",
            name="Illicit Adulterants in Dietary Supplements (SERS)",
            short_name="Illicit Adult. (SERS)",
            license="CC BY 4.0",
            loader=lambda cache_path: MendeleyLoader._load_adulterants(cache_path, instrument="sers"),
            metadata={
                "full_name": "Dataset of Raman and SERS spectra of illicit adulterants added to dietary supplements",
                "source": "https://data.mendeley.com/datasets/y4md8znppn/1",
                "doi": "10.17632/y4md8znppn/1",
                "paper": "https://doi.org/10.1016/j.heliyon.2023.e18509",
                "bibtex": "@article{Rizzo_2023, title={A multi-analyte screening method for the rapid detection of illicit adulterants in dietary supplements using a portable SERS analyzer}, volume={9}, ISSN={2405-8440}, url={http://dx.doi.org/10.1016/j.heliyon.2023.e18509}, DOI={10.1016/j.heliyon.2023.e18509}, number={8}, journal={Heliyon}, publisher={Elsevier BV}, author={Rizzo, Serena and Weesepoel, Yannick and Erasmus, Sara and Sinkeldam, Joost and Piccinelli, Anna Lisa and van Ruth, Saskia}, year={2023}, month=aug, pages={e18509}}",
                "citation": _CITATION,
                "description": (
                    "Portable SERS spectra (785 nm, 400–2300 cm⁻¹, 8 cm⁻¹ FWHM, 3 scans) of 11 SERS-active "
                    "illicit adulterants in dietary supplements. Acquired with a Metrohm Instant SERS analyzer "
                    "(MISA) using silver printed-SERS (p-SERS) substrates and Orbital Raster Scan (ORS™) technology "
                    "(30 µm spot, ~2 mm raster, 100 mW, ASTM 1840 calibration). Sample prep: 10 mg/100 µL EtOH, "
                    "centrifuge 806,400 RCF, 10 µL on strip, dry 15 min, triplicates. Enhancement factors 10³–10⁴. "
                    "Adulteration LOI 0.1–5.0% w/w in 18 commercial dietary supplements. "
                    "Developed with the Dutch Food and Consumer Product Safety Authority. "
                    "Target: compound identity (classification)."
                ),
            },
        ),
        "microplastics_weathered": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="microplastics_weathered",
            name="Weathered Microplastics",
            short_name="Weathered Microplastics",
            license="CC BY 4.0",
            loader=lambda cache_path: MendeleyLoader._load_microplastics(cache_path),
            metadata={
                "full_name": "Weathered Microplastics Dataset",
                "source": "https://data.mendeley.com/datasets/kpygrf9fg6",
                "doi": "10.17632/kpygrf9fg6",
                "paper": "https://doi.org/10.1016/j.scitotenv.2020.139990",
                "bibtex": (
                    "@article{DONG2020139990,\n"
                    "  title = {Raman spectra and surface changes of microplastics weathered under natural environments},\n"
                    "  journal = {Science of The Total Environment},\n"
                    "  volume = {739},\n"
                    "  pages = {139990},\n"
                    "  year = {2020},\n"
                    "  issn = {0048-9697},\n"
                    "  doi = {https://doi.org/10.1016/j.scitotenv.2020.139990},\n"
                    "  url = {https://www.sciencedirect.com/science/article/pii/S0048969720335105},\n"
                    "  author = {Mingtan Dong and Qiaoqiao Zhang and Xinli Xing and Wei Chen and Zhenbing She and Zejiao Luo},\n"
                    "  keywords = {Microplastics, Raman spectra, Weathering, Surface changes, Fourier transform infrared spectrometry},\n"
                    "}"
                ),
                "description": (
                    "Raman spectra of field-collected weathered microplastic debris (155 samples) and unweathered "
                    "standard plastics (18 samples), acquired from sediments near waste plastics recycling industries "
                    "in Laizhou City, Shandong Province, China (August 2018). WITec alpha300-R confocal Raman "
                    "(532 nm, 5 mW typical, 200–3500 cm⁻¹, 600 l/mm grating, 4 cm⁻¹ resolution, 5 s integration, "
                    "4 accumulations typical). 124 spectra with distinct peaks selected for the Raman Database of "
                    "Weathered Plastics (RDWP); polymer types: PE (62), PP (44), PET (10), PVC (4), PA (2), plus "
                    "minor classes and 20 unidentified. Weathered spectra show weakened CH₂ peaks and broad "
                    "2100–2200 cm⁻¹ bands absent from standard spectra. Prefixes: 'sta-' = unweathered standards, "
                    "'wea-' = field-weathered debris. Target: polymer type (classification)."
                ),
            },
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        "water_quality_ions": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="cxf3zxsc3k",
            name="Water Quality Ion Quantification (Nitrate/Nitrite/Sulphate)",
            short_name="Water Quality Ions",
            license="CC BY 4.0",
            loader=lambda cache_path: MendeleyLoader._load_water_quality_ions(cache_path),
            metadata={
                "full_name": (
                    "A Dataset of Raman Spectra with Quantitative Labels of Mixtures of "
                    "Nitrate, Nitrite and Sulphate (Luciani et al. 2026)"
                ),
                "source": "https://data.mendeley.com/datasets/cxf3zxsc3k/3",
                "doi": "10.17632/cxf3zxsc3k.3",
                "paper": "https://doi.org/10.1016/j.dib.2026.113319",
                "bibtex": (
                    "@article{Luciani_2026, title={A Dataset of Raman Spectra with Quantitative "
                    "Labels of Mixtures of Nitrate, Nitrite and Sulphate}, volume={113319}, "
                    "url={http://dx.doi.org/10.1016/j.dib.2026.113319}, DOI={10.1016/j.dib.2026.113319}, "
                    "journal={Data in Brief}, publisher={Elsevier}, "
                    "author={Luciani, Lorenzo and Nocera, Antonio and Raimondi, Michela and "
                    "Sbrollini, Agnese and Morettini, Micaela and Galassi, Rossana}, year={2026}}"
                ),
                "citation": [
                    "Luciani, L., Nocera, A., Raimondi, M., Sbrollini, A., Morettini, M., and Galassi, R. (2026). "
                    "A Dataset of Raman Spectra with Quantitative Labels of Nitrate, Nitrite and Sulphate for "
                    "Application of Artificial Intelligence to Water Quality Sensing. Mendeley Data, v3. "
                    "https://doi.org/10.17632/cxf3zxsc3k.3"
                ],
                "description": (
                    "Raw Raman spectra of aqueous solutions containing nitrate, nitrite and sulphate ions, "
                    "for multi-analyte concentration quantification (mg/L). 694 spectra (IDs 0-693) covering "
                    "single-analyte (289), binary (215) and ternary (190) mixtures, prepared in ultrapure "
                    "Milli-Q water from KNO3, NaNO2 and Na2SO4 (counterions K+/Na+). 401-3,899 cm^-1 at "
                    "1,019 points (non-uniform ~2.7-4.3 cm^-1 grid) on a Horiba iHR320 micro-Raman system "
                    "(Olympus BXFM microscope, 50x objective, Horiba Syncerity CCD) under 532 nm excitation. "
                    "Targets: three columns of nominal concentrations in mg/L — nitrate 122-30,667, "
                    "nitrite 250-15,333, sulphate 253-16,907 (0 = analyte absent). group_ids identify the "
                    "70 prepared solutions (one unique concentration combination each), each measured up to "
                    "10 times, so replicate spectra must not be split across train/test. Lab-prepared "
                    "standards in ultrapure water, not field matrices."
                ),
            },
            # Checked: each unique (nitrate, nitrite, sulphate) combination
            # was prepared once and measured repeatedly — 70 solutions with
            # up to 10 spectra each (dataset README + 'Replicate number'
            # column of the composition table).
            is_grouped=True,
            # Checked: no missing (NaN) target values; 0 mg/L means the
            # analyte is absent from the sample.
            has_missing_labels=False,
            # Raw spectra as deposited ("694 raw Raman spectra" per the
            # dataset README and the Data in Brief article).
            embedded_preprocessing=None,
        ),
        "single_cell_prokaryotes": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="8cd34fckgt",
            name="Single-Cell Prokaryotes (Bacteria + Archaea)",
            short_name="Single-Cell Prokaryotes",
            license="CC BY 4.0",
            loader=lambda cache_path: MendeleyLoader._load_single_cell_prokaryotes(cache_path),
            metadata={
                "full_name": (
                    "Single-Cell Raman Fingerprinting of Prokaryotes "
                    "(Kanno et al. 2021)"
                ),
                "source": "https://data.mendeley.com/datasets/8cd34fckgt/1",
                "doi": "10.17632/8cd34fckgt.1",
                "paper": "https://doi.org/10.1016/j.isci.2021.102975",
                "bibtex": (
                    "@article{Kanno_2021, title={Machine learning-assisted "
                    "single-cell Raman fingerprinting for in situ and "
                    "nondestructive classification of prokaryotes}, "
                    "volume={24}, ISSN={2590-3425}, url={http://dx.doi.org/"
                    "10.1016/j.isci.2021.102975}, DOI={10.1016/"
                    "j.isci.2021.102975}, number={8}, journal={iScience}, "
                    "publisher={Elsevier}, author={Kanno, Nanako and Kato, "
                    "Shingo and Ohkuma, Moriya and Matsui, Motomu and "
                    "Iwasaki, Wataru and Shigeto, Shinsuke}, year={2021}, "
                    "month=aug, pages={102975}}"
                ),
                "citation": [
                    "Shigeto, S. (2021). Machine learning-assisted single-cell "
                    "Raman fingerprinting for in situ and nondestructive "
                    "classification of prokaryotes. Kanno et al. Mendeley "
                    "Data, v1. https://doi.org/10.17632/8cd34fckgt.1"
                ],
                "description": (
                    "Spontaneous Raman spectra of single prokaryotic cells for "
                    "species classification: 782 cells of 6 species -- 3 "
                    "bacteria (Bacillus subtilis JCM 1465T, Escherichia coli "
                    "JCM 20135, Thermus thermophilus JCM 10941T) and 3 archaea "
                    "(Sulfolobus acidocaldarius JCM 8929T, Thermococcus "
                    "kodakarensis JCM 12380T, Nitrososphaera viennensis JCM "
                    "19564T) -- across three replicate acquisitions (2021), "
                    "including 62 cells picked out of a three-species mixed "
                    "culture (21 Bacillus_subtilis, 21 Nitrososphaera "
                    "_viennensis, 20 Thermus_thermophilus). 632.8 nm HeNe "
                    "excitation (3 mW at the sample point), 30 s integration "
                    "per cell, Nikon 100x NA 1.3 oil-immersion objective, "
                    "CCD at -70 C. Native spectra span ~671-3,070 cm^-1 at "
                    "1,600 points (~1.48 cm^-1 spacing); every acquisition "
                    "session has its own per-day wavenumber axis (un-corrected "
                    "in the deposit), so the loader interpolates all spectra "
                    "onto a common 710-3,042 cm^-1, 1 cm^-1 grid. group_ids "
                    "are the 19 acquisition sessions (18 species x date "
                    "directories of 40 cells + 1 mixed-culture session of 62); "
                    "no two spectra are measurements of the same cell, so the "
                    "session is the finest correlated unit (shared instrument "
                    "state and culture). Spectra are raw as deposited: the "
                    "authors' pipeline (cosmic-ray despiking, PBS subtraction, "
                    "neon-lamp wavenumber correction, dual-region baseline "
                    "subtraction and vector normalization per the STAR "
                    "Protocols paper doi:10.1016/j.xpro.2022.101812) is not "
                    "applied; the 19 per-session PBS references and 12 neon "
                    "standards needed to replicate it are in the archive "
                    "(excluded from this dataset)."
                ),
            },
            # Checked: cells within one acquisition session share instrument
            # state, wavenumber axis and (for the mixed culture) the same
            # physical culture; each cell is otherwise a distinct physical
            # specimen. Note: the mixed-culture session is multi-label by
            # nature (one culture, three species) -- the only group that is
            # not label-pure.
            is_grouped=True,
            # Checked: every cell file carries a species label via its
            # session directory; the unlabeled reference files (PBS, neon)
            # are excluded.
            has_missing_labels=False,
            # Raw spectra as deposited (no cosmic-ray removal, PBS
            # subtraction, wavenumber correction, baseline correction, or
            # normalization); see the authors' protocol for the full
            # pipeline.
            embedded_preprocessing=None,
        ),
    }

    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_file_listing(dataset_id: str = _MENDELEY_DATASET_ID, version: int = _MENDELEY_VERSION) -> List[dict]:
        """Return the list of file metadata dicts from the Mendeley Data API."""
        url = f"https://data.mendeley.com/api/datasets/{dataset_id}/files?version={version}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _download_files(shared_root: str, dataset_id: str = _MENDELEY_DATASET_ID, version: int = _MENDELEY_VERSION) -> None:
        """Download all dataset files into *shared_root* if not already cached."""
        os.makedirs(shared_root, exist_ok=True)

        files = MendeleyLoader._fetch_file_listing(dataset_id=dataset_id, version=version)
        for f in files:
            file_name = f.get("filename") or f.get("name")
            download_url = f.get("download_url") or f.get("content_details", {}).get("download_url")
            if not file_name or not download_url:
                MendeleyLoader.logger.warning(f"Could not determine filename or URL for entry: {f}")
                continue
            out_path = os.path.join(shared_root, file_name)
            if not os.path.exists(out_path):
                MendeleyLoader.logger.debug(f"Downloading {file_name} …")
                LoaderTools.download(
                    url=download_url,
                    out_dir_path=shared_root,
                    out_file_name=file_name,
                )

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_ft_raman_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a ``.0`` FT-Raman file.

        Format: plain-text, two comma-separated columns (wavenumber cm⁻¹, intensity).
        Returns ``(raman_shifts, intensities)`` sorted by ascending wavenumber.
        """
        data = np.loadtxt(path, delimiter=",")
        wavenumbers = data[:, 0]
        intensities = data[:, 1]
        order = np.argsort(wavenumbers)
        return wavenumbers[order], intensities[order]

    @staticmethod
    def _read_spc_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a ``.spc`` SERS file using spectrochempy.

        Returns ``(raman_shifts, intensities)`` sorted by ascending wavenumber.
        """
        import spectrochempy as scp
        ds = scp.read_spc(path)
        wavenumbers = ds.x.data.astype(float)
        intensities = ds.data.squeeze().astype(float)
        order = np.argsort(wavenumbers)
        return wavenumbers[order], intensities[order]

    # ------------------------------------------------------------------
    # Dataset loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_adulterants(
        cache_path: str,
        instrument: str = "sers",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Download (if needed) and parse illicit-adulterant spectra.

        Args:
            cache_path: Unused; kept for API compatibility. Cache root is resolved
                from the loader's ``CACHE_DIR.Mendeley`` environment variable.
            instrument: ``"sers"`` for ``.spc`` MISA files or
                ``"ft_raman"`` for ``.0`` FT-Raman files.

        Returns:
            Tuple of ``(spectra, raman_shifts, targets, class_names)``.
        """
        if instrument not in ("sers", "ft_raman"):
            raise ValueError(f"Unknown instrument '{instrument}'. Use 'sers' or 'ft_raman'.")

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Mendeley)
        if cache_root is None:
            raise ValueError("Mendeley cache root is not set.")

        shared_root = os.path.join(cache_root, "illicit_adulterants")
        MendeleyLoader._download_files(shared_root)

        spectra_list: List[np.ndarray] = []
        raman_shifts_ref: Optional[np.ndarray] = None
        class_names: List[str] = []

        if instrument == "ft_raman":
            # Each *.0 file is one compound spectrum
            files = sorted(
                f for f in os.listdir(shared_root) if f.endswith(".0")
            )
            if not files:
                raise FileNotFoundError(f"No .0 FT-Raman files found in {shared_root}.")
            for fname in files:
                compound = os.path.splitext(fname)[0]  # e.g. "Acetildenafil"
                wn, intensity = MendeleyLoader._read_ft_raman_file(os.path.join(shared_root, fname))
                if raman_shifts_ref is None:
                    raman_shifts_ref = wn
                spectra_list.append(intensity)
                class_names.append(compound)

        else:  # sers
            # Each *_MISA.spc file is one compound spectrum
            files = sorted(
                f for f in os.listdir(shared_root) if f.endswith("_MISA.spc")
            )
            if not files:
                raise FileNotFoundError(f"No _MISA.spc SERS files found in {shared_root}.")
            for fname in files:
                compound = fname.replace("_MISA.spc", "")  # e.g. "Acetildenafil"
                wn, intensity = MendeleyLoader._read_spc_file(os.path.join(shared_root, fname))
                if raman_shifts_ref is None:
                    raman_shifts_ref = wn
                spectra_list.append(intensity)
                class_names.append(compound)

        spectra = np.stack(spectra_list).astype(float)
        encoded_targets, target_names = encode_labels(pd.Series(class_names))

        MendeleyLoader.logger.debug(
            f"Loaded {instrument}: {spectra.shape[0]} spectra × {spectra.shape[1]} points, "
            f"{len(target_names)} compounds"
        )

        return spectra, raman_shifts_ref, encoded_targets, list(target_names)

    @staticmethod
    def _load_microplastics(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Download (if needed) and parse weathered microplastics Raman spectra.

        Dataset: Mendeley kpygrf9fg6 — 167 TXT files (``sta-*.txt`` standard,
        ``wea-*.txt`` weathered) plus ``content.xlsx`` metadata mapping filenames
        to polymer type labels.

        Returns:
            Tuple of ``(spectra, raman_shifts, targets, class_names)``.
        """
        _DATASET_ID = "kpygrf9fg6"
        _VERSION = 1

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Mendeley)
        if cache_root is None:
            raise ValueError("Mendeley cache root is not set.")

        shared_root = os.path.join(cache_root, "microplastics_weathered")
        MendeleyLoader._download_files(shared_root, dataset_id=_DATASET_ID, version=_VERSION)

        xlsx_path = os.path.join(shared_root, "content.xlsx")
        filename_to_label: dict = {}
        if os.path.isfile(xlsx_path):
            try:
                meta = pd.read_excel(xlsx_path)
                fname_col = next(
                    (c for c in meta.columns if c.lower() in ("filename", "file", "name")),
                    meta.columns[0],
                )
                label_col = next(
                    (c for c in meta.columns if c.lower() in ("polymer", "type", "material", "class", "label")),
                    meta.columns[1] if len(meta.columns) > 1 else meta.columns[0],
                )
                filename_to_label = dict(zip(meta[fname_col].astype(str), meta[label_col].astype(str)))
            except Exception as e:
                MendeleyLoader.logger.warning(f"Could not read content.xlsx: {e}")

        txt_files = sorted(
            f for f in os.listdir(shared_root)
            if f.endswith(".txt") and (f.startswith("sta-") or f.startswith("wea-"))
        )
        if not txt_files:
            raise FileNotFoundError(f"No TXT spectrum files found in {shared_root}")

        spectra_list: List[np.ndarray] = []
        raman_shifts_list: List[np.ndarray] = []
        labels_list: List[str] = []

        for fname in txt_files:
            fpath = os.path.join(shared_root, fname)
            try:
                data = np.loadtxt(fpath)
                if data.ndim != 2 or data.shape[1] < 2:
                    continue
                wn = data[:, 0]
                intensity = data[:, 1]
                order = np.argsort(wn)
                raman_shifts_list.append(wn[order])
                spectra_list.append(intensity[order])
                label = filename_to_label.get(fname) or filename_to_label.get(os.path.splitext(fname)[0])
                if label is None:
                    label = "standard" if fname.startswith("sta-") else "weathered"
                labels_list.append(label)
            except Exception as e:
                MendeleyLoader.logger.warning(f"Failed to read {fname}: {e}")
                continue

        if not spectra_list:
            raise FileNotFoundError(f"No spectra could be loaded from {shared_root}")

        raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)
        encoded_targets, target_names = encode_labels(pd.Series(labels_list))

        MendeleyLoader.logger.debug(
            f"Loaded microplastics_weathered: {spectra.shape[0]} spectra × {spectra.shape[1]} points, "
            f"{len(target_names)} polymer types"
        )

        return spectra, raman_shifts, encoded_targets, list(target_names)

    # ------------------------------------------------------------------
    # water_quality_ions (Mendeley cxf3zxsc3k)
    # ------------------------------------------------------------------

    # The three regression targets, in the deposited column order.
    _WATER_QUALITY_TARGETS = ["Nitrate [mg/L]", "Nitrite [mg/L]", "Sulphate [mg/L]"]

    @staticmethod
    def _load_water_quality_ions(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Download (if needed) and parse the water-quality ion quantification
        dataset (Mendeley cxf3zxsc3k v3, Luciani et al. 2026).

        694 raw Raman spectra (532 nm, Horiba iHR320 micro-Raman) of aqueous
        nitrate/nitrite/sulphate solutions with three concentration targets
        in mg/L. File layout (all_spectra_dataset_CSV.csv): 'Spectra ID'
        column, 1,019 wavenumber columns (401-3,899 cm^-1, non-uniform
        grid), then the three concentration label columns.

        Returns a tuple of (spectra, raman_shifts, targets, target_names,
        group_ids) where targets is (694, 3) and group_id identifies the
        prepared solution (one unique concentration combination each,
        measured up to 10 times).
        """
        _DATASET_ID = "cxf3zxsc3k"
        _VERSION = 3

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Mendeley)
        if cache_root is None:
            raise ValueError("Mendeley cache root is not set.")

        shared_root = os.path.join(cache_root, "water_quality_ions")
        MendeleyLoader._download_files(shared_root, dataset_id=_DATASET_ID, version=_VERSION)

        csv_path = os.path.join(shared_root, "all_spectra_dataset_CSV.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Could not find all_spectra_dataset_CSV.csv in {shared_root}")

        df = pd.read_csv(csv_path)
        id_col = "Spectra ID"
        if id_col not in df.columns:
            raise RuntimeError(f"[!] water_quality_ions: missing '{id_col}' column in {csv_path}")
        label_cols = [str(c).strip() for c in df.columns[-3:]]
        if label_cols != ["Nitrate", "Nitrite", "Sulphate"]:
            raise RuntimeError(f"[!] water_quality_ions: unexpected label columns {label_cols}")

        wn_cols = df.columns[1:-3]
        raman_shifts = np.array([float(c) for c in wn_cols])
        if raman_shifts.shape != (1019,):
            raise RuntimeError(
                f"[!] water_quality_ions: expected 1,019 wavenumber columns, "
                f"got {raman_shifts.shape[0]}."
            )
        if not np.all(np.diff(raman_shifts) > 0):
            raise RuntimeError("[!] water_quality_ions: wavenumber axis is not ascending.")

        ids = df[id_col].to_numpy()
        if len(ids) != 694 or len(set(ids.tolist())) != 694:
            raise RuntimeError(
                f"[!] water_quality_ions: expected 694 unique spectra IDs, "
                f"got {len(set(ids.tolist()))} of {len(ids)}."
            )

        spectra = df[wn_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError("[!] water_quality_ions: missing intensity values.")
        targets = df.iloc[:, -3:].to_numpy(dtype=float)
        if np.isnan(targets).any():
            raise RuntimeError("[!] water_quality_ions: missing target values.")

        # Group id = the prepared solution: each unique (nitrate, nitrite,
        # sulphate) combination was prepared once and measured repeatedly
        # (70 solutions, up to 10 spectra each; see the dataset README).
        group_ids = np.array(
            [f"NO3_{r[0]:g}_NO2_{r[1]:g}_SO4_{r[2]:g}" for r in targets], dtype=str
        )

        MendeleyLoader.logger.debug(
            f"Loaded water_quality_ions: {spectra.shape[0]} spectra × "
            f"{spectra.shape[1]} points, {len(set(group_ids.tolist()))} prepared solutions"
        )

        return spectra, raman_shifts, targets, list(MendeleyLoader._WATER_QUALITY_TARGETS), group_ids

    # ------------------------------------------------------------------
    # single_cell_prokaryotes (Mendeley 8cd34fckgt)
    # ------------------------------------------------------------------

    # Archive layout: three replicate zips, each with one directory per
    # (species, acquisition date) plus a PBS reference file (excluded).
    # Directory names carry a 'Bacilus' typo (replicate 1) and a space in
    # 'Thermus thermophilus' (replicate 1).
    _SINGLE_CELL_ZIPS = (
        "replicate_1_for_main_figures.zip",
        "replicate_2_for_supplymental_replicate1.zip",
        "replicate_3_for_supplymental_replicate2.zip",
    )

    # Normalized species-directory prefix -> canonical species name.
    _SINGLE_CELL_SPECIES = {
        "bacillus_subtilis": "Bacillus_subtilis",
        "bacilus_subtilis": "Bacillus_subtilis",  # typo in the archive
        "escherichia_coli": "Escherichia_coli",
        "nitrososphaera_viennensis": "Nitrososphaera_viennensis",
        "sulfolobus_acidocaldarius": "Sulfolobus_acidocaldarius",
        "thermococcus_kodakarensis": "Thermococcus_kodakarensis",
        "thermus_thermophilus": "Thermus_thermophilus",
    }

    # All session axes lie within [709.97, 3042.55]; the common grid is a
    # 1 cm^-1 step over the intersection (finer than the native ~1.48 cm^-1).
    _SINGLE_CELL_GRID = np.arange(710.0, 3043.0, 1.0)

    @staticmethod
    def _session_species(dir_name: str) -> Optional[str]:
        """Map a session directory name to its canonical species, or None."""
        s = dir_name.lower().replace(" ", "_")
        if s.endswith("_in_mixture"):
            s = s[: -len("_in_mixture")]
        s = re.sub(r"_\d{6}$", "", s)  # acquisition date
        s = re.sub(r"_[23]$", "", s)  # replicate marker
        return MendeleyLoader._SINGLE_CELL_SPECIES.get(s)

    @staticmethod
    def _load_single_cell_prokaryotes(
        cache_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray] | None:
        """
        Download (if needed) and parse the single-cell prokaryote dataset
        (Mendeley 8cd34fckgt v1, Kanno et al. 2021).

        Single-cell Raman spectra (632.8 nm, 3 mW, 30 s/cell, Nikon 100x
        NA 1.3) of 6 prokaryotic species — 3 bacteria and 3 archaea —
        across three replicate acquisitions, plus cells picked out of a
        three-species mixed culture. Each .asc is one cell: two
        whitespace-separated columns (wavenumber, intensity), 1,600
        points. Every acquisition session has its own (per-day) wavenumber
        axis, so spectra are interpolated onto a common 710-3,042 cm^-1
        1 cm^-1 grid.

        Excluded (reference standards, in the archive for replicating the
        authors' pipeline): the per-session PBS references (pbs10t.asc)
        and the 12 neon lamp standards (neon_standards.zip).

        Returns a tuple of (spectra, raman_shifts, targets, class_names,
        group_ids) with group_id = acquisition session (species x
        replicate x date; the three mixed-culture species share one
        session). No two spectra are measurements of the same cell.
        """
        _DATASET_ID = "8cd34fckgt"
        _VERSION = 1

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Mendeley)
        if cache_root is None:
            raise ValueError("Mendeley cache root is not set.")

        shared_root = os.path.join(cache_root, "single_cell_prokaryotes")
        MendeleyLoader._download_files(shared_root, dataset_id=_DATASET_ID, version=_VERSION)

        grid = MendeleyLoader._SINGLE_CELL_GRID
        spectra_list: List[np.ndarray] = []
        labels: List[str] = []
        group_ids: List[str] = []

        for zip_name in MendeleyLoader._SINGLE_CELL_ZIPS:
            zip_path = os.path.join(shared_root, zip_name)
            if not os.path.isfile(zip_path):
                raise FileNotFoundError(f"Could not find {zip_name} in {shared_root}")
            zip_stem = os.path.splitext(zip_name)[0]
            extract_dir = os.path.join(shared_root, zip_stem)
            if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
                if LoaderTools.extract_zip_file_content(zip_path, zip_stem) is None:
                    raise RuntimeError(f"[!] single_cell_prokaryotes: could not extract {zip_name}.")
            # The zip holds a root folder named like itself.
            roots = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
            if len(roots) != 1:
                raise RuntimeError(
                    f"[!] single_cell_prokaryotes: expected one root directory in {extract_dir}, found {roots}."
                )
            sessions_root = os.path.join(extract_dir, roots[0])
            rep_match = re.match(r"replicate_(\d+)_", zip_name)
            rep = rep_match.group(1) if rep_match else "?"

            for dir_name in sorted(os.listdir(sessions_root)):
                session_dir = os.path.join(sessions_root, dir_name)
                if not os.path.isdir(session_dir):
                    continue
                files = sorted(os.listdir(session_dir))
                # The 3-species mixed culture: one session directory with a
                # subdirectory per identified species.
                subdirs = [d for d in files if os.path.isdir(os.path.join(session_dir, d))]
                if subdirs:
                    if dir_name != "3species_mixed_200802":
                        raise RuntimeError(f"[!] single_cell_prokaryotes: unexpected session {dir_name}.")
                    group = f"rep{rep}_mixture_200802"
                    for sub in subdirs:
                        species = MendeleyLoader._session_species(sub)
                        if species is None:
                            raise RuntimeError(
                                f"[!] single_cell_prokaryotes: unrecognized species directory {sub}."
                            )
                        for f in sorted(os.listdir(os.path.join(session_dir, sub))):
                            if not re.fullmatch(r"\d+\.asc", f):
                                continue
                            axis, inten = np.loadtxt(os.path.join(session_dir, sub, f), unpack=True)
                            if axis.shape != (1600,) or not np.all(np.diff(axis) > 0):
                                raise RuntimeError(
                                    f"[!] single_cell_prokaryotes: unexpected axis in {dir_name}/{sub}/{f}."
                                )
                            spectra_list.append(np.interp(grid, axis, inten))
                            labels.append(species)
                            group_ids.append(group)
                    continue
                species = MendeleyLoader._session_species(dir_name)
                if species is None:
                    raise RuntimeError(
                        f"[!] single_cell_prokaryotes: unrecognized species directory {dir_name}."
                    )
                for f in sorted(files):
                    if not re.fullmatch(r"\d+\.asc", f):
                        continue  # PBS reference (pbs10t.asc)
                    axis, inten = np.loadtxt(os.path.join(session_dir, f), unpack=True)
                    if axis.shape != (1600,) or not np.all(np.diff(axis) > 0):
                        raise RuntimeError(
                            f"[!] single_cell_prokaryotes: unexpected axis in {dir_name}/{f}."
                        )
                    date = re.search(r"(\d{6})$", dir_name)
                    if date is None:
                        raise RuntimeError(
                            f"[!] single_cell_prokaryotes: no acquisition date in directory {dir_name}."
                        )
                    spectra_list.append(np.interp(grid, axis, inten))
                    labels.append(species)
                    group_ids.append(f"rep{rep}_{species.lower()}_{date.group(1)}")

        if not spectra_list:
            raise RuntimeError("[!] single_cell_prokaryotes: no spectra parsed.")

        spectra = np.asarray(spectra_list, dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError("[!] single_cell_prokaryotes: missing intensity values.")

        counts = Counter(labels)
        expected = {
            "Bacillus_subtilis": 141,
            "Escherichia_coli": 120,
            "Nitrososphaera_viennensis": 141,
            "Sulfolobus_acidocaldarius": 120,
            "Thermococcus_kodakarensis": 120,
            "Thermus_thermophilus": 140,
        }
        if dict(counts) != expected:
            raise RuntimeError(
                f"[!] single_cell_prokaryotes: unexpected per-species counts {dict(counts)} (expected {expected})."
            )
        per_group = Counter(group_ids)
        if len(per_group) != 19 or any(v not in (40, 62) for v in per_group.values()):
            raise RuntimeError(
                f"[!] single_cell_prokaryotes: expected 19 sessions (18 x 40 + 1 x 62), "
                f"got {len(per_group)} (sizes {sorted(per_group.values())})."
            )

        targets, class_names = encode_labels(labels)
        MendeleyLoader.logger.debug(
            f"Loaded single_cell_prokaryotes: {spectra.shape[0]} spectra x {spectra.shape[1]} points, "
            f"{len(class_names)} species, {len(per_group)} sessions"
        )
        return spectra, grid, targets, list(class_names), np.array(group_ids)

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    @staticmethod
    def download_dataset(dataset_name: str, cache_path: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError(
            "Use load_dataset() — it handles downloading automatically via the Mendeley Data API."
        )

    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None,
        load_data: bool = True,
    ) -> Optional[RamanDataset]:
        """
        Load a Mendeley-hosted dataset, downloading it first if necessary.

        Args:
            dataset_name: Key from :attr:`DATASETS`.
            cache_path: Override the default cache directory.
            load_data: If ``False``, return a metadata-only :class:`RamanDataset`.

        Returns:
            A :class:`RamanDataset` object.
        """
        if not LoaderTools.is_dataset_available(dataset_name, MendeleyLoader.DATASETS):
            raise FileNotFoundError(f"Dataset '{dataset_name}' is not available in MendeleyLoader.")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Mendeley)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Mendeley)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        MendeleyLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = MendeleyLoader.DATASETS[dataset_name]

        if load_data:
            result = dataset_info.loader(dataset_cache_path)
            if result is None:
                raise FileNotFoundError(
                    f"Could not load dataset '{dataset_name}'. "
                    "Expected files may be missing — check the logs for details."
                )
            if len(result) == 5:
                spectra, raman_shifts, targets, class_names, group_ids = result
            else:
                spectra, raman_shifts, targets, class_names = result
                group_ids = None
        else:
            spectra = raman_shifts = targets = class_names = group_ids = None

        return RamanDataset(
            info=dataset_info,
            raman_shifts=raman_shifts,
            spectra=spectra,
            targets=targets,
            target_names=class_names,
            group_ids=group_ids,
        )
