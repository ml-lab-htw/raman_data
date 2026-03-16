import logging
import os
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
            loader=lambda cache_path: MendeleyLoader._load_adulterants(cache_path, instrument="ft_raman"),
            metadata={
                "full_name": "Dataset of Raman and SERS spectra of illicit adulterants added to dietary supplements",
                "source": "https://data.mendeley.com/datasets/y4md8znppn/1",
                "doi": "10.17632/y4md8znppn/1",
                "paper": "https://doi.org/10.1016/j.heliyon.2023.e18509",
                "citation": _CITATION,
                "license": "CC BY 4.0",
                "description": (
                    "FT-Raman spectra (1064 nm, ~33–3600 cm⁻¹, 1851 points) of 11 SERS-active pharmaceutically "
                    "active adulterants commonly found in adulterated dietary supplements. Acquired with a benchtop "
                    "Bruker RAM II FT-IR Raman module. One spectrum per compound. "
                    "Target: compound identity (classification)."
                ),
            },
        ),
        "illicit_adulterants_sers": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="illicit_adulterants_sers",
            name="Illicit Adulterants in Dietary Supplements (SERS)",
            loader=lambda cache_path: MendeleyLoader._load_adulterants(cache_path, instrument="sers"),
            metadata={
                "full_name": "Dataset of Raman and SERS spectra of illicit adulterants added to dietary supplements",
                "source": "https://data.mendeley.com/datasets/y4md8znppn/1",
                "doi": "10.17632/y4md8znppn/1",
                "paper": "https://doi.org/10.1016/j.heliyon.2023.e18509",
                "citation": _CITATION,
                "license": "CC BY 4.0",
                "description": (
                    "SERS spectra (785 nm, 400–2300 cm⁻¹, 1901 points) of 11 SERS-active illicit adulterants "
                    "found in dietary supplements. Acquired with a portable Metrohm MISA analyzer using silver "
                    "printed-SERS substrates. One spectrum per compound. "
                    "Target: compound identity (classification)."
                ),
            },
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
    def _download_files(shared_root: str) -> None:
        """Download all dataset files into *shared_root* if not already cached."""
        os.makedirs(shared_root, exist_ok=True)

        files = MendeleyLoader._fetch_file_listing()
        for f in files:
            file_name = f.get("filename") or f.get("name")
            download_url = f.get("download_url") or f.get("content_details", {}).get("download_url")
            if not file_name or not download_url:
                MendeleyLoader.logger.warning(f"Could not determine filename or URL for entry: {f}")
                continue
            out_path = os.path.join(shared_root, file_name)
            if not os.path.exists(out_path):
                MendeleyLoader.logger.info(f"Downloading {file_name} …")
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
