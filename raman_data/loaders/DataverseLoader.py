import logging
import os
import zipfile
from typing import List, Optional

import numpy as np
import requests

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, HASH_TYPE, APPLICATION_TYPE


_DATAVERSE_SERVER = "https://dataverse.harvard.edu"

# RamanSTS (Boroji et al. 2025): one .npy per patient and tissue, <TISSUE>/P<patient>_<TISSUE>.npy inside
# RamanSTS_part1.zip / RamanSTS_part2.zip; each holds whole or partial 51x51 maps of 1,401-point spectra.
_STS_DOI = "10.7910/DVN/NGEZPA"
# Pinned dataset version: a revision (e.g. the planned map ids) is picked up only by bumping this,
# and the cache is per version so a bump downloads the new files.
_STS_VERSION = "1.1"
_STS_N_SPECTRA = 286_672
_STS_FILES = ("RamanSTS_part1.zip", "RamanSTS_part2.zip")
_STS_TISSUES = {
    "FAT": "Fat",
    "MSC": "Muscle",
    "SKN": "Skin layers",
    "LEM": "Leiomyoma",
    "MLS": "Myxoid liposarcoma",
    "PLS": "High-grade pleomorphic liposarcoma",
    "LEI": "Leiomyosarcoma",
    "HMS": "High-grade myxofibrosarcoma",
}
_STS_BIBTEX = (
    "@article{Boroji_2025, title={Ex-vivo Raman spectroscopy and AI-based classification of soft tissue sarcomas}, "
    "volume={20}, ISSN={1932-6203}, url={http://dx.doi.org/10.1371/journal.pone.0330618}, "
    "DOI={10.1371/journal.pone.0330618}, number={9}, journal={PLOS ONE}, publisher={Public Library of Science (PLoS)}, "
    "author={Boroji, Maede and Danesh, Vahid and Barrera, Doris and Lee, Eric and Arauz, Paul G. and others}, "
    "year={2025}, pages={e0330618}}"
)


class DataverseLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on a Dataverse installation (Harvard Dataverse).

    Lists a dataset's files through the Dataverse native API and downloads each one, checked
    against the MD5 the API reports.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "dataverse")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Dataverse)

    DATASETS = {
        "soft_tissue_sarcoma": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="soft_tissue_sarcoma",
            name="Soft Tissue Sarcoma (RamanSTS)",
            short_name="RamanSTS",
            license="CC BY-NC 4.0",
            loader=lambda cache_path: DataverseLoader._load_ramansts(cache_path),
            metadata={
                "full_name": "RamanSTS dataset, a large collection of Raman spectra of soft tissue sarcomas",
                "source": f"https://doi.org/{_STS_DOI}",
                "doi": _STS_DOI,
                "paper": "https://doi.org/10.1371/journal.pone.0330618",
                "bibtex": _STS_BIBTEX,
                "citation": [
                    "Boroji M, Danesh V, Barrera D, Lee E, Arauz PG, et al. (2025) Ex-Vivo Raman Spectroscopy and "
                    "AI-Based Classification of Soft Tissue Sarcomas. PLOS ONE 20(9): e0330618. "
                    "https://doi.org/10.1371/journal.pone.0330618"
                ],
                "description": (
                    "286,672 ex-vivo Raman spectra (Renishaw inVia, 633 nm, 400-1,800 cm⁻¹, 1,401 points) "
                    "of surgically resected soft tissue from 7 patients, measured as 51x51 maps over 100 x 100 µm. "
                    "Target: tissue type (WHO classification), 8 classes: fat, muscle and skin (normal), leiomyoma "
                    "(benign), myxoid liposarcoma, high-grade pleomorphic liposarcoma, leiomyosarcoma and high-grade "
                    "myxofibrosarcoma (malignant). Leiomyoma, myxoid liposarcoma, leiomyosarcoma and high-grade "
                    "myxofibrosarcoma each come from a single patient, so for them grouped splits measure "
                    "generalisation across regions of one specimen, not across patients. Three maps (P3 muscle, "
                    "P3 pleomorphic liposarcoma, P4 myxoid liposarcoma) are partial: spectra with artefacts (tissue "
                    "contamination, ink) were removed in WiRE. Group ids are the 18 patient x tissue files (e.g. \"P3_PLS\"); the map "
                    "of each spectrum is not published yet."
                ),
            },
            # One group per patient x tissue file, the finest unit the data identifies. Four classes
            # (LEM, MLS, LEI, HMS) come from a single file each.
            is_grouped=True,
            has_missing_labels=False,
            embedded_preprocessing=(
                "fifth-order polynomial baseline correction, median-filter cosmic-ray removal, Savitzky-Golay "
                "smoothing (polynomial order 3) and min-max normalization to [0, 1] per spectrum in Renishaw WiRE, "
                "then interpolation to a 1 cm\u207b\u00b9 grid and cropping to 400-1,800 cm\u207b\u00b9 "
                "(source-applied, per the authors)"
            ),
        ),
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        if not LoaderTools.is_dataset_available(dataset_name, DataverseLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Dataverse)
        dataset_cache_path = os.path.join(LoaderTools.get_cache_root(CACHE_DIR.Dataverse), dataset_name)
        out_dir = os.path.join(dataset_cache_path, f"v{_STS_VERSION}")
        DataverseLoader._download_files(_STS_DOI, _STS_VERSION, out_dir, list(_STS_FILES))
        return out_dir

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, DataverseLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Dataverse)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Dataverse)
        dataset_cache_path = os.path.join(cache_root, dataset_name)
        dataset_info = DataverseLoader.DATASETS[dataset_name]

        if load_data:
            result = dataset_info.loader(dataset_cache_path)
            if result is None:
                raise FileNotFoundError(f"Could not load dataset {dataset_name}. Please check logs for details.")
            spectra, raman_shifts, targets, class_names, group_ids = result
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

    # ------------------------------------------------------------------
    # Dataverse API
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_file_listing(doi: str, version: str) -> List[dict]:
        """The ``dataFile`` entries of version *version* of dataset *doi*."""
        response = requests.get(
            f"{_DATAVERSE_SERVER}/api/datasets/:persistentId/versions/{version}",
            params={"persistentId": f"doi:{doi}"},
            # Harvard Dataverse answers 403 to the default python-requests user agent.
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36", "Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        return [f["dataFile"] for f in response.json()["data"]["files"]]

    @staticmethod
    def _download_files(doi: str, version: str, out_dir: str, names: List[str]) -> dict:
        """Download the files *names* of version *version* of dataset *doi* into *out_dir* (MD5-checked);
        returns name -> path."""
        os.makedirs(out_dir, exist_ok=True)
        paths = {name: os.path.join(out_dir, name) for name in names}
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if not missing:
            return paths
        listing = {f["filename"]: f for f in DataverseLoader._fetch_file_listing(doi, version)}
        for name in missing:
            if name not in listing:
                raise FileNotFoundError(f"{name} is not in Dataverse dataset doi:{doi}")
            md5 = listing[name].get("checksum", {})
            md5 = md5.get("value") if md5.get("type") == "MD5" else None
            # Dataverse converts uploaded tables (.csv -> .tab); the checksum is the uploaded file's.
            original = "?format=original" if listing[name].get("originalFileFormat") else ""
            LoaderTools.download(
                url=f"{_DATAVERSE_SERVER}/api/access/datafile/{listing[name]['id']}{original}",
                out_dir_path=out_dir,
                out_file_name=name,
                hash_target=md5,
                hash_type=HASH_TYPE.md5 if md5 else None,
            )
        return paths

    # ------------------------------------------------------------------
    # Dataset readers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ramansts(cache_path: str):
        """Tissue-type spectra of RamanSTS with patient x tissue groups (Boroji et al. 2025)."""
        paths = DataverseLoader._download_files(_STS_DOI, _STS_VERSION, os.path.join(cache_path, f"v{_STS_VERSION}"),
                                                list(_STS_FILES))

        # Read the arrays straight from the archives (no 3 GB extraction), without pickle.
        members = []
        for zip_path in paths.values():
            with zipfile.ZipFile(zip_path) as zf:
                members += [(zip_path, n) for n in zf.namelist() if n.endswith(".npy")]
        members.sort(key=lambda m: os.path.basename(m[1]))

        tissues = list(_STS_TISSUES)
        spectra, targets, groups = [], [], []
        for zip_path, member in members:
            patient, tissue = os.path.splitext(os.path.basename(member))[0].split("_")
            if tissue not in _STS_TISSUES or os.path.dirname(member) != tissue:
                raise ValueError(f"unexpected RamanSTS file {member}")
            with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
                block = np.lib.format.read_array(fh, allow_pickle=False).astype(np.float32)
            spectra.append(block)
            targets += [tissues.index(tissue)] * len(block)
            groups += [f"{patient}_{tissue}"] * len(block)

        spectra = np.concatenate(spectra, axis=0)
        if spectra.shape != (_STS_N_SPECTRA, 1401):
            raise ValueError(f"expected {_STS_N_SPECTRA} spectra of 1,401 points, got {spectra.shape}")
        raman_shifts = np.linspace(400, 1800, 1401, dtype=np.float32)
        return (
            spectra,
            raman_shifts,
            np.array(targets, dtype=np.int32),
            [_STS_TISSUES[t] for t in tissues],
            np.array(groups),
        )
