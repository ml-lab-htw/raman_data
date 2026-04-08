import logging
import os
import sqlite3
import zlib
import json
from typing import Optional

import numpy as np
import pandas as pd
import requests

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, HASH_TYPE, APPLICATION_TYPE


class FigshareLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on Figshare.

    Downloads datasets via the Figshare API.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "figshare")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Figshare)

    DATASETS = {
        "pharmaceutical_ingredients": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="pharmaceutical_ingredients",
            name="Pharmaceutical Ingredients",
            short_name="Pharma Ingredients",
            license="CC BY 4.0",
            loader=lambda cache_path: FigshareLoader._load_api(cache_path),
            metadata={
                "full_name": "Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development",
                "source": "https://springernature.figshare.com/ndownloader/articles/27931131/versions/1",
                "paper": "https://doi.org/10.1038/s41597-025-04848-6",
                "bibtex": "@article{Flanagan_2025, title={Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development}, volume={12}, ISSN={2052-4463}, url={http://dx.doi.org/10.1038/s41597-025-04848-6}, DOI={10.1038/s41597-025-04848-6}, number={1}, journal={Scientific Data}, publisher={Springer Science and Business Media LLC}, author={Flanagan, Aaron R. and Glavin, Frank G.}, year={2025}, month=mar}",
                "citation": [
                    "Flanagan, A.R., Glavin, F.G. Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development. Sci Data 12, 498 (2025)."
                ],
                "description": "A Raman spectral dataset comprising 3,510 spectra from 32 chemical substances. This dataset includes organic solvents and reagents commonly used in API development, along with information regarding the products in the XLSX, and code to visualise and perform technical validation on the data.",
            }
        ),
        "serum_prostate_cancer": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="serum_prostate_cancer",
            name="Prostate Cancer SERS Serum",
            short_name="SERS PCa/BPH/Control",
            license="CC BY 4.0",
            loader=lambda cache_path: FigshareLoader._load_serum_prostate_cancer(cache_path),
            metadata={
                "full_name": "ComFilE for PCa",
                "source": "https://figshare.com/articles/dataset/ComFilE_for_PCa/28107395",
                "paper": "https://doi.org/10.1016/j.xcrm.2024.101579",
                "bibtex": "@article{bi2024sersomes, title={SERSomes for metabolic phenotyping and prostate cancer diagnosis}, author={Bi, Xinyuan and Wang, Jiayi and Xue, Bingsen and He, Chang and Liu, Fugang and Chen, Haoran and Lin, Linley Li and Dong, Baijun and Li, Butang and Jin, Cheng and others}, journal={Cell Reports Medicine}, volume={5}, number={6}, year={2024}, publisher={Elsevier}}",
                "doi": "10.6084/m9.figshare.28107395.v1",
                "citation": [
                    "Xue, Bingsen (2024). ComFilE for PCa. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28107395.v1"
                ],
                "description": "SERS serum metabolite spectra for classifying prostate cancer (PCa), benign prostatic hyperplasia (BPH), and healthy controls. 424 serum samples from male participants (ages 41–89) collected at Ren Ji Hospital, Shanghai Jiao Tong University. Organized as SERSomes (200 spectra per sample, 638 nm laser, quartz capillary), spanning 600–1800 cm⁻¹ (724 data points per spectrum).",
            }
        ),
        "serum_alzheimer_disease": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="serum_alzheimer_disease",
            name="Alzheimer's SERS Serum",
            short_name="SERS AD/MCI/Control",
            license="CC BY 4.0",
            loader=lambda cache_path: FigshareLoader._load_serum_alzheimer_disease(cache_path),
            metadata={
                "full_name": "ComFilE for AD",
                "source": "https://figshare.com/articles/dataset/ComFilE_for_AD/28107578",
                "doi": "10.6084/m9.figshare.28107578.v1",
                "citation": [
                    "Xue, Bingsen (2024). ComFilE for AD. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28107578.v1"
                ],
                "description": "SERS serum metabolite spectra for classifying Alzheimer's disease (AD), mild cognitive impairment (MCI), and healthy controls. 139 serum samples (57 male, 82 female) collected at Rui Jin Hospital, Shanghai Jiao Tong University. Organized as SERSomes (200 spectra per sample, 638 nm laser, quartz capillary), spanning 600–1800 cm⁻¹. 17 PyTorch tensor files organised by class label.",
            }
        ),
        "comfile_stroke": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="comfile_stroke",
            name="Stroke SERS Serum",
            short_name="Stroke SERS Serum",
            license="CC BY 4.0",
            loader=lambda cache_path: FigshareLoader._load_comfile_stroke(cache_path),
            metadata={
                "full_name": "ComFilE: SERS serum spectra for stroke classification",
                "source": "https://figshare.com/articles/dataset/ComFilE_/28107431",
                "doi": "10.6084/m9.figshare.28107431",
                "paper": "https://www.nature.com/articles/s42256-025-01027-5",
                "bibtex": "@article{Xue_2025, title={Deep spectral component filtering as a foundation model for spectral analysis demonstrated in metabolic profiling}, volume={7}, ISSN={2522-5839}, url={http://dx.doi.org/10.1038/s42256-025-01027-5}, DOI={10.1038/s42256-025-01027-5}, number={5}, journal={Nature Machine Intelligence}, publisher={Springer Science and Business Media LLC}, author={Xue, Bingsen and Bi, Xinyuan and Dong, Zheyi and Xu, Yunzhe and Liang, Minghui and Fang, Xin and Yuan, Yizhe and Wang, Ruoxi and Liu, Shuyu and Jiao, Rushi and Chen, Yuze and Zu, Weitao and Wang, Chengxiang and Zhang, Jianhao and Liu, Jiang and Zhang, Qin and Yuan, Ye and Xu, Midie and Zhang, Ya and Wang, Yanfeng and Ye, Jian and Jin, Cheng}, year={2025}, month=may, pages={743--757}}",
                "citation": [
                    "Xue, B. et al. ComFilE: a large-scale benchmark dataset for computational spectroscopy. "
                    "Nat Mach Intell (2025). https://doi.org/10.1038/s42256-025-01027-5"
                ],
                "description": (
                    "SERS serum spectra for binary stroke vs. healthy-control classification. "
                    "20 tab-separated files (10 stroke, 10 healthy control), each containing ~201 spectra "
                    "across 723 wavenumber points (202.985–1999.92 cm⁻¹). ~4,020 spectra total."
                ),
            }
        ),
        "chembl_molecules": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="chembl_molecules",
            name="Raman-ChEMBL Molecules",
            short_name="Raman-ChEMBL",
            license="CC BY-NC-ND 4.0",
            loader=lambda cache_path: FigshareLoader._load_raman_chembl_part2(cache_path),
            metadata={
                "full_name": "Raman-ChEMBL-part2",
                "source": "https://figshare.com/articles/dataset/Raman-ChEMBL-part2/28594295",
                "doi": "10.6084/m9.figshare.28594295.v3",
                "paper": "https://doi.org/10.1038/s41597-025-05289-x",
                "bibtex": "@article{liang2025dataset, title={A Dataset of Raman and Infrared Spectra as an Extension to the ChEMBL}, author={Liang, Jiechun and Ling, Jack and Xu, Limin and Zhu, Xi}, journal={Scientific Data}, volume={12}, number={1}, pages={939}, year={2025}, publisher={Nature Publishing Group UK London}}",
                "citation": [
                    "Liang, J., Ling, J., Zhu, X. Raman-ChEMBL-part2. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28594295.v3"
                ],
                "description": "140k DFT-computed Raman spectra for ChEMBL drug-like molecules. Targets: HOMO-LUMO gap, HOMO/LUMO energies, isotropic polarizability, heat capacity, dipole moment.",
            }
        ),
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from Figshare loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, FigshareLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Figshare)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        FigshareLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = FigshareLoader.DATASETS[dataset_name]

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
    def fetch_figshare_metadata(article_id: int) -> dict:
        r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}")
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _load_api(cache_path):

        metadata = FigshareLoader.fetch_figshare_metadata(27931131)
        files = metadata["files"]

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        dataset_cache = os.path.join(
            cache_root,
            "pharmaceutical_ingredient",
        )
        os.makedirs(dataset_cache, exist_ok=True)

        for f in files:
            file_url = f["download_url"]
            file_name = f["name"]
            file_md5 = f.get("computed_md5")

            out_path = os.path.join(dataset_cache, file_name)

            # download only if missing
            if not os.path.exists(out_path):
                LoaderTools.download(
                    url=file_url,
                    out_dir_path=dataset_cache,
                    out_file_name=file_name,
                    hash_target=file_md5,
                    hash_type=HASH_TYPE.md5,
                    referer="https://figshare.com/",
                )

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
    def _broaden_stick_spectrum(
            freqs: list,
            activities: list,
            grid: np.ndarray,
            fwhm: float = 10.0,
    ) -> np.ndarray:
        """Broaden a stick Raman spectrum onto a wavenumber grid using Lorentzian functions."""
        spectrum = np.zeros(len(grid), dtype=np.float32)
        gamma = fwhm / 2.0
        for freq, activity in zip(freqs, activities):
            if freq <= 0:
                continue
            spectrum += activity * (gamma ** 2) / ((grid - freq) ** 2 + gamma ** 2)
        return spectrum

    _CHEMBL_TARGET_COLS = ["Eg", "Homo", "Lumo", "isotropic_pol", "heat_capacity", "Dtotal"]

    @staticmethod
    def _load_raman_chembl_part2(cache_path):
        file_id = 54667760
        file_name = "Raman-ChEMBL-part2.db"
        file_md5 = "d8e05c28db8c5d533e297124640ce269"

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        dataset_cache = os.path.join(cache_root, "raman_chembl_part2")
        os.makedirs(dataset_cache, exist_ok=True)

        db_path = os.path.join(dataset_cache, file_name)

        if not os.path.exists(db_path):
            download_url = f"https://ndownloader.figshare.com/files/{file_id}"
            LoaderTools.download(
                url=download_url,
                out_dir_path=dataset_cache,
                out_file_name=file_name,
                hash_target=file_md5,
                hash_type=HASH_TYPE.md5,
                referer="https://figshare.com/",
            )

        target_cols = FigshareLoader._CHEMBL_TARGET_COLS
        raman_shifts = np.arange(0, 3802, 2, dtype=np.float32)
        spectra_cache = os.path.join(dataset_cache, "spectra.npy")
        targets_cache = os.path.join(dataset_cache, "targets.npy")

        if os.path.exists(spectra_cache) and os.path.exists(targets_cache):
            FigshareLoader.logger.info("Loading Raman-ChEMBL-part2 from cache")
            return np.load(spectra_cache), raman_shifts, np.load(targets_cache), target_cols

        con = sqlite3.connect(db_path)
        try:
            if os.path.exists(spectra_cache):
                # Spectra already cached — only fetch scalar target columns
                FigshareLoader.logger.info("Extracting target columns from db")
                cols_sql = ", ".join(target_cols)
                rows = con.execute(
                    f"SELECT {cols_sql} FROM molecule WHERE blob_data IS NOT NULL"
                ).fetchall()
                targets = np.array(rows, dtype=np.float32)
                np.save(targets_cache, targets)
                return np.load(spectra_cache), raman_shifts, targets, target_cols

            cols_sql = ", ".join(target_cols)
            rows = con.execute(
                f"SELECT {cols_sql}, blob_data FROM molecule WHERE blob_data IS NOT NULL"
            ).fetchall()
        finally:
            con.close()

        FigshareLoader.logger.info(f"Processing {len(rows)} molecules from Raman-ChEMBL-part2")

        n_targets = len(target_cols)
        spectra = np.empty((len(rows), len(raman_shifts)), dtype=np.float32)
        targets = np.empty((len(rows), n_targets), dtype=np.float32)

        for i, row in enumerate(rows):
            target_vals = row[:n_targets]
            blob = row[n_targets]
            vib = json.loads(zlib.decompress(bytes(blob)))
            spectra[i] = FigshareLoader._broaden_stick_spectrum(
                vib["freq"], vib["Raman Activ"], raman_shifts
            )
            targets[i] = target_vals

        np.save(spectra_cache, spectra)
        np.save(targets_cache, targets)

        return spectra, raman_shifts, targets, target_cols

    _COMFILE_STROKE_CLASSES = {1: "Stroke", 2: "Control"}

    @staticmethod
    def _load_comfile_stroke(cache_path):
        article_id = 28107431
        metadata = FigshareLoader.fetch_figshare_metadata(article_id)
        files = metadata["files"]

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        if cache_root is None:
            raise Exception(f"No cache root found for Figshare")

        dataset_cache = os.path.join(cache_root, "comfile_stroke")
        os.makedirs(dataset_cache, exist_ok=True)

        # Download only .txt files (the SERS spectra)
        txt_files = [f for f in files if f["name"].endswith(".txt")]
        for f in txt_files:
            out_path = os.path.join(dataset_cache, f["name"])
            if not os.path.exists(out_path):
                LoaderTools.download(
                    url=f["download_url"],
                    out_dir_path=dataset_cache,
                    out_file_name=f["name"],
                    hash_target=f.get("computed_md5"),
                    hash_type=HASH_TYPE.md5,
                    referer="https://figshare.com/",
                )

        spectra_list, label_list = [], []
        raman_shifts = None

        for f in txt_files:
            name = f["name"]
            # filename format: "[1]T ..." (stroke) or "[2]H ..." (healthy control)
            bracket_end = name.find("]")
            if bracket_end == -1:
                continue
            class_idx = int(name[1:bracket_end])

            file_path = os.path.join(dataset_cache, name)
            # Tab-separated: row 0 = header (wavenumbers + leading empty col),
            # rows 1+ = spectra (first col = spatial position, rest = intensities)
            df = pd.read_csv(file_path, sep="\t", header=0, index_col=0)
            # Row 0 of the original file is the wavenumber header — after read_csv
            # with header=0, the columns are the wavenumber strings.
            if raman_shifts is None:
                raman_shifts = np.array([float(c) for c in df.columns], dtype=np.float32)
            data = df.values.astype(np.float32)
            spectra_list.append(data)
            label_list.extend([class_idx] * len(data))

        spectra = np.concatenate(spectra_list, axis=0)
        labels = np.array(label_list, dtype=np.int32)
        class_map = FigshareLoader._COMFILE_STROKE_CLASSES
        target_names = [class_map[i] for i in sorted(class_map)]
        targets = labels - 1  # remap to 0-based

        return spectra, raman_shifts, targets, target_names

    @staticmethod
    def _load_serum_alzheimer_disease(cache_path):
        return FigshareLoader._load_comfile(
            article_id=28107578,
            cache_dir_name="serum_alzheimer_disease",
            class_map={1: "AD", 2: "MCI", 3: "Control"},
        )

    @staticmethod
    def _load_serum_prostate_cancer(cache_path):
        return FigshareLoader._load_comfile(
            article_id=28107395,
            cache_dir_name="serum_prostate_cancer",
            class_map={1: "PCa", 2: "BPH", 3: "Control"},
        )

    @staticmethod
    def _load_comfile(article_id: int, cache_dir_name: str, class_map: dict):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Loading ComFilE datasets requires PyTorch. "
                "Install it with: pip install torch"
            )

        metadata = FigshareLoader.fetch_figshare_metadata(article_id)
        files = metadata["files"]

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Figshare)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        dataset_cache = os.path.join(cache_root, cache_dir_name)
        os.makedirs(dataset_cache, exist_ok=True)

        for f in files:
            out_path = os.path.join(dataset_cache, f["name"])
            if not os.path.exists(out_path):
                LoaderTools.download(
                    url=f["download_url"],
                    out_dir_path=dataset_cache,
                    out_file_name=f["name"],
                    hash_target=f.get("computed_md5"),
                    hash_type=HASH_TYPE.md5,
                    referer="https://figshare.com/",
                )

        spectra_list, label_list = [], []
        raman_shifts = None

        for f in files:
            name = f["name"]
            # filename format: "[N] ID nobase.pt"
            bracket_end = name.find("]")
            if bracket_end == -1:
                continue
            class_idx = int(name[1:bracket_end])

            tensor = torch.load(
                os.path.join(dataset_cache, name), map_location="cpu", weights_only=True
            )
            # tensor shape: (N_spectra, N_wavenumbers) or (N_wavenumbers,)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            data = tensor.numpy().astype(np.float32)
            spectra_list.append(data)
            label_list.extend([class_idx] * len(data))

            if raman_shifts is None:
                raman_shifts = np.arange(data.shape[1], dtype=np.float32)

        spectra = np.concatenate(spectra_list, axis=0)
        labels = np.array(label_list, dtype=np.int32)
        target_names = [class_map[i] for i in sorted(class_map)]
        # remap class indices to 0-based
        targets = labels - 1

        return spectra, raman_shifts, targets, target_names
