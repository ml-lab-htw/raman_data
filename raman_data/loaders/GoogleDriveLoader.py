import io
import logging
import os
import pickle
import re
from typing import Optional, Tuple

import gdown
import numpy as np

from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE


# Lita et al. 2024 (APOLLO): one Raman map per FFPE glioma slide, sorted into the six
# DNA-methylation subtypes LGm1-6 (Drive folder 1CNAwZa3H22FQtprSRXwHXsJ8FdazIUBd, raw/LGm-<k>/).
# (subtype, slide name, Drive file id)
_GLIOMA_SLIDES = [
    (1, "HF-1293_13_1.h5_0", "1FCcdMeRog8f2SCWP6K9aLbgZ9l2-XMxZ"),
    (1, "HF-1295_V3AM_2.h5_1", "1FI9jm70u8WRwE0AMfs7_IuM1A63Qazr3"),
    (1, "HF-2534_V2B_1.h5_2", "1FTJ3POtEwzmCD5RwZSApg1x7RhkWrI7s"),
    (1, "HF-448_V5B_1.h5_3", "1FF3969y98Ofu-Io7AyYm-rGna3aQ4rAs"),
    (1, "HF-868_1_2.h5_4", "1FPBjI_MSbC7vuZbf5c38l4og0msglsUH"),
    (2, "HF-1010_V1T_1.h5_0", "1HcefaQrN_gmCy5NdI6g3SPazXiMtwYsZ"),
    (2, "HF-1016_IAT_2.h5_1", "1IkxpEBVbfqBOs_AJbLmg7bmTvy6ylomq"),
    (2, "HF-1334_V58-B_2_1.h5_2", "1IDdl7ru62dISZQtsGnslAssEK0xuatsO"),
    (2, "HF-1887_via-t_2_1.h5_3", "1I0GmWQV1zjIaMr6hK030iD4xGZw04pf7"),
    (2, "HF-2070_V1T_1.h5_4", "1I3cw8ZQ9rM5wueTdOCanqitPpbQdAC9O"),
    (2, "HF-2776_V2B_2.h5_5", "1IuM9MNxkxtFtrBuPGb3GKEgnBFan0J7i"),
    (2, "HF-305_v4b_1_1.h5_6", "1Hat_JJiCHt5ql62H-a8l2MLsZFZXWpo2"),
    (2, "HF-3271_VIB_2.h5_7", "1I2ZHjweu1bzOd2PmDk-SGfHmpmXW4XQw"),
    (2, "HF-3337_V3T_1.h5_8", "1IUoj631xmRD3ROkTFqF2mK-URwRGdmf-"),
    (2, "HF-615_V5BB_1.h5_9", "1InWcIkigPzoCoNRlglQaepwthWqysUy9"),
    (2, "HF-901_V2T_2.h5_10", "1ICtbxCU89oOz9HueM3980Kc8Oz9JwKEk"),
    (2, "HF-960_VIAT_2.h5_11", "1HbXemIWNTLEhX7lOjES0uUAnPIQDgH0f"),
    (3, "HF-2104_#5_1.h5_0", "1JMAEDHW0yDhpRgmYquTol9_iUNw_Dsl_"),
    (3, "HF-2104_#9_1.h5_1", "1JGqiMt52Y_OqiUeyTiPPOy9cypyGM5iy"),
    (3, "HF-2104_V1T_1.h5_2", "1JQnpZieyLjxY5EK5TYuz5V3EHTuYxu71"),
    (3, "HF-2614_V1B_1.h5_3", "1J2ysU3FUSkqTag41RYq-USd_-d0rDdSD"),
    (3, "HF-2849_VIT2_1.h5_4", "1Iw8Ix4dLKB3gMCLaZYTYmTKGV9JOpHc7"),
    (3, "HF-2849_VIT2_1.h5_5", "1JO0FJ8cCKPBqtA0cT9-M-rMaYCQV2FtR"),
    (3, "HF-2849_VIT2_2.h5_6", "1Iyt8R4NeWkE0n3PQ64bDLRR0wG12uCEk"),
    (3, "HF-2849_VIT_2_new2021.h5_7", "1JBU_OUyMMvtNlZU8FIb1voJoy4WE-sfY"),
    (3, "HF-2852_VIT_2_2.h5_8", "1J6ZeMQzCjn4tVyH3NLOfBD8CkP6lW1Yg"),
    (4, "HF-1002_V1AT_1.h5_0", "1GNGGArX8b3cxp-sf_y80896x0kgETDrY"),
    (4, "HF-1002_V2AT_1.h5_1", "1FjGw91IHbb1y62QmvijtbVZvc_mBNRk9"),
    (4, "HF-1825_V2B_1.h5_2", "1GAY2rb2W093lStpqTXRvY1Fv66Heu7xc"),
    (4, "HF-2102_V2BB_1.h5_3", "1GDc-2HU3A8OJ44wCx3A9KUT9t7wLn5uT"),
    (4, "HF-2102_V3AM_1.h5_4", "1GIfeSplECliCeS-X0Eu-R6OD-6030vfh"),
    (4, "HF-2102_V3AM_2.h5_5", "1G0cB6D8GnUg8a9F0yTSms2cuPc3JwGYy"),
    (4, "HF-2454_V1AT_1.h5_6", "1GRZqwxrhQj4nLinZR_6So8tPADqck67Z"),
    (4, "HF-2548_V1T_1.h5_7", "1GZjEA0hvV5ySuWDBSw7cq7AC9DFYn2C0"),
    (4, "HF-2715_VIL_1.h5_8", "1GSJ3pGcwRwrAdD9vVCTHHe-X08dOMERl"),
    (4, "HF-2802_V3T_1.h5_9", "1G-0pUttJuDA3k9FpH5wZQPScx78p7FqY"),
    (4, "HF-2876_V1T_1.h5_10", "1FgiwLskLJ8EAzllcUm3pP3yliBBk1UR6"),
    (4, "HF-2898_V1T_1.h5_11", "1FYDrHaFwFkswrjDBR21fcFhwKiNWIcWH"),
    (4, "HF-442_V4BB_1.h5_12", "1FZqfI7QPS4q9h0ebUqUKOWYOt33lmBWA"),
    (5, "HF-1043_V1AM_1.h5_0", "1HRbS9k_rnuYPu_pLSkepPD4aNuUZ-iWO"),
    (5, "HF-1086_#1_1.h5_1", "1Gopmi72t7gv0NAic5nAewEpcTSROTQjk"),
    (5, "HF-2355_V2AM_1.h5_2", "1H-IjItSGzurS5h3dDfHymUs14ggsqSHq"),
    (5, "HF-2485_V1B_1.h5_3", "1H12SBr1XJSrpiOpT4OxdHtXArX_kOxVn"),
    (5, "HF-2600_V1B_1.h5_4", "1HKUaOlt0wnYJ3cMnZy7o79KAR4kRuvyJ"),
    (5, "HF-2608_V1T_1.h5_5", "1HQoTHlHMPVRuQFluIGFL8WAX80ZaSuu0"),
    (5, "HF-2619_V1T_1.h5_6", "1HFHiJ1XgnSrNyWEmcAIKQI4vw5rRQyKX"),
    (5, "HF-2619_V4T_1.h5_7", "1H1wFZjAd80RzEOsRzm-t2Q-2HhwgleYQ"),
    (5, "HF-2666_V2B_1.h5_8", "1H2Mwalk-SwCNYHRrhpua_USatD6zltYD"),
    (5, "HF-682_V3AT_1.h5_9", "1Gz9K1MRITFSESYbrOptrxL3476Yt4Ty5"),
    (5, "HF-682_V3BB_1.h5_10", "1H0lhdGbsBQylIkpY3X_mBKlzZA53LvXv"),
    (5, "HF-894_9_1.h5_11", "1HOKjfk683CXQZsBFFwFCgq5kEIsLwgcu"),
    (5, "HF-894_V1BB_1.h5_12", "1GpqmpXUzxayCQUW5352cFI22SPYPTkdN"),
    (5, "HF-988_V1-T_1.h5_13", "1GnObBYhluC_-chrOfgH4nYi47xqZvgpF"),
    (5, "HF-988_V1B_1.h5_14", "1HM064aDKpRFJ-HLF6W2KQ4rTDsdfAlNr"),
    (6, "HF-2106_V3AM_1.h5_0", "1Gms6XQ9Ns_PAN0yyvsqIA7aAYFVFZn3h"),
    (6, "HF-2493_V1T_1.h5_1", "1Giw_XJybjQ6tIvknIsJZJjloAcVdi49J"),
    (6, "HF-2493_V1T_2.h5_2", "1GdCEKEqlCNA0bR1gf4JQAJ97BgmqRwGx"),
    (6, "HF-2544_V1B_1.h5_3", "1Gl1vnZb1UCLot7g2WnEWJtQmWH-N2TII"),
    (6, "HF-592_V3T_1.h5_4", "1GcSENim8rEzTB2iZa8IcVPG_xvnQEBEO"),
]
_GLIOMA_TUMOR_MAPS_ID = "1DlQhy7rAiHNj5pz5OV7nw918Mw_8eaDd"
_GLIOMA_FREQ_TABLE_ID = "1K2SP7l1RkGA8bNCYySbdo3ZYr1S8mwO7"
# LGm1-3 are the IDH-mutant subtypes, LGm4-6 the IDH-wildtype ones (Ceccarelli et al. 2016).
_GLIOMA_TASKS = {
    "methylation": (["LGm1", "LGm2", "LGm3", "LGm4", "LGm5", "LGm6"], lambda k: k - 1),
    "idh": (["IDH-mutant", "IDH-wildtype"], lambda k: 0 if k <= 3 else 1),
}
_GLIOMA_BIBTEX = "@article{Lita_2024, title={Raman-based machine-learning platform reveals unique metabolic differences between IDHmut and IDHwt glioma}, volume={26}, ISSN={1523-5866}, url={http://dx.doi.org/10.1093/neuonc/noae101}, DOI={10.1093/neuonc/noae101}, number={11}, journal={Neuro-Oncology}, publisher={Oxford University Press (OUP)}, author={Lita, Adrian and Sj\u00f6berg, Joel and P\u0103cioianu, David and Siminea, Nicoleta and Celiku, Orieta and Dowdy, Tyrone and P\u0103un, Andrei and Gilbert, Mark R and Noushmehr, Houtan and Petre, Ion and Larion, Mioara}, year={2024}, month=June, pages={1994--2009}}"


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickles numpy arrays only, so a downloaded pickle cannot run code."""

    _ALLOWED = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to unpickle {module}.{name}")


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


class GoogleDriveLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on Google Drive.

    Downloads datasets via gdown from shared Google Drive links.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "gdrive")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.GoogleDrive)

    DATASETS = {
        **{
            f"rruff_mineral_{processed.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.MaterialScience,
                id=f"rruff_mineral_{processed.lower()}",
                name=f"RRUFF Minerals ({processed})",
                short_name=f"RRUFF ({'Raw' if processed == 'Raw' else 'Pre.'})",
                license="Free access",
                loader=lambda cache_path, p=processed: GoogleDriveLoader._load_onewarmheart(
                    cache_path, split=f"mineral_{p.lower()}"),
                metadata={
                    "full_name": f"RRUFF Database - {processed} Spectra",
                    "source": "https://rruff.info/",
                    "paper": "https://doi.org/10.1515/9783110417104-003",
                    "bibtex": "@inbook{Lafuente_2015, title={The power of databases: The RRUFF project}, ISBN={9783110417104}, url={http://dx.doi.org/10.1515/9783110417104-003}, DOI={10.1515/9783110417104-003}, booktitle={Highlights in Mineralogical Crystallography}, publisher={DE GRUYTER}, author={Lafuente, Barbara and Downs, R. T. and Yang, H. and Stone, N.}, year={2015}, month=nov, pages={1--30}}",
                    "citation": [
                        "Lafuente, B., Downs, R. T., Yang, H., & Stone, N. (2015). The power of databases: the RRUFF project. Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, Eds., Berlin, Germany, W. De Gruyter, 1–30."
                    ],
                    "description": "Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm).",
                },
                # Checked (Raw variant only): no missing (NaN) label values.
                # Preprocess variant not checked -- left as None, not assumed.
                has_missing_labels=(False if processed == "Raw" else None),
            )
            for processed in ["Raw", "Preprocess"]
        },
        **{
            f"organic_compounds_{processed.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Chemical,
                id=f"organic_compounds_{processed.lower()}",
                name=f"Organic Compounds ({processed})",
                short_name=f"Organic Cpds. ({'Raw' if processed == 'Raw' else 'Pre.'})",
                license="unknown",
                loader=lambda cache_path, p=processed: GoogleDriveLoader._load_onewarmheart(
                    cache_path, split=f"organic_{p.lower()}"),
                metadata={
                    "full_name": f"Organic Compounds Multi-Excitation Dataset - {processed}",
                    "source": "https://data.dtu.dk/api/files/36144495",
                    "paper": "https://doi.org/10.1002/jrs.5750",
                    "bibtex": "@article{Zhang_2019, title={Transfer-learning-based Raman spectra identification}, volume={51}, ISSN={1097-4555}, url={http://dx.doi.org/10.1002/jrs.5750}, DOI={10.1002/jrs.5750}, number={1}, journal={Journal of Raman Spectroscopy}, publisher={Wiley}, author={Zhang, Rui and Xie, Huimin and Cai, Shuning and Hu, Yong and Liu, Guo-kun and Hong, Wenjing and Tian, Zhong-qun}, year={2019}, month=nov, pages={176--186}}",
                    "citation": [
                        "Zhang, Rui et al., Transfer-learning-based Raman spectra identification, Journal of Raman Spectroscopy, 2020, 51, 1, 176-186. https://doi.org/10.1002/jrs.5992"
                    ],
                    "description": f"{processed} Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data.",
                }
            )
            for processed in ["Raw", "Preprocess"]
        },
        **{
            f"glioma_{task}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Medical,
                id=f"glioma_{task}",
                name=f"Glioma {'Methylation Subtype' if task == 'methylation' else 'IDH Status'}",
                short_name=f"Glioma ({'LGm1-6' if task == 'methylation' else 'IDH'})",
                license="CC BY-NC (paper; non-commercial)",
                loader=lambda cache_path, t=task: GoogleDriveLoader._load_glioma_apollo(cache_path, t),
                metadata={
                    "full_name": "APOLLO glioma Raman maps (Lita et al. 2024)",
                    "source": "https://drive.google.com/drive/folders/1CNAwZa3H22FQtprSRXwHXsJ8FdazIUBd",
                    "paper": "https://doi.org/10.1093/neuonc/noae101",
                    "doi": "10.1093/neuonc/noae101",
                    "bibtex": _GLIOMA_BIBTEX,
                    "citation": [
                        "Lita, A., Sj\u00f6berg, J., P\u0103cioianu, D., Siminea, N., Celiku, O., Dowdy, T., P\u0103un, A., Gilbert, M. R., Noushmehr, H., Petre, I., & Larion, M. (2024). Raman-based machine-learning platform reveals unique metabolic differences between IDHmut and IDHwt glioma. Neuro-Oncology, 26(11), 1994\u20132009. https://doi.org/10.1093/neuonc/noae101"
                    ],
                    "description": (
                        "Raman maps of 59 glioma tissue slides from 46 patients, 1,738 bands from 50 to 3,400 cm\u207b\u00b9. "
                        "Only the spectra the authors' DBSCAN tumour separation marks as tumour are kept. "
                        + ("Target: DNA-methylation subtype LGm1-6." if task == "methylation"
                           else "Target: IDH status (LGm1-3 IDH-mutant, LGm4-6 IDH-wildtype).")
                        + " Group ids are patients."
                    ),
                },
                # Patient ids from the slide file names (HF-<patient>); every patient has
                # one subtype, some have several slides.
                is_grouped=True,
                has_missing_labels=False,
            )
            for task in _GLIOMA_TASKS
        },
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from Google Drive loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, GoogleDriveLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.GoogleDrive)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.GoogleDrive)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        GoogleDriveLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = GoogleDriveLoader.DATASETS[dataset_name]

        if load_data:
            result = dataset_info.loader(dataset_cache_path)
            if result is None:
                raise FileNotFoundError(f"Could not load dataset {dataset_name}. Expected files may be missing. Please check logs for details.")
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

    @staticmethod
    def _load_glioma_apollo(cache_path: str, task: str):
        """Tumour spectra of the APOLLO slides with patient groups (Lita et al. 2024)."""
        import pandas as pd

        shared_root = os.path.join(os.path.dirname(cache_path), "glioma_apollo")
        os.makedirs(shared_root, exist_ok=True)

        def fetch(file_id: str, name: str) -> str:
            # gdown cannot fetch the .xlsx from this folder; Drive's direct download URL can.
            path = os.path.join(shared_root, name)
            if not os.path.exists(path):
                LoaderTools.download(
                    url=f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                    out_dir_path=shared_root,
                    out_file_name=name,
                )
            return path

        freq = pd.read_excel(fetch(_GLIOMA_FREQ_TABLE_ID, "RamanFreqTable.xlsx"), header=None)
        raman_shifts = freq.iloc[:, 1].to_numpy(dtype=np.float32)

        # One binary map per slide (1 = tumour), saved by the APOLLO notebook in the
        # natural-sort order of the slide names.
        with open(fetch(_GLIOMA_TUMOR_MAPS_ID, "tumor_maps.npy"), "rb") as fh:
            if np.lib.format.read_magic(fh) == (1, 0):
                np.lib.format.read_array_header_1_0(fh)
            else:
                np.lib.format.read_array_header_2_0(fh)
            tumor_maps = _NumpyOnlyUnpickler(io.BytesIO(fh.read())).load()
        slides = sorted(_GLIOMA_SLIDES, key=lambda s: _natural_key(s[1]))
        if len(tumor_maps) != len(slides):
            raise ValueError(f"{len(tumor_maps)} tumour maps for {len(slides)} slides")

        class_names, to_target = _GLIOMA_TASKS[task]
        spectra, targets, groups = [], [], []
        for (subtype, name, file_id), tumor in zip(slides, tumor_maps):
            cube = np.load(fetch(file_id, f"LGm{subtype}_{name}.npy"), allow_pickle=False)
            tumor = np.asarray(tumor)
            if cube.shape[:2] != tumor.shape or cube.shape[2] != len(raman_shifts):
                raise ValueError(f"{name}: map {cube.shape} does not match tumour map {tumor.shape}")
            kept = cube.reshape(-1, cube.shape[2])[tumor.reshape(-1) == 1].astype(np.float32)
            spectra.append(kept)
            targets.extend([to_target(subtype)] * len(kept))
            groups.extend([int(re.match(r"HF-(\d+)_", name).group(1))] * len(kept))

        return (
            np.concatenate(spectra, axis=0),
            raman_shifts,
            np.array(targets, dtype=np.int32),
            class_names,
            np.array(groups, dtype=np.int32),
        )

    @staticmethod
    def _load_onewarmheart(cache_path: str, split: str):
        # note that there is another repo loading these datasets differently:
        # https://github.com/lyn1874/raman_spectra_matching_with_contrastive_learning

        shared_root = os.path.join(os.path.dirname(cache_path), "dtu_raman_shared")
        zip_path = os.path.join(shared_root, "public_dataset.zip")
        extracted_dir = os.path.join(shared_root, "data")
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
                    GoogleDriveLoader.logger.error(f"[!] Failed to remove corrupted zip: {e}")

            file_id = "1X5OAdugcHVOF6k9WaWwCAu_IFh_7OwzR"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, zip_path, quiet=False)

            LoaderTools.extract_zip_file_content(zip_path)

            if not LoaderTools.is_valid_zip(zip_path):
                raise CorruptedZipFileError(zip_path)

        if "mineral" in split:
            subfolder = "RRUFF_database"
        elif "organic" in split:
            subfolder = "organics_database"
        else:
            raise ValueError(f"Unknown split name: {split}")

        split_root = os.path.join(extracted_dir, subfolder)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"[!] Expected directory not found: {split_root}")

        if "mineral" in split:
            if "raw" in split:
                raman_shifts_list = np.load(os.path.join(split_root, "xx_raw.npy"), allow_pickle=True).tolist()
                spectra_list = np.load(os.path.join(split_root, "xy_raw.npy"), allow_pickle=True).tolist()
                targets = np.load(os.path.join(split_root, "yclass.npy"), allow_pickle=True)

                raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)
            elif "preprocess" in split:
                # see: https://github.com/onewarmheart/Raman/blob/master/code-ZR/code/preprocess/rruff_interpolate.py
                raman_shifts = np.linspace(0, 1700, 1024)
                spectra_train = np.load(os.path.join(split_root, "after-preprocess", "xy_train.npy"), allow_pickle=True)
                spectra_val = np.load(os.path.join(split_root, "after-preprocess", "xy_val.npy"), allow_pickle=True)
                spectra = np.concatenate([spectra_train, spectra_val], axis=0)
                targets_train = np.load(os.path.join(split_root, "after-preprocess", "yclass_train.npy"), allow_pickle=True)
                targets_val = np.load(os.path.join(split_root, "after-preprocess", "yclass_val.npy"), allow_pickle=True)
                targets = np.concatenate([targets_train, targets_val], axis=0)
            else:
                raise ValueError(f"Unknown split type in name: {split}")
        elif "organic" in split:
            if "raw" in split:
                raman_shifts_list = np.load(os.path.join(split_root, "xx.npy"), allow_pickle=True).tolist()
                spectra_list = np.load(os.path.join(split_root, "xy.npy"), allow_pickle=True).tolist()
                targets = np.load(os.path.join(split_root, "yclass.npy"), allow_pickle=True)

                raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)
            elif "preprocess" in split:
                # see: https://github.com/onewarmheart/Raman/blob/master/code-ZR/code/preprocess/organics_interpolate.py
                raman_shifts = np.linspace(200, 3700, 1100)[:1024] # this is from the original repo. idk why they do this so complicated
                spectra = np.load(os.path.join(split_root, "after-preprocess", "xy.npy"), allow_pickle=True)
                targets = np.load(os.path.join(split_root, "after-preprocess","yclass.npy"), allow_pickle=True)

        else:
            raise ValueError(f"Unknown split name: {split}")

        class_names = np.unique(targets).tolist()
        return spectra, raman_shifts, targets, class_names
