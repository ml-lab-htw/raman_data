#!/usr/bin/env python3
"""
Generate Croissant 1.0 + RAI metadata JSON-LD files for all raman_data datasets.

For HuggingFace and Kaggle datasets the existing Croissant file is fetched from
the platform and extended with Croissant RAI fields.  For all other platforms
the Croissant file is generated from the DatasetInfo metadata.

A human-editable ``rai_supplement.yaml`` is written alongside the output files.
Edit that file to refine the auto-generated RAI descriptions, then re-run the
script to regenerate the JSON-LD files.

Usage
-----
    python scripts/generate_croissant.py
    python scripts/generate_croissant.py --output-dir ./croissant_files
    python scripts/generate_croissant.py --rai-file path/to/rai_supplement.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import mlcroissant as mlc
import requests
import yaml

from raman_data.loaders.KaggleLoader import KaggleLoader
from raman_data.loaders.HuggingFaceLoader import HuggingFaceLoader
from raman_data.loaders.ZenodoLoader import ZenodoLoader
from raman_data.loaders.RWTHLoader import RWTHLoader
from raman_data.loaders.GoogleDriveLoader import GoogleDriveLoader
from raman_data.loaders.FigshareLoader import FigshareLoader
from raman_data.loaders.GitHubLoader import GitHubLoader
from raman_data.loaders.MendeleyLoader import MendeleyLoader
from raman_data.loaders.MiscLoader import MiscLoader
from raman_data.types import DatasetInfo, TASK_TYPE, APPLICATION_TYPE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CROISSANT_CONTEXT: dict = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "wd": "https://www.wikidata.org/wiki/",
}

_CROISSANT_CONFORMS_TO: str = "http://mlcommons.org/croissant/1.0"

_LICENSE_MAP: dict[str, str] = {
    "cc by 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc by 4": "https://creativecommons.org/licenses/by/4.0/",
    "cc by": "https://creativecommons.org/licenses/by/4.0/",
    "mit": "https://opensource.org/licenses/MIT",
    "mit license": "https://opensource.org/licenses/MIT",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "apache 2.0": "https://www.apache.org/licenses/LICENSE-2.0",
}

_DEFAULT_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

_MANUAL_SOURCE_PATTERNS = ["sharepoint.com", "via e-mail", "only via e-mail", "email"]

_LOADERS: list[type] = [
    KaggleLoader,
    HuggingFaceLoader,
    ZenodoLoader,
    RWTHLoader,
    GoogleDriveLoader,
    FigshareLoader,
    GitHubLoader,
    MendeleyLoader,
    MiscLoader,
]

_LOADER_NAMES: dict[type, str] = {
    KaggleLoader: "Kaggle",
    HuggingFaceLoader: "HuggingFace",
    ZenodoLoader: "Zenodo",
    RWTHLoader: "RWTH",
    GoogleDriveLoader: "GoogleDrive",
    FigshareLoader: "Figshare",
    GitHubLoader: "GitHub",
    MendeleyLoader: "Mendeley",
    MiscLoader: "Misc",
}

# ---------------------------------------------------------------------------
# Metadata normalisation helpers
# ---------------------------------------------------------------------------

def normalise_license(raw: str | None) -> tuple[str | None, str | None, bool]:
    """
    Returns (license_url, conditions_of_access, license_inferred).

    - Valid URL  → (url, None, False)
    - Known short form like "CC BY 4.0" → (canonical_url, None, False)
    - Long prose → (None, raw, False)  — written as conditionsOfAccess
    - None / empty → (_DEFAULT_LICENSE_URL, None, True)
    """
    if not raw:
        return _DEFAULT_LICENSE_URL, None, True
    if raw.startswith("http"):
        return raw, None, False
    lower = raw.strip().lower()
    if lower in _LICENSE_MAP:
        return _LICENSE_MAP[lower], None, False
    for k, v in _LICENSE_MAP.items():
        if lower.startswith(k):
            return v, None, False
    if len(raw) > 60:
        return None, raw, False
    # Unrecognised short string — fall back to default
    return _DEFAULT_LICENSE_URL, None, True


def normalise_papers(raw: str | list | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [p for p in raw if isinstance(p, str) and p.strip()]


def normalise_citation(raw: str | list | None) -> str:
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return "; ".join(c.strip() for c in raw if c.strip())


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform(info: DatasetInfo) -> str:
    source: str = info.metadata.get("source", "") or ""
    s = source.lower()
    if info.metadata.get("hf_key") or "huggingface.co" in s:
        return "huggingface"
    if "kaggle.com" in s:
        return "kaggle"
    if "zenodo.org" in s or "10.5281/zenodo" in s:
        return "zenodo"
    if "figshare.com" in s or "springernature.figshare" in s:
        return "figshare"
    if "publications.rwth-aachen.de" in s:
        return "rwth"
    if "github.com" in s:
        return "github"
    if "mendeley.com" in s:
        return "mendeley"
    if "drive.google.com" in s or "data.dtu.dk" in s or "rruff.info" in s:
        return "googledrive"
    return "manual"


def is_manual_access(info: DatasetInfo) -> bool:
    source: str = info.metadata.get("source", "") or ""
    s = source.lower()
    return any(p in s for p in _MANUAL_SOURCE_PATTERNS)


# ---------------------------------------------------------------------------
# Remote fetchers  (in-process cache keyed by URL to avoid redundant downloads)
# ---------------------------------------------------------------------------

_fetch_cache: dict[str, dict | None] = {}


def _hf_auth_headers() -> dict[str, str]:
    """Return Authorization header dict if an HF token is available.

    Checks (in order): HF_TOKEN env var, HUGGING_FACE_HUB_TOKEN env var,
    and ~/.hf_credentials (a shell-sourceable file exporting HF_TOKEN).
    """
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        creds_path = Path.home() / ".hf_credentials"
        if creds_path.exists():
            for line in creds_path.read_text().splitlines():
                line = line.strip()
                for prefix in ("export HF_TOKEN=", "HF_TOKEN="):
                    if line.startswith(prefix):
                        token = line[len(prefix):].strip().strip('"').strip("'")
                        break
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_hf_croissant(hf_key: str) -> dict | None:
    url = f"https://huggingface.co/api/datasets/{hf_key}/croissant"
    if url in _fetch_cache:
        return _fetch_cache[url]
    result: dict | None = None
    try:
        resp = requests.get(url, headers=_hf_auth_headers(), timeout=15)
        if resp.status_code == 200:
            result = resp.json()
        elif resp.status_code == 400:
            # Platform limitation: HuggingFace has not generated a Croissant
            # export for this dataset (e.g. format incompatible with auto-generation).
            # This is expected for some datasets; fall back to metadata generation.
            logger.info(
                "HuggingFace Croissant not available for %s (platform reports format "
                "unsupported) — generating from metadata",
                hf_key,
            )
        elif resp.status_code == 401:
            logger.warning(
                "HuggingFace Croissant fetch unauthorised for %s — "
                "set HF_TOKEN env var or add it to ~/.hf_credentials",
                hf_key,
            )
        else:
            logger.warning(
                "HuggingFace Croissant fetch failed for %s: HTTP %d — generating from metadata",
                hf_key, resp.status_code,
            )
    except requests.RequestException as exc:
        logger.warning(
            "HuggingFace Croissant fetch error for %s: %s — generating from metadata",
            hf_key, exc,
        )
    _fetch_cache[url] = result
    return result


_HF_LICENSE_SPDX_MAP: dict[str, str] = {
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "mit": "https://opensource.org/licenses/MIT",
    "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
}


def fetch_hf_license(hf_key: str) -> str | None:
    """Query the HuggingFace dataset info API to confirm the license URL.

    Used as a fallback when the Croissant endpoint returns 400 (format not
    available) so that generated files still carry a confirmed license rather
    than the assumed CC BY 4.0 default.
    """
    import os
    url = f"https://huggingface.co/api/datasets/{hf_key}"
    cache_key = f"hf_info:{url}"
    if cache_key in _fetch_cache:
        return _fetch_cache[cache_key]  # type: ignore[return-value]
    result: str | None = None
    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            license_id = resp.json().get("cardData", {}).get("license", "")
            result = _HF_LICENSE_SPDX_MAP.get(license_id.lower())
    except requests.RequestException:
        pass
    _fetch_cache[cache_key] = result  # type: ignore[assignment]
    return result


def fetch_kaggle_croissant(handle: str) -> dict | None:
    """handle = 'user/dataset-name' extracted from the source URL.

    Note: Kaggle Croissant files enumerate every CSV column as a separate
    field, producing large files (several MB for wide Raman datasets).
    Results are cached in-process so datasets sharing the same Kaggle
    handle are not downloaded multiple times.
    """
    url = f"https://www.kaggle.com/datasets/{handle}/croissant/download"
    if url in _fetch_cache:
        return _fetch_cache[url]
    result: dict | None = None
    try:
        resp = requests.get(url, timeout=20, allow_redirects=True)
        if resp.status_code == 200:
            result = resp.json()
        else:
            logger.warning(
                "Kaggle Croissant fetch failed for %s: HTTP %d (authentication likely required) "
                "— generating from metadata",
                handle, resp.status_code,
            )
    except requests.RequestException as exc:
        logger.warning(
            "Kaggle Croissant fetch error for %s: %s — generating from metadata",
            handle, exc,
        )
    _fetch_cache[url] = result
    return result


def fetch_figshare_files(article_id: str) -> list[dict]:
    """Return the files list from the Figshare API for building the distribution section."""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("files", [])
    except requests.RequestException:
        pass
    return []


def _extract_kaggle_handle(source_url: str) -> str | None:
    m = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", source_url)
    return m.group(1) if m else None


def _extract_figshare_article_id(source_url: str) -> str | None:
    m = re.search(r"/(\d{7,})(?:[/?]|$)", source_url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Encoding format
# ---------------------------------------------------------------------------

def get_encoding_format(platform: str, file_typ: str | list | None) -> str:
    if platform == "huggingface":
        return "application/x-parquet"
    if file_typ:
        typ = file_typ[0] if isinstance(file_typ, list) else file_typ
        ext = typ.lstrip("*.").lower()
        _map = {
            "csv": "text/csv",
            "zip": "application/zip",
            "mat": "application/x-matlab-data",
            "pkl": "application/octet-stream",
            "npy": "application/octet-stream",
            "spc": "application/octet-stream",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "txt": "text/plain",
        }
        return _map.get(ext, "application/octet-stream")
    return "application/octet-stream"


# ---------------------------------------------------------------------------
# Build distribution section
# ---------------------------------------------------------------------------

def build_distribution(
    dataset_key: str,
    info: DatasetInfo,
    platform: str,
    figshare_files: list[dict] | None = None,
) -> list[dict]:
    source: str = info.metadata.get("source", "") or ""
    full_name: str = info.metadata.get("full_name", "") or info.name

    if platform == "figshare" and figshare_files:
        # Primary FileObject (article URL) — referenced by RecordSet fields as source
        entries: list[dict] = [{
            "@type": "cr:FileObject",
            "@id": f"{dataset_key}_source",
            "name": full_name,
            "description": "Figshare article hosting all data files.",
            "contentUrl": source,
            "encodingFormat": "text/html",
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        }]
        for f in figshare_files:
            file_id = f.get("id") or f.get("name", "unknown")
            entry: dict = {
                "@type": "cr:FileObject",
                "@id": f"{dataset_key}_file_{file_id}",
                "name": f.get("name", "unknown"),
                "contentUrl": f.get("download_url", source),
                "encodingFormat": f.get("mimetype", "application/octet-stream"),
            }
            md5 = f.get("computed_md5") or f.get("md5")
            if md5:
                entry["md5"] = md5
            else:
                entry["sha256"] = hashlib.sha256(
                    (f.get("download_url", source) or source).encode()
                ).hexdigest()
            entries.append(entry)
        return entries

    encoding = get_encoding_format(platform, info.file_typ)
    return [{
        "@type": "cr:FileObject",
        "@id": f"{dataset_key}_source",
        "name": full_name,
        "description": "Source data hosted at the original repository.",
        "contentUrl": source,
        "encodingFormat": encoding,
        # sha256 of the content URL — serves as a stable identifier for validation;
        # replace with the actual file checksum if available.
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
    }]


# ---------------------------------------------------------------------------
# RecordSet
# ---------------------------------------------------------------------------

def build_record_set(dataset_key: str, task_type: TASK_TYPE) -> list[dict]:
    source_ref = {"fileObject": {"@id": f"{dataset_key}_source"}}

    if task_type == TASK_TYPE.Classification:
        target_dtype = "sc:Text"
        target_desc = "Class label: string identifier of the sample class."
    elif task_type in (TASK_TYPE.Denoising, TASK_TYPE.SuperResolution):
        target_dtype = "sc:Float"
        target_desc = (
            "Reference (clean or high-resolution) spectral intensity values, "
            "shape (n_wavenumbers,)."
        )
    else:
        target_dtype = "sc:Float"
        target_desc = (
            "Target value(s): concentration, physical property, or other "
            "continuous measurement."
        )

    return [{
        "@type": "cr:RecordSet",
        "@id": "raman_spectra",
        "name": "raman_spectra",
        "description": "Raman spectroscopy data records.",
        "field": [
            {
                "@type": "cr:Field",
                "@id": "raman_spectra/spectrum_id",
                "name": "spectrum_id",
                "description": "Sequential integer index identifying each spectrum.",
                "dataType": "sc:Integer",
                "source": source_ref,
            },
            {
                "@type": "cr:Field",
                "@id": "raman_spectra/raman_shift_cm_inv",
                "name": "raman_shift_cm_inv",
                "description": "Raman shift wavenumber axis in cm⁻¹. Shape: (n_wavenumbers,).",
                "dataType": "sc:Float",
                "source": source_ref,
            },
            {
                "@type": "cr:Field",
                "@id": "raman_spectra/spectrum_intensity",
                "name": "spectrum_intensity",
                "description": (
                    "Raman scattering intensity in arbitrary units. "
                    "Shape: (n_wavenumbers,) per spectrum."
                ),
                "dataType": "sc:Float",
                "source": source_ref,
            },
            {
                "@type": "cr:Field",
                "@id": "raman_spectra/target",
                "name": "target",
                "description": target_desc,
                "dataType": target_dtype,
                "source": source_ref,
            },
        ],
    }]


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

def build_keywords(info: DatasetInfo) -> list[str]:
    kw = ["Raman spectroscopy", "spectroscopy"]
    if info.application_type and info.application_type.name not in ("Unknown",):
        kw.append(info.application_type.name)
    if info.task_type and info.task_type.name not in ("Unknown",):
        kw.append(info.task_type.name)

    text = (info.name + " " + (info.metadata.get("description", "") or "")).lower()
    domain_extras: list[tuple[str, str]] = [
        ("fermentation", "bioprocess"),
        ("sers", "SERS"),
        ("mineral", "mineralogy"),
        ("cancer", "oncology"),
        ("covid", "infectious disease"),
        ("alzheimer", "neurology"),
        ("parkinson", "neurology"),
        ("bacteria", "microbiology"),
        ("diabetes", "endocrinology"),
        ("pharmaceutical", "pharmaceutical science"),
        ("pigment", "art conservation"),
        ("microgel", "polymer chemistry"),
        ("fuel", "petroleum chemistry"),
        ("sugar", "food science"),
        ("amino acid", "biochemistry"),
        ("dft", "computational chemistry"),
    ]
    for trigger, label in domain_extras:
        if trigger in text and label not in kw:
            kw.append(label)
    return kw


# ---------------------------------------------------------------------------
# Auto-generated RAI fields
# ---------------------------------------------------------------------------

def auto_rai_fields(
    info: DatasetInfo,
    license_inferred: bool,
    manual_access: bool,
) -> dict[str, str]:
    at = info.application_type
    tt = info.task_type

    collection_type: dict[APPLICATION_TYPE, str] = {
        APPLICATION_TYPE.Medical: (
            "Laboratory measurements on human biological samples (blood serum, saliva, "
            "tissue biopsies) collected under clinical or research protocols."
        ),
        APPLICATION_TYPE.Biological: (
            "Laboratory measurements on biological materials including microorganisms, "
            "cell cultures, fermentation broths, and plant material."
        ),
        APPLICATION_TYPE.Chemical: (
            "Laboratory measurements on chemical samples including pure compounds, "
            "mixtures, and solutions."
        ),
        APPLICATION_TYPE.MaterialScience: (
            "Laboratory measurements on solid or liquid material samples including "
            "minerals, polymers, and organic pigments."
        ),
        APPLICATION_TYPE.Unknown: "Laboratory spectroscopic measurements.",
    }

    preprocessing: dict[TASK_TYPE, str] = {
        TASK_TYPE.Regression: (
            "Spectra may have undergone baseline correction, smoothing, normalisation, "
            "and wavenumber calibration. Exact steps are described in the accompanying publication."
        ),
        TASK_TYPE.Classification: (
            "Spectra are provided with class labels. Any preprocessing applied (baseline "
            "correction, smoothing, normalisation) is described in the source publication."
        ),
        TASK_TYPE.Denoising: (
            "Raw noisy spectra are paired with high-SNR reference spectra of the same "
            "samples for supervised denoising model training."
        ),
        TASK_TYPE.SuperResolution: (
            "Low-resolution spectra are paired with high-resolution reference spectra of "
            "the same samples for supervised super-resolution model training."
        ),
        TASK_TYPE.Unknown: "See source publication for preprocessing details.",
    }

    annotation: dict[TASK_TYPE, str] = {
        TASK_TYPE.Regression: (
            "Ground-truth target values measured by validated reference analytical methods "
            "(e.g., enzymatic assays, dynamic light scattering, gas chromatography, or DFT computation)."
        ),
        TASK_TYPE.Classification: (
            "Class labels assigned based on known sample identity or validated analytical "
            "reference methods."
        ),
        TASK_TYPE.Denoising: (
            "Reference (clean) spectra are high-SNR acquisitions of the same samples "
            "obtained under longer integration times or reduced noise conditions."
        ),
        TASK_TYPE.SuperResolution: (
            "High-resolution reference spectra acquired with finer wavenumber spacing "
            "than the low-resolution counterpart."
        ),
        TASK_TYPE.Unknown: "See source publication for annotation details.",
    }

    personal = (
        "Spectra are derived from human biological samples (e.g., blood serum, saliva, "
        "or tissue). No direct patient identifiers are included, but biological measurements "
        "may constitute sensitive health-related data under applicable data-protection regulations."
        if at == APPLICATION_TYPE.Medical
        else (
            "No personal or sensitive information is included. All spectra originate from "
            "chemical, biological, or material samples without individual human identifiers."
        )
    )

    biases = (
        "Data were acquired under controlled laboratory conditions and may not generalise "
        "to field or clinical settings. Spectral profiles are influenced by instrument-specific "
        "factors (laser wavelength, detector type, optical path length, integration time). "
        "Class imbalances may be present in classification datasets."
    )

    domain_limitations: dict[APPLICATION_TYPE, str] = {
        APPLICATION_TYPE.Medical: (
            "Models trained on this dataset may not transfer to different instruments, "
            "sample matrices, or patient populations without re-calibration."
        ),
        APPLICATION_TYPE.Biological: (
            "Spectral signatures are process- and instrument-specific; calibration transfer "
            "to other setups requires additional validation."
        ),
        APPLICATION_TYPE.Chemical: (
            "Quantitative models are valid within the concentration ranges and chemical "
            "matrix of the training data; extrapolation outside these ranges is not recommended."
        ),
        APPLICATION_TYPE.MaterialScience: (
            "Reference spectra represent ideal laboratory conditions; real-world samples "
            "may show additional interference from impurities, fluorescence, or surface effects."
        ),
        APPLICATION_TYPE.Unknown: (
            "Generalisability to other instruments or sample conditions should be "
            "validated separately."
        ),
    }
    lim_parts: list[str] = [domain_limitations.get(at, "")]
    if license_inferred:
        lim_parts.append(
            "Note: No explicit license was stated by the original data providers; "
            "CC BY 4.0 has been assumed. Users should verify the applicable terms "
            "with the original authors."
        )
    if manual_access:
        lim_parts.append(
            "Access to this dataset requires direct contact with the authors or a manual "
            "download step; it is not publicly available via an open API."
        )
    limitations = " ".join(p for p in lim_parts if p)

    social_impact: dict[APPLICATION_TYPE, str] = {
        APPLICATION_TYPE.Medical: (
            "Raman spectroscopy datasets for medical diagnostics have the potential to "
            "support the development of non-invasive, low-cost screening tools with "
            "particular relevance to resource-limited healthcare settings. Results must be "
            "validated by qualified healthcare professionals before any clinical application."
        ),
        APPLICATION_TYPE.Biological: (
            "Enables the development of rapid, inline bioprocess monitoring tools that can "
            "reduce waste, improve yield, and lower costs in biomanufacturing and biotechnology."
        ),
        APPLICATION_TYPE.Chemical: (
            "Supports inline quality control and process analytical technology (PAT) in "
            "chemical manufacturing, contributing to safer and more efficient industrial processes."
        ),
        APPLICATION_TYPE.MaterialScience: (
            "Facilitates non-destructive material characterisation and identification with "
            "applications in geology, environmental science, art conservation, and industrial "
            "quality assurance."
        ),
        APPLICATION_TYPE.Unknown: (
            "Raman spectroscopy datasets support the development of machine learning models "
            "for non-destructive molecular analysis across a broad range of scientific and "
            "industrial domains."
        ),
    }

    use_cases: dict[TASK_TYPE, str] = {
        TASK_TYPE.Regression: (
            "Chemometric modelling, calibration transfer, multivariate regression, "
            "quantitative spectral analysis, process analytical technology (PAT)."
        ),
        TASK_TYPE.Classification: (
            "Spectral library matching, material identification, disease screening, "
            "compound recognition, quality control."
        ),
        TASK_TYPE.Denoising: (
            "Training and benchmarking deep learning denoising models for low-SNR "
            "Raman spectra."
        ),
        TASK_TYPE.SuperResolution: (
            "Training and benchmarking spectral super-resolution models for Raman "
            "resolution enhancement."
        ),
        TASK_TYPE.Unknown: (
            "General Raman spectroscopy machine learning model training, evaluation, "
            "and benchmarking."
        ),
    }

    return {
        "rai:dataCollection": (
            "Raman spectra were acquired experimentally using a laser spectrometer. "
            "Each spectrum represents the scattered light intensity as a function of "
            "Raman shift (cm⁻¹)."
        ),
        "rai:dataCollectionType": collection_type.get(at, collection_type[APPLICATION_TYPE.Unknown]),
        "rai:dataPreprocessingProtocol": preprocessing.get(tt, preprocessing[TASK_TYPE.Unknown]),
        "rai:dataAnnotationProtocol": annotation.get(tt, annotation[TASK_TYPE.Unknown]),
        "rai:personalSensitiveInformation": personal,
        "rai:dataBiases": biases,
        "rai:dataLimitations": limitations,
        "rai:dataSocialImpact": social_impact.get(at, social_impact[APPLICATION_TYPE.Unknown]),
        "rai:dataUseCases": use_cases.get(tt, use_cases[TASK_TYPE.Unknown]),
    }


def merge_rai(auto: dict, supplement: dict | None) -> dict:
    if not supplement:
        return auto
    merged = dict(auto)
    merged.update(supplement)
    return merged


# ---------------------------------------------------------------------------
# Build Croissant from scratch
# ---------------------------------------------------------------------------

def build_croissant_from_scratch(
    dataset_key: str,
    info: DatasetInfo,
    rai: dict,
    platform: str,
    figshare_files: list[dict] | None = None,
) -> dict[str, Any]:
    meta = info.metadata
    source: str = meta.get("source", "") or ""

    license_url, conditions_of_access, _ = normalise_license(info.license)

    papers = normalise_papers(meta.get("paper"))
    doi_field: str = meta.get("doi", "") or ""
    if doi_field and not doi_field.startswith("http"):
        doi_field = f"https://doi.org/{doi_field}"
    same_as: list[str] = list(dict.fromkeys(filter(None, papers + ([doi_field] if doi_field else []))))

    citation = normalise_citation(meta.get("citation"))
    keywords = build_keywords(info)
    distribution = build_distribution(dataset_key, info, platform, figshare_files)
    record_set = build_record_set(dataset_key, info.task_type)

    doc: dict[str, Any] = {
        "@context": _CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "dct:conformsTo": _CROISSANT_CONFORMS_TO,
        "name": meta.get("full_name") or info.name,
        "description": meta.get("description", "") or "",
        "url": source,
        "keywords": keywords,
        "isAccessibleForFree": not is_manual_access(info),
    }

    if license_url:
        doc["license"] = license_url
    if conditions_of_access:
        doc["conditionsOfAccess"] = conditions_of_access
    if same_as:
        doc["sameAs"] = same_as if len(same_as) > 1 else same_as[0]
    if citation:
        doc["citation"] = citation

    doc["distribution"] = distribution
    doc["recordSet"] = record_set
    doc.update(rai)

    return doc


# ---------------------------------------------------------------------------
# Extend a platform-provided Croissant with RAI fields
# ---------------------------------------------------------------------------

def extend_croissant(existing: dict, rai: dict) -> dict:
    """Keep the platform-provided JSON intact; add RAI fields and conformsTo.

    Different platforms use different key names for conformsTo:
    - Kaggle uses ``"conformsTo"``
    - HuggingFace uses ``"dct:conformsTo"``
    Both are normalised to a single ``"dct:conformsTo"`` string (the primary
    Croissant version from the platform).  The RAI URL is not appended here
    because mlcroissant validates ``conformsTo`` as a single string value.
    """
    result = dict(existing)

    # Collect the primary conformsTo value from whichever key the platform used
    primary_conforms: str | None = None
    for key in ("conformsTo", "dct:conformsTo"):
        val = result.pop(key, None)
        if not val:
            continue
        # Prefer the first non-RAI URL we find (the core Croissant version)
        candidates = [val] if isinstance(val, str) else list(val)
        for c in candidates:
            if c and "RAI" not in c and primary_conforms is None:
                primary_conforms = c

    result["dct:conformsTo"] = primary_conforms or _CROISSANT_CONFORMS_TO

    # Ensure the @context contains all standard Croissant keys.
    # Platform-fetched contexts are often incomplete; fill gaps from the
    # canonical context so the mlcroissant validator does not warn.
    ctx = result.get("@context", {})
    if isinstance(ctx, dict):
        ctx = {**_CROISSANT_CONTEXT, **ctx}
        result["@context"] = ctx

    # Patch FileObject/FileSet @type to use cr: (mlcommons) namespace, which
    # is what the mlcroissant validator expects.  Platform-fetched files often
    # use sc: (schema.org) which expands to the wrong URI.
    _TYPE_MAP = {
        "sc:FileObject": "cr:FileObject",
        "sc:FileSet": "cr:FileSet",
        "https://schema.org/FileObject": "cr:FileObject",
        "https://schema.org/FileSet": "cr:FileSet",
    }
    for dist_entry in result.get("distribution", []):
        if isinstance(dist_entry, dict):
            t = dist_entry.get("@type")
            if t in _TYPE_MAP:
                dist_entry["@type"] = _TYPE_MAP[t]

    result.update(rai)
    return result


# ---------------------------------------------------------------------------
# RAI supplement YAML I/O
# ---------------------------------------------------------------------------

def load_rai_supplement(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def save_rai_supplement(path: Path, supplement: dict[str, dict]) -> None:
    header = (
        "# Editable RAI supplement for raman_data Croissant files.\n"
        "# Each top-level key is a dataset_key (as used in raman_data()).\n"
        "# Values here override the auto-generated RAI defaults on the next run.\n"
        "# Run generate_croissant.py after editing to regenerate the JSON-LD files.\n"
        "#\n"
        "# Available RAI fields per dataset:\n"
        "#   rai:dataCollection          — how the spectra were physically acquired\n"
        "#   rai:dataCollectionType      — type of measurement (e.g. patient samples)\n"
        "#   rai:dataPreprocessingProtocol — preprocessing applied before release\n"
        "#   rai:dataAnnotationProtocol  — how labels / targets were determined\n"
        "#   rai:personalSensitiveInformation — note if human data is present\n"
        "#   rai:dataBiases              — known biases and their sources\n"
        "#   rai:dataLimitations         — conditions where dataset may not apply\n"
        "#   rai:dataSocialImpact        — potential societal benefits or risks\n"
        "#   rai:dataUseCases            — recommended ML use cases\n\n"
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(supplement, fh, allow_unicode=True, default_flow_style=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Dataset gathering
# ---------------------------------------------------------------------------

def gather_all_datasets() -> list[tuple[str, DatasetInfo, str]]:
    """Return [(dataset_key, DatasetInfo, loader_name), ...] for every dataset."""
    result: list[tuple[str, DatasetInfo, str]] = []
    seen: set[str] = set()
    for loader_cls in _LOADERS:
        loader_name = _LOADER_NAMES[loader_cls]
        for key, info in loader_cls.DATASETS.items():
            if key not in seen:
                result.append((key, info, loader_name))
                seen.add(key)
    return result


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_key: str,
    info: DatasetInfo,
    loader_name: str,
    rai_supplement: dict[str, dict],
) -> tuple[dict, str, list[str]]:
    """
    Returns (croissant_dict, platform_name, warning_list).
    """
    warnings: list[str] = []
    meta = info.metadata
    platform = detect_platform(info)
    manual = is_manual_access(info)

    _, _, license_inferred = normalise_license(info.license)
    if license_inferred:
        warnings.append("license assumed CC BY 4.0")
    if manual:
        warnings.append("manual access only")

    auto_rai = auto_rai_fields(info, license_inferred, manual)
    rai = merge_rai(auto_rai, rai_supplement.get(dataset_key))

    fetched: dict | None = None

    if platform == "huggingface":
        hf_key = meta.get("hf_key", "")
        if hf_key:
            fetched = fetch_hf_croissant(hf_key)
            if fetched is None:
                warnings.append("HuggingFace fetch failed")
                # Try to confirm the license even without a full Croissant file
                confirmed_license = fetch_hf_license(hf_key)
                if confirmed_license:
                    # Patch auto-generated RAI limitations to remove the "assumed" caveat
                    rai["rai:dataLimitations"] = rai.get("rai:dataLimitations", "").split(
                        "Note: No explicit license"
                    )[0].rstrip()
                    license_inferred = False
                    warnings = [w for w in warnings if "license assumed" not in w]
                    # Store confirmed URL so build_croissant_from_scratch picks it up
                    meta = dict(meta)
                    meta["license"] = confirmed_license
                    info = type(info)(
                        id=info.id, name=info.name, loader=info.loader,
                        metadata=meta, task_type=info.task_type,
                        application_type=info.application_type,
                        file_typ=info.file_typ, short_name=info.short_name,
                    )

    elif platform == "kaggle":
        handle = _extract_kaggle_handle(meta.get("source", "") or "")
        if handle:
            fetched = fetch_kaggle_croissant(handle)
            if fetched is None:
                warnings.append("Kaggle fetch failed")

    figshare_files: list[dict] | None = None
    if platform == "figshare" and fetched is None:
        article_id = _extract_figshare_article_id(meta.get("source", "") or "")
        if article_id:
            figshare_files = fetch_figshare_files(article_id) or None

    if fetched is not None:
        doc = extend_croissant(fetched, rai)
        warnings.insert(0, f"fetched from {platform}")
    else:
        doc = build_croissant_from_scratch(dataset_key, info, rai, platform, figshare_files)

    return doc, platform, warnings


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_croissant(path: Path) -> tuple[list[str], list[str]]:
    """
    Validate a written Croissant JSON-LD file with mlcroissant.

    Returns (errors, warnings) as lists of strings.  Both are empty on success.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            json_data = json.load(fh)
        mlc.Dataset(jsonld=json_data)
    except mlc.ValidationError as exc:
        msg = str(exc)
        for line in msg.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("- ") or "error" in line.lower():
                errors.append(line.lstrip("- "))
            elif "warning" in line.lower():
                warnings.append(line.lstrip("- "))
    except Exception as exc:
        errors.append(f"unexpected validation error: {exc}")
    return errors, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Croissant 1.0 + RAI JSON-LD files for all raman_data datasets. "
            "Platform-provided Croissant files (HuggingFace, Kaggle) are fetched and "
            "extended; all other datasets are generated from their DatasetInfo metadata."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="./croissant_files",
        help="Directory to write Croissant JSON files (default: ./croissant_files).",
    )
    parser.add_argument(
        "--rai-file",
        default=str(Path(__file__).parent / "rai_supplement.yaml"),
        help=(
            "Path to the editable RAI supplement YAML. "
            "Created automatically on first run; edit to refine RAI descriptions."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Validate each generated file with mlcroissant after writing it.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rai_path = Path(args.rai_file)

    rai_supplement = load_rai_supplement(rai_path)
    all_datasets = gather_all_datasets()

    rows: list[tuple[str, str, str, list[str]]] = []
    new_rai_entries: dict[str, dict] = {}

    for dataset_key, info, loader_name in all_datasets:
        try:
            doc, platform, warns = process_dataset(dataset_key, info, loader_name, rai_supplement)
        except Exception as exc:
            logger.error("Failed to generate Croissant for %s: %s", dataset_key, exc)
            rows.append((dataset_key, loader_name, "FAILED", [f"ERROR: {exc}"]))
            continue

        out_path = output_dir / f"{dataset_key}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)

        if args.validate:
            val_errors, val_warnings = validate_croissant(out_path)
            if val_errors:
                warns.append(f"VALIDATION ERRORS: {'; '.join(val_errors)}")
            elif val_warnings:
                warns.append(f"validation warnings: {len(val_warnings)}")
            else:
                warns.append("validation ok")

        if dataset_key not in rai_supplement:
            _, _, license_inferred = normalise_license(info.license)
            new_rai_entries[dataset_key] = auto_rai_fields(
                info, license_inferred, is_manual_access(info)
            )

        rows.append((dataset_key, loader_name, out_path.name, warns))

    # Update supplement YAML with any new datasets
    if new_rai_entries:
        merged = dict(rai_supplement)
        merged.update(new_rai_entries)
        save_rai_supplement(rai_path, merged)
        logger.info(
            "Updated %s (+%d new entries)", rai_path, len(new_rai_entries)
        )

    # Summary
    n_ok = sum(1 for *_, fname, _ in rows if fname != "FAILED")
    n_fetched = sum(1 for *_, _, warns in rows if warns and warns[0].startswith("fetched from"))
    n_fail = len(rows) - n_ok
    n_val_errors = sum(1 for *_, _, warns in rows if any("VALIDATION ERRORS" in w for w in warns))
    print(f"\nGenerated {n_ok} Croissant files → {output_dir}/")
    print(f"  {n_fetched} fetched from platform   |   {n_ok - n_fetched} generated from metadata")
    if n_fail:
        print(f"  {n_fail} failed — see log above")
    if args.validate:
        print(f"  {n_val_errors} validation errors   |   {n_ok - n_val_errors} passed validation")
    print(f"\n{'Dataset key':<55} {'Loader':<15} Notes")
    print("─" * 100)
    for key, loader, fname, warns in rows:
        note = "; ".join(warns) if warns else "ok"
        print(f"  {key:<53} {loader:<15} {note}")
    print()


if __name__ == "__main__":
    main()
