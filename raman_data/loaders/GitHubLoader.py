import logging
import os
import re
from typing import Optional, List

import numpy as np
import pandas as pd

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import is_wavenumber, encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE, HASH_TYPE


class GitHubLoader(BaseLoader):
    """
    Loader for Raman spectroscopy datasets hosted on GitHub repositories.

    Downloads datasets from GitHub repo archives.
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "github")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.GitHub)

    DATASETS = {
        "covid19_salvia": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="covid19_salvia",
            name="Saliva COVID-19",
            short_name="Saliva COVID-19",
            license="Authors contacted",
            loader=lambda cache_path: GitHubLoader._load_mind_dataset(cache_path, "covid_dataset", ["CTRL", "COV+", "COV-"]),
            metadata={
                "full_name": "Saliva COVID-19 Raman Dataset",
                "source": "https://github.com/piazzam/Robust-SVM-Raman",
                "description": "Curated for non-invasive SARS-CoV-2 screening. Includes ~25 spectral replicates per subject from 101 patients (positive, negative symptomatic, and healthy controls) acquired from dried saliva drops using a 785 nm spectrometer.",
                "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                "bibtex": "@article{Bertazioli_2024, title={An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples}, volume={171}, ISSN={0010-4825}, url={http://dx.doi.org/10.1016/j.compbiomed.2024.108028}, DOI={10.1016/j.compbiomed.2024.108028}, journal={Computers in Biology and Medicine}, publisher={Elsevier BV}, author={Bertazioli, Dario and Piazza, Marco and Carlomagno, Cristiano and Gualerzi, Alice and Bedoni, Marzia and Messina, Enza}, year={2024}, month=mar, pages={108028}}",
                "citation": [
                    "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                ],
            },
            # Explicit group ids: one directory per subject in the source
            # repository (~25 replicate spectra per subject, 101 subjects).
            is_grouped=True,
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        **{
            f"{disease.lower()}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.Medical,
                id=f"{disease.lower()}",
                name=f"Saliva {disease}",
                short_name=f"Saliva {disease}",
                license="Authors contacted",
                loader=lambda cache_path, c=disease[0]: GitHubLoader._load_mind_dataset(cache_path, "pd_ad_dataset", [f"{c}D", "CTRL"]),
                metadata={
                    "full_name": f"Saliva Neurodegenerative Disease Raman Dataset ({disease})",
                    "source": "https://github.com/piazzam/Robust-SVM-Raman",
                    "description": f"Raman spectra from dried saliva drops targeting {disease}'s Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment.",
                    "paper": "https://doi.org/10.1016/j.compbiomed.2024.108028",
                    "bibtex": "@article{Bertazioli_2024, title={An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples}, volume={171}, ISSN={0010-4825}, url={http://dx.doi.org/10.1016/j.compbiomed.2024.108028}, DOI={10.1016/j.compbiomed.2024.108028}, journal={Computers in Biology and Medicine}, publisher={Elsevier BV}, author={Bertazioli, Dario and Piazza, Marco and Carlomagno, Cristiano and Gualerzi, Alice and Bedoni, Marzia and Messina, Enza}, year={2024}, month=mar, pages={108028}}",
                    "citation": [
                        "Bertazioli, D., Piazza, M., Carlomagno, C., Gualerzi, A., Bedoni, M. and Messina, E., 2024. An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples. Computers in Biology and Medicine, 171, p.108028."
                    ],
                },
                # Explicit group ids: one directory per subject in the source
                # repository. Replicate counts vary (16-50 spectra per subject).
                is_grouped=True,
                # Checked: no missing (NaN) label values for either disease.
                has_missing_labels=False,
            )
            for disease in ["Parkinson", "Alzheimer"]
        },
        "chlorinated_samples": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Chemical,
            id="chlorinated_samples",
            name="Chlorinated Sample Identification",
            short_name="Chlorinated Samples",
            license="Authors contacted (data provided by Analyze IQ Limited)",
            loader=lambda cache_path: GitHubLoader._load_chlorinated_samples(cache_path),
            is_grouped=False,
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
            metadata={
                "full_name": "Chlorinated Sample Identification (Raman)",
                "source": "https://github.com/AaronFlanagan20/Analysis-of-Data-Synthesis-for-Raman-Spectroscopy",
                "paper": "https://doi.org/10.1021/acs.jcim.3c00761",
                "bibtex": "@article{Flanagan_2023, title={A Comparative Analysis of Data Synthesis Techniques to Improve Classification Accuracy of Raman Spectroscopy Data}, ISSN={1549-960X}, url={http://dx.doi.org/10.1021/acs.jcim.3c00761}, DOI={10.1021/acs.jcim.3c00761}, journal={Journal of Chemical Information and Modeling}, publisher={American Chemical Society (ACS)}, author={Flanagan, Aaron and Glavin, Frank}, year={2023}}",
                "citation": [
                    "Flanagan, A. and Glavin, F., 2023. A Comparative Analysis of Data Synthesis Techniques to Improve Classification Accuracy of Raman Spectroscopy Data. Journal of Chemical Information and Modeling."
                ],
                "description": (
                    "Binary Raman classification task: detect the presence of chloroform in a "
                    "sample. 230 spectra across 2473 wavenumbers (350–3500 cm⁻¹). Class balance "
                    "{0: 76, 1: 154}. Data provided by Analyze IQ Limited; predefined 3-fold "
                    "splits ship with the source repository."
                ),
            },
        ),
        "biomolecules_reference": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="biomolecules_reference",
            name="Biomolecules",
            short_name="Biomolecules Ref.",
            license="GPL-3.0",
            loader=lambda cache_path: GitHubLoader._load_ramanbiolib(cache_path),
            metadata={
                "full_name": "RamanBioLib — Reference Biomolecules",
                "source": "https://github.com/mteranm/ramanbiolib",
                "paper": "https://doi.org/10.1016/j.chemolab.2025.105476",
                "bibtex": "@article{TERAN2025105476, title = {Open Raman spectral library for biomolecule identification}, journal = {Chemometrics and Intelligent Laboratory Systems}, volume = {264}, pages = {105476}, year = {2025}, issn = {0169-7439}, doi = {https://doi.org/10.1016/j.chemolab.2025.105476}, url = {https://www.sciencedirect.com/science/article/pii/S0169743925001613}, author = {Marcelo Terán and José Javier Ruiz and Pablo Loza-Alvarez and David Masip and David Merino}, keywords = {Raman spectroscopy, Spectral library, Biomolecules, Biomedicine, Database, Open-source}}",
                "description": (
                    "Reference Raman spectra (450–1800 cm⁻¹, 1 cm⁻¹ resolution) of ~140 pure biomolecules "
                    "including amino acids, nucleotides, lipids, and sugars. Each spectrum is labelled by "
                    "biomolecule name. Useful for spectral assignment and as a reference library for "
                    "classification benchmarks."
                ),
            },
        ),
        # ------------------------------------------------------------------
        # PROVENANCE / ETHICS CAVEAT -- read before using or re-mirroring
        # this dataset. Summarized here; full detail in
        # `_load_ait_glucose_blood_sers`'s docstring below.
        #
        # This dataset does NOT meet raman_data's own stated inclusion
        # criterion of "accompanied by a citable reference (paper, report,
        # or dataset DOI)" -- no such reference exists for the source repo.
        # No informed-consent, IRB, or ethics-review documentation was found
        # anywhere associated with this data (repo README, full wiki
        # content, predecessor repo, or lab project pages) as of integration
        # (2026-08-21), despite this being real human blood/glucose data.
        # The source repo's own predecessor states no journal paper was
        # ever produced from this work.
        #
        # Included at the explicit direction of the RamanBench maintainer
        # (Mario Koddenbrock), who intends to follow up directly with the
        # originating lab (AIT Brain Lab / AIT-brainlab, Asian Institute of
        # Technology, Thailand) regarding consent/ethics documentation. Not
        # a routine inclusion -- do not use this as a template for skipping
        # the citable-reference or consent checks on future datasets.
        # ------------------------------------------------------------------
        "ait_glucose_blood_sers": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Medical,
            id="ait_glucose_blood_sers",
            name="AIT Blood Glucose (SERS)",
            short_name="AIT Blood SERS",
            license=(
                "Unclear. Repo root LICENSE is MIT, covering \"the Software\" "
                "(code) -- no separate license or reuse terms were found for "
                "the data files themselves. See metadata['description'] for "
                "the full provenance/consent caveat."
            ),
            loader=lambda cache_path: GitHubLoader._load_ait_glucose_blood_sers(cache_path),
            metadata={
                "full_name": "AIT-brainlab Blood-SERS Glucose Calibration Dataset",
                "source": "https://github.com/AIT-brainlab/raman-for-glucose-measurement",
                "paper": None,  # No paper, report, or DOI exists for this source (see caveat).
                "description": (
                    "CAVEAT (read first): No informed consent, IRB approval, or ethics "
                    "review documentation was found associated with this data as of "
                    "integration (2026-08-21); the data appears to be informal/unpublished "
                    "student research (the source repo's predecessor explicitly states no "
                    "journal paper was produced). There is no paper, report, or DOI to cite "
                    "for this dataset, which does not meet raman_data's own stated "
                    "citable-reference inclusion criterion. Included at the explicit "
                    "direction of the RamanBench maintainer, who intends to follow up "
                    "directly with the originating lab. A paper citing data matching this "
                    "description (arXiv:2608.14227, 'Attributing Preprocessing Invariance "
                    "in Spectral Foundation Models') attributes it to 'AIT brainlab and "
                    "MIT' -- the 'MIT' half of that attribution could not be verified and "
                    "is likely erroneous: the source repo's actual copyright holder is "
                    "'Future Lab' (a named AIT x BUPT joint facility, Asian Institute of "
                    "Technology x Beijing University of Posts and Telecommunications), "
                    "not Massachusetts Institute of Technology, and the repo has a single "
                    "GitHub contributor with no visible MIT (Massachusetts) collaboration."
                    "\n\n"
                    "DATA: SERS spectra of blood samples spiked with 6 distinct known "
                    "glucose concentrations (88, 95, 98, 121, 166, 168 -- units not "
                    "confirmed by any source metadata; inferred to be mg/dL from research "
                    "context, not verified). 785 nm excitation, 5x lens, 60 s exposure. "
                    "Each spectrum has 1999 points spanning roughly -1392 to 2746 cm⁻¹ "
                    "(includes an uncalibrated negative-shift region; no cropping applied "
                    "here). One same-folder 'foil_...' reference/blank measurement is "
                    "excluded (no concentration label). This is a best-effort, independently "
                    "chosen subset of the source repo -- NOT a reproduction of the citing "
                    "paper's reported 435-sample count, which could not be reconstructed "
                    "from the repo's current structure or any available methodology "
                    "description; 'skin' and 'pilot' (OGTT) subsets of the same repo were "
                    "evaluated but not included here (skin has no discoverable label; pilot "
                    "was out of scope for this pass). Target: glucose_concentration "
                    "(regression). Grouped by concentration level: all replicate readings "
                    "of a given spiked concentration were acquired within a single "
                    "contiguous session (same spiked blood aliquot), so group-aware "
                    "splitting is used to avoid replicate leakage across train/test -- "
                    "note this also means only 6 distinct groups exist, an unusually small "
                    "and near-degenerate group count for benchmark splitting."
                ),
            },
            # Checked: replicate spectra from the same spiked-concentration
            # session share a group id (see loader docstring) -- real
            # metadata-derived grouping, not inferred from target equality.
            is_grouped=True,
            # Checked: every file with a `blood-<value>` filename prefix carries
            # a concentration value; none missing.
            has_missing_labels=False,
        ),
        "liver_cancer_raman": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="liver_cancer_raman",
            name="Liver Cancer vs Adjacent Tissue (Huang 2023 demo)",
            short_name="Liver Cancer Raman",
            license="MIT",
            loader=lambda cache_path: GitHubLoader._load_liver_cancer_raman(
                cache_path
            ),
            metadata={
                "full_name": (
                    "Raman spectra of hepatic carcinoma and adjacent "
                    "non-tumour liver tissue (Huang et al. 2023, "
                    "Nat. Commun. 14:48; the authors' GitHub demo data)"
                ),
                "source": "https://github.com/AzyroSH/CNN_liver-cancer_Raman",
                "paper": "https://doi.org/10.1038/s41467-022-35696-2",
                "bibtex": "@article{Huang_2023, title={Rapid, label-free histopathological diagnosis of liver cancer based on Raman spectroscopy and deep learning}, volume={14}, number={1}, pages={48}, ISSN={2041-1723}, url={http://dx.doi.org/10.1038/s41467-022-35696-2}, DOI={10.1038/s41467-022-35696-2}, journal={Nature Communications}, publisher={Springer Nature and its contributors}, author={Huang, Liping and Sun, Hongwei and Sun, Liangbin and Shi, Keqing and Chen, Yuzhe and Ren, Xueqian and Ge, Yuancai and Jiang, Danfeng and Liu, Xiaohu and Knoll, Wolfgang and Zhang, Qingwen and Wang, Yi}, year={2023}, month=jan}",
                "citation": [
                    "Huang, L., Sun, H., Sun, L., Shi, K., Chen, Y., Ren, X., "
                    "Ge, Y., Jiang, D., Liu, X., Knoll, W., Zhang, Q., and "
                    "Wang, Y. (2023). Rapid, label-free histopathological "
                    "diagnosis of liver cancer based on Raman spectroscopy "
                    "and deep learning. Nature Communications, 14(1), 48. "
                    "https://doi.org/10.1038/s41467-022-35696-2",
                ],
                "description": (
                    "Binary classification of hepatic carcinoma vs adjacent "
                    "non-tumour (paracarcinoma) liver tissue from the demo "
                    "data of Huang et al. 2023 (Nat. Commun. 14:48, "
                    "doi:10.1038/s41467-022-35696-2). The paper's full "
                    "database -- 120 pairs (240 tissue samples) from 120 "
                    "liver-cancer patients (98 HCC + 22 ICC), 50 spectra "
                    "per sample, 12,000 spectra total -- was not released; "
                    "the authors' MIT-licensed GitHub repo "
                    "(AzyroSH/CNN_liver-cancer_Raman, pinned at commit "
                    "fdb3743) ships a 6,000-spectrum demo split into 4,000 "
                    "train / 1,000 validate / 1,000 test files, balanced "
                    "3,000 carcinoma (label 1) / 3,000 paracarcinoma "
                    "(label 0). Confocal micro-Raman with 532 nm "
                    "excitation (Renishaw), 500-2,000 cm^-1, 889 points "
                    "(the paper's canonical grid); the train/validate "
                    "demo files carry 892 data columns and are resampled "
                    "onto the 889-point grid (the 3 extra points sit at "
                    "the high-wavenumber end; alignment verified by "
                    "mean-spectrum correlation). Files are label-first "
                    "tab-separated text; intensities are raw as deposited "
                    "(arbitrary units, negative dark-subtracted points, "
                    "maxima ~6.2e3) and the authors' z-score "
                    "normalization is not applied. group_ids: the "
                    "train/validate files are ordered in label-pure "
                    "50-spectrum blocks (one tissue sample each; verified "
                    "against shuffled controls), giving 80 + 20 sample "
                    "groups; the authors shuffled the test file, so its "
                    "1,000 rows are exposed as singleton groups (no "
                    "sample ids are in the deposit) -- 1,100 groups in "
                    "total. The authors report ~90% accuracy with their "
                    "1D VGG-16 CNN on this split."
                ),
            },
            # Checked: the train/validate files are ordered in
            # label-pure 50-spectrum sample blocks (within-block
            # variance 13-17% below shuffled controls); the test file
            # is shuffled, so its rows get singleton groups.
            is_grouped=True,
            # Checked: every row of every demo file carries a 0/1
            # label in the first column; no missing values.
            has_missing_labels=False,
        ),
        "sers_gna_bacteria": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="sers_gna_bacteria",
            name="SERS of Nine Bacterial Strains on a Gold Nanorod Array (Gulmus 2026)",
            short_name="SERS GNA Bacteria",
            license="CC BY 4.0",
            loader=lambda cache_path: GitHubLoader._load_sers_gna_bacteria(
                cache_path
            ),
            metadata={
                "full_name": (
                    "SERS spectra of nine bacterial strains (E. coli, "
                    "P. aeruginosa, S. aureus) on an anisotropic gold "
                    "nanorod array (Gulmus et al. 2026)"
                ),
                "source": "https://github.com/erkankaratas-eng/sers-gna-bacteria",
                "paper": "https://doi.org/10.5281/zenodo.23426967",
                "bibtex": "@dataset{Gulmus_2026_sers_gna, title={SERS spectra of nine bacterial strains on a gold nanorod array - data and analysis code}, author={Can Gulmus, Feyzanur and Oncer, Nazli and Karatas, Erkan and Gormez, Arzu and Yilmaz, Asli and Yilmaz, Mehmet}, year={2026}, publisher={Zenodo}, doi={10.5281/zenodo.23426967}, url={https://doi.org/10.5281/zenodo.23426967}}",
                "citation": [
                    "Gulmus, F. C., Oncer, N., Karatas, E., Gormez, A., "
                    "Yilmaz, A., and Yilmaz, M. (2026). SERS spectra of "
                    "nine bacterial strains on a gold nanorod array - "
                    "data and analysis code (v1.0.1). Zenodo. "
                    "https://doi.org/10.5281/zenodo.23426967",
                ],
                "description": (
                    "117 SERS spectra of nine bacterial strains (3 each "
                    "of E. coli 29922/K4/BAA-2523, P. aeruginosa "
                    "9027/10145/27853 and S. aureus 6538/25923/43300) "
                    "on an anisotropic gold nanorod array (Gulmus et "
                    "al. 2026; the associated article is under review, "
                    "so the Zenodo record v1.0.1, "
                    "doi:10.5281/zenodo.23426967, is the citation; the "
                    "repo's earlier v1.0.0 record is "
                    "doi:10.5281/zenodo.23284998). WITec alpha 300R "
                    "confocal Raman microscope, 785 nm excitation, x50 "
                    "objective, 10 mW, ~3.03 s integration, 4 cm^-1 "
                    "spectral resolution. Each strain was cultivated "
                    "once and deposited as one 5 uL (0.5 McFarland) "
                    "droplet on a single GNA substrate; all 13 spectra "
                    "of a strain were recorded from that droplet on the "
                    "same day, so group_ids = strain (9 groups of 13) "
                    "and leave-strain-out is the paper's validation "
                    "protocol (it also analyses acquisition-order "
                    "trends within each strain). The loader reads the "
                    "raw WITec instrument text files (counts, before "
                    "any baseline correction, smoothing or "
                    "normalization): all 117 share one 1,024-point "
                    "247.99-1,997.9 cm^-1 axis (non-uniform, "
                    "1.44-2.04 cm^-1 spacing), used as is. The "
                    "deposit's preprocessed CSV (Data S1; OPUS 5.5 "
                    "rubberband + Savitzky-Golay + min-max "
                    "normalization, 1,215 points, with the S. aureus "
                    "spectra exported on a 0-1 scale and the other two "
                    "species on 0-2) is not loaded. Strain "
                    "classification is the primary task; the 3-species "
                    "task is derivable from the class-name prefixes. "
                    "The repo is served as a commit-pinned archive "
                    "(1028526, size + sha256 pinned; the Zenodo v1.0.1 "
                    "archive's data are byte-identical)."
                ),
            },
            # Checked: all 13 spectra of a strain come from one droplet
            # on one GNA substrate of one culture (same day), so the
            # strain is the finest correlated unit (9 groups).
            is_grouped=True,
            # Checked: all 117 rows of the Data S1 manifest join a raw
            # instrument file of the same strain; no missing values.
            has_missing_labels=False,
        ),
        "sers_pesticide_produce": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="sers_pesticide_produce",
            name="SERS Pesticide Mixtures in Fresh Produce (Hegde 2025)",
            short_name="SERS Pesticide Produce",
            license="GPL-3.0",
            loader=lambda cache_path: GitHubLoader._load_sers_pesticide_produce(
                cache_path
            ),
            metadata={
                "full_name": (
                    "SERS spectra of pesticide mixtures in spinach and "
                    "strawberry (Hegde et al. 2025, ACS Appl. Mater. "
                    "Interfaces; the SERSFormer2 repository)"
                ),
                "source": "https://github.com/jianlin-cheng/SERSFormer2",
                "paper": "https://doi.org/10.1021/acsami.4c17777",
                "bibtex": "@article{Hegde_2025, title={Leveraging SERS and Transformer Models for Simultaneous Detection of Multiple Pesticides in Fresh Produce}, volume={17}, number={1}, pages={2018--2031}, url={https://doi.org/10.1021/acsami.4c17777}, DOI={10.1021/acsami.4c17777}, journal={ACS Applied Materials \\& Interfaces}, author={Hegde, Akshata and Hajikhani, Mehdi and Snyder, John and Cheng, Jianlin and Lin, Mengshi}, year={2025}, month=jan}",
                "citation": [
                    "Hegde, A., Hajikhani, M., Snyder, J., Cheng, J., and "
                    "Lin, M. (2025). Leveraging SERS and transformer "
                    "models for simultaneous detection of multiple "
                    "pesticides in fresh produce. ACS Applied Materials "
                    "& Interfaces, 17(1), 2018-2031. "
                    "https://doi.org/10.1021/acsami.4c17777",
                ],
                "description": (
                    "33 SERS spectra of pesticide-spiked fresh produce "
                    "from the SERSFormer2 repository (Hegde et al. "
                    "2025, ACS Appl. Mater. Interfaces 17(1):2018-2031, "
                    "doi:10.1021/acsami.4c17777; the repo's README "
                    "cites 2024, the journal issue is 2025-01). The "
                    "public deposit is a pruned subset of the paper's "
                    "10+ mixture panel: a spinach control "
                    "(No_pest_present, 3 spectra), a spinach "
                    "carbophenothion+oxamyl 50:50 mixture (5 "
                    "concentration levels x 3) and a strawberry "
                    "carbophenothion+phosmet 50:50 mixture (5 levels "
                    "x 3). Multi-output regression: the per-pesticide "
                    "concentration in ppm for the three pesticides "
                    "present in the deposit (carbophenothion, oxamyl, "
                    "phosmet); the 50:50 folders halve the nominal "
                    "level, so the per-pesticide levels are "
                    "0.25/0.5/1.0/2.5/5.0 ppm (0 = absent). The "
                    "paper's multi-label presence task is derivable "
                    "(a pesticide is present iff its concentration > "
                    "0; the control is the all-zero row). The other "
                    "two members of the paper's 5-pesticide panel "
                    "(coumaphos, thiabendazole) have no spectra in the "
                    "public deposit, so they are not shipped as "
                    "degenerate (constant-zero) targets. Native axes "
                    "are 1,556 points over ~500-2,000 cm^-1 with three "
                    "slightly different batch grids; the loader "
                    "resamples every spectrum onto the common "
                    "501-1,999 cm^-1 @1 cm^-1 grid (1,499 points), "
                    "which all 33 files fully cover (no "
                    "extrapolation). Intensities are raw as deposited "
                    "(arbitrary units, negative baseline-subtracted "
                    "points down to ~-17, maxima ~7.8e3); the "
                    "authors' log1p/threshold/Hann-convolution "
                    "(Dataset.py) is model-side preprocessing, not "
                    "applied here. group_ids = condition (produce x "
                    "mixture x concentration level, 11 groups of 3: "
                    "the three spectra of a condition are replicate "
                    "SERS measurements, so grouped CV is "
                    "leave-one-condition-out). The repository is "
                    "GPL-3.0 licensed (code and data shipped "
                    "together, no separate data licence stated); the "
                    "data are served as a commit-pinned archive "
                    "(bee22a7, size + sha256 pinned)."
                ),
            },
            # Checked: the three spectra of each (produce, mixture,
            # concentration) condition are replicate measurements of
            # the same spiked sample; condition is the correlated
            # unit (11 groups of 3).
            is_grouped=True,
            # Checked: every file carries a full target row (all three
            # pesticide concentrations defined, 0 for absent); no
            # missing values.
            has_missing_labels=False,
            embedded_preprocessing=(
                "None beyond resampling onto the common 501-1,999 "
                "cm^-1 @1 cm^-1 grid (1,499 points). The CSVs store "
                "raw instrument intensities (arbitrary units, "
                "negative baseline-subtracted points); the authors' "
                "log1p/thresholding/Hann-window convolution "
                "(Dataset.py) is model-side preprocessing and is not "
                "applied."
            ),
        ),
        "uraman_mofs": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="uraman_mofs",
            name="IBM uRaman MOF Reference Spectra (Ferreira 2023)",
            short_name="uRaman MOFs",
            license="CDLA-Sharing-1.0",
            loader=lambda cache_path: GitHubLoader._load_uraman_mofs(
                cache_path
            ),
            metadata={
                "full_name": (
                    "IBM uRaman-Dataset: peak-normalized Raman "
                    "reference spectra of three MOF compounds "
                    "(HKUST-1, Mg-MOF74, ZIF-8)"
                ),
                "source": "https://github.com/IBM/uRaman-Dataset",
                "bibtex": (
                    "@dataset{ferreira2023uraman, title={uRaman-"
                    "Dataset: Raman spectroscopy spectra of Metal "
                    "Organic Framework (MOF) compounds}, "
                    "author={Ferreira, Matheus Esteves}, "
                    "year={2023}, publisher={GitHub, IBM}, "
                    "url={https://github.com/IBM/uRaman-Dataset}, "
                    "note={commit "
                    "47e712d326fb0458ee8218ab73a2cfc269dcdfab, "
                    "license CDLA-Sharing-1.0}}"
                ),
                "citation": [
                    "Ferreira, M. E. (2023). uRaman-Dataset: Raman "
                    "spectroscopy spectra of Metal Organic Framework "
                    "(MOF) compounds (commit 47e712d3). GitHub "
                    "(IBM). https://github.com/IBM/uRaman-Dataset",
                ],
                "description": (
                    "Peak-normalized Raman reference spectra of the "
                    "three MOF compounds listed in the deposit's "
                    "README: HKUST-1 (Basolite C300, Sigma-Aldrich), "
                    "Mg-MOF74 and ZIF-8 (Basolite Z1200, "
                    "Sigma-Aldrich), measured with a WITec UHTS300S "
                    "Raman system (DU970_BVF detector, 63x "
                    "objective, 100 accumulations x 0.5 s). The "
                    "acquisition conditions differ per compound "
                    "(recorded in each folder's "
                    "Experiment_Information.txt): HKUST-1 "
                    "531.970 nm at 1 mW (2023-02-14), Mg-MOF74 "
                    "632.810 nm at 1 mW (2023-03-01), ZIF-8 "
                    "531.970 nm at 2 mW (2023-04-27); the "
                    "Mg-MOF74 export is labelled '(Sub BG) (Norm)' "
                    "(background-subtracted) and carries 989 "
                    "negative points. All three spectra are "
                    "peak-normalized (max = 1.0) as deposited, so "
                    "the relative peak structure is comparable "
                    "across the different excitation wavelengths "
                    "and powers. The two native grids (1,806 points "
                    "0-3,997.65 cm^-1 for HKUST-1/ZIF-8, 2,768 "
                    "points 0-3,998.38 cm^-1 for Mg-MOF74) are "
                    "resampled with linear interpolation onto the "
                    "common 0-3,997 cm^-1 @1 cm^-1 grid (3,998 "
                    "points). This is a reference set: one stitched "
                    "spectrum per class (3 classes), so no "
                    "cross-validation-based classification is "
                    "possible - the key is intended for spectral "
                    "matching, few-shot and augmentation use. "
                    "is_grouped=False (one spectrum per sample). "
                    "The repo is served as a commit-pinned codeload "
                    "archive (commit 47e712d326fb0458ee8218ab73a2c"
                    "fc269dcdfab, 2023-06-19, 819,336 bytes, size + "
                    "sha256 pinned); no indexed parent paper "
                    "exists, so the repository itself is the "
                    "citation (maintainer: Matheus Esteves "
                    "Ferreira, IBM)."
                ),
            },
            # Checked: one stitched spectrum per compound; no
            # replicate structure exists in the deposit.
            is_grouped=False,
            # Checked: every CSV in the pinned archive parses to a
            # finite, strictly increasing two-column spectrum; no
            # missing values.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Peak normalization to max = 1.0 as deposited "
                "(the Mg-MOF74 export is additionally "
                "background-subtracted, with 989 negative points); "
                "no further baseline correction or normalization is "
                "applied here. The two native grids are linearly "
                "interpolated onto the common 0-3,997 cm^-1 @1 "
                "cm^-1 grid (3,998 points)."
            ),
        ),
        "uraman_mof74_1000_2000": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="uraman_mof74_1000_2000",
            name=(
                "IBM uRaman MOF-74 Metal Series, 1,000-2,000 cm^-1 "
                "Window (Ferreira 2023)"
            ),
            short_name="uRaman MOF-74 1000-2000",
            license="CDLA-Sharing-1.0",
            loader=lambda cache_path: GitHubLoader._load_uraman_mof74_1000_2000(
                cache_path
            ),
            metadata={
                "full_name": (
                    "IBM uRaman-Dataset: peak-normalized Raman "
                    "spectra of six MOF-74 metal-series samples in "
                    "the nominal 1,000-2,000 cm^-1 window"
                ),
                "source": "https://github.com/IBM/uRaman-Dataset",
                "bibtex": (
                    "@dataset{ferreira2023uraman, title={uRaman-"
                    "Dataset: Raman spectroscopy spectra of Metal "
                    "Organic Framework (MOF) compounds}, "
                    "author={Ferreira, Matheus Esteves}, "
                    "year={2023}, publisher={GitHub, IBM}, "
                    "url={https://github.com/IBM/uRaman-Dataset}, "
                    "note={commit "
                    "47e712d326fb0458ee8218ab73a2cfc269dcdfab, "
                    "license CDLA-Sharing-1.0}}"
                ),
                "citation": [
                    "Ferreira, M. E. (2023). uRaman-Dataset: Raman "
                    "spectroscopy spectra of Metal Organic Framework "
                    "(MOF) compounds (commit 47e712d3). GitHub "
                    "(IBM). https://github.com/IBM/uRaman-Dataset",
                ],
                "description": (
                    "Peak-normalized Raman spectra of the six "
                    "MOF-74 metal-series samples of the deposit's "
                    "MeMOF-74 set ('Me' = metal: Mg, Zn and the "
                    "Zn75Mg25 alloy, plus the gold-coated Au@ "
                    "variants), measured in the nominal 1,000-2,000 "
                    "cm^-1 window with a WITec Alpha 300 RAS "
                    "micro-Raman system (532 nm at 1 mW, 1800 "
                    "g/mm grating, 200-400 accumulations x 0.5 s; "
                    "the samples were provided by Prof. Carlos "
                    "Basilio Pinheiro's group, per the deposit's "
                    "About.md). Five files span 969.006-1,980.60 "
                    "cm^-1 (1,600 points each); Au@Mg-MOF74 spans "
                    "860.797-1,887.76 cm^-1 (1,600 points), off the "
                    "nominal window, so the common window is the "
                    "intersection of the six native axes, resampled "
                    "with linear interpolation onto 970-1,887 cm^-1 "
                    "@1 cm^-1 (918 points). All six spectra are "
                    "peak-normalized (max = 1.0) as deposited, with "
                    "no negative points in this window. One "
                    "spectrum per class (6 classes): a reference "
                    "set, not CV-learnable. is_grouped=False. The "
                    "1,000-2,000 cm^-1 file of Au@Zn75Mg25-MOF74 is "
                    "spelled with a double dash in the deposit "
                    "('Au@Zn75Mg25--MOF74.csv'); its label is "
                    "normalized to a single dash. Served from the "
                    "same commit-pinned codeload archive as "
                    "uraman_mofs (commit "
                    "47e712d326fb0458ee8218ab73a2cfc269dcdfab, "
                    "819,336 bytes, size + sha256 pinned)."
                ),
            },
            # Checked: one stitched spectrum per sample; no
            # replicate structure exists in the deposit.
            is_grouped=False,
            # Checked: all six CSVs in the pinned archive parse to
            # finite, strictly increasing two-column spectra; no
            # missing values.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Peak normalization to max = 1.0 as deposited; no "
                "further preprocessing is applied here. The native "
                "1,600-point grids are linearly interpolated onto "
                "the common 970-1,887 cm^-1 @1 cm^-1 grid (918 "
                "points); each spectrum is truncated outside the "
                "common window (the five nominal-window files lose "
                "1,888-1,980.60 cm^-1, Au@Mg-MOF74 loses "
                "860.797-969 cm^-1)."
            ),
        ),
        "uraman_mof74_70_3000": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="uraman_mof74_70_3000",
            name=(
                "IBM uRaman MOF-74 Metal Series, 70-3,000 cm^-1 "
                "Stitched Window (Ferreira 2023)"
            ),
            short_name="uRaman MOF-74 70-3000",
            license="CDLA-Sharing-1.0",
            loader=lambda cache_path: GitHubLoader._load_uraman_mof74_70_3000(
                cache_path
            ),
            metadata={
                "full_name": (
                    "IBM uRaman-Dataset: peak-normalized stitched "
                    "Raman spectra of five MOF-74 metal-series "
                    "samples in the nominal 70-3,000 cm^-1 window"
                ),
                "source": "https://github.com/IBM/uRaman-Dataset",
                "bibtex": (
                    "@dataset{ferreira2023uraman, title={uRaman-"
                    "Dataset: Raman spectroscopy spectra of Metal "
                    "Organic Framework (MOF) compounds}, "
                    "author={Ferreira, Matheus Esteves}, "
                    "year={2023}, publisher={GitHub, IBM}, "
                    "url={https://github.com/IBM/uRaman-Dataset}, "
                    "note={commit "
                    "47e712d326fb0458ee8218ab73a2cfc269dcdfab, "
                    "license CDLA-Sharing-1.0}}"
                ),
                "citation": [
                    "Ferreira, M. E. (2023). uRaman-Dataset: Raman "
                    "spectroscopy spectra of Metal Organic Framework "
                    "(MOF) compounds (commit 47e712d3). GitHub "
                    "(IBM). https://github.com/IBM/uRaman-Dataset",
                ],
                "description": (
                    "Peak-normalized stitched Raman spectra of the "
                    "five MOF-74 metal-series samples available in "
                    "the deposit's 70-3,000 cm^-1 window (nominal; "
                    "the native axis is 0-2,999.35 cm^-1, 5,266 "
                    "points), measured with a WITec Alpha 300 RAS "
                    "micro-Raman system (532 nm at 1 mW, 1800 "
                    "g/mm grating, 100 accumulations x 0.5 s; the "
                    "samples were provided by Prof. Carlos Basilio "
                    "Pinheiro's group, per the deposit's About.md). "
                    "Au@Mg-MOF74 has no stitched spectrum - the "
                    "deposit's About.md states 'No spectral "
                    "stitching is available for Au@Mg-MOF74 due to "
                    "a high luminescence background'. The native "
                    "grids are resampled with linear interpolation "
                    "onto the common 0-2,999 cm^-1 @1 cm^-1 grid "
                    "(3,000 points). All five spectra are "
                    "peak-normalized (max = 1.0) as deposited, with "
                    "no negative points in this window. One "
                    "spectrum per class (5 classes): a reference "
                    "set, not CV-learnable. is_grouped=False. "
                    "Served from the same commit-pinned codeload "
                    "archive as uraman_mofs (commit "
                    "47e712d326fb0458ee8218ab73a2cfc269dcdfab, "
                    "819,336 bytes, size + sha256 pinned)."
                ),
            },
            # Checked: one stitched spectrum per sample; no
            # replicate structure exists in the deposit.
            is_grouped=False,
            # Checked: all five CSVs in the pinned archive parse to
            # finite, strictly increasing two-column spectra; no
            # missing values.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Peak normalization to max = 1.0 as deposited; no "
                "further preprocessing is applied here. The native "
                "5,266-point 0-2,999.35 cm^-1 grids are linearly "
                "interpolated onto the common 0-2,999 cm^-1 @1 "
                "cm^-1 grid (3,000 points)."
            ),
        ),
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from GitHub loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        if not LoaderTools.is_dataset_available(dataset_name, GitHubLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.GitHub)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.GitHub)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        GitHubLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        dataset_info = GitHubLoader.DATASETS[dataset_name]

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
    def _load_mind_dataset(cache_path: str, dataset_subfolder: str, category_filter: List[str]):
        """
        Load MIND-Lab datasets (covid_dataset or pd_ad_dataset).

        The expected layout (inside dataset folder):
          <patient_id>/spectra.csv
                          /raman_shift.csv
                          /user_information.csv

        Returns: spectra, raman_shifts, targets, class_names, group_ids
        """
        shared_root = os.path.join(os.path.dirname(cache_path), "mind_shared")
        shared_main = os.path.join(shared_root, "Raman-Spectra-Data-main")
        if os.path.isdir(shared_main) and os.listdir(shared_main):
            GitHubLoader.logger.debug(f"Using existing dataset folder at {shared_main}")
        else:
            zip_name = "Raman-Spectra-Data.zip"
            zip_file = os.path.join(shared_root, zip_name)

            if not os.path.exists(shared_root):
                GitHubLoader.logger.debug(f"Attempting to download dataset {dataset_subfolder} to {shared_root}")
                os.makedirs(shared_root, exist_ok=True)

                primary_url = "https://github.com/MIND-Lab/Raman-Spectra-Data/archive/refs/heads/main.zip"
                # Fallback: GitHub Release asset on ml-lab-htw/rb_data_fallback.
                # Release assets are not LFS-gated, unlike `archive/refs/heads/main.zip`,
                # which only ships LFS pointer stubs for LFS-tracked CSVs.
                # The asset must be named Raman-Spectra-Data.zip and extract to "Raman-Spectra-Data-main/".
                fallback_url = "https://github.com/ml-lab-htw/rb_data_fallback/releases/latest/download/Raman-Spectra-Data.zip"

                if not os.path.exists(zip_file):
                    try:
                        LoaderTools.download(
                            url=primary_url,
                            out_dir_path=shared_root,
                            out_file_name=zip_name,
                        )
                    except Exception as e:
                        GitHubLoader.logger.warning(
                            f"Primary MIND-Lab repo unreachable ({e}); falling back to {fallback_url}"
                        )
                        LoaderTools.download(
                            url=fallback_url,
                            out_dir_path=shared_root,
                            out_file_name=zip_name,
                        )

                LoaderTools.extract_zip_file_content(zip_file)

        # Iterate patient folders.
        #
        # Spectra, labels and group ids are collected in a single pass. The
        # directory name is the subject id, and every spectrum under it is a
        # replicate of the same subject -- replicate counts vary (16-50), so
        # the grouping cannot be reconstructed downstream from row counts.
        spectra_list = []
        raman_shifts_list = []
        categories = []          # one entry per spectrum
        group_list = []          # one entry per spectrum
        next_group_id = 0

        dataset_dir = os.path.join(shared_main, dataset_subfolder)
        for entry in sorted(os.listdir(dataset_dir)):
            patient_dir = os.path.join(dataset_dir, entry)
            if not os.path.isdir(patient_dir):
                continue

            user_info_path = os.path.join(patient_dir, "user_information.csv")
            spectra_path = os.path.join(patient_dir, "spectra.csv")
            shifts_path = os.path.join(patient_dir, "raman_shift.csv")

            if not (os.path.exists(user_info_path) and os.path.exists(spectra_path) and os.path.exists(shifts_path)):
                GitHubLoader.logger.warning(f"[!] Skipping patient folder (missing files): {patient_dir}")
                continue

            try:
                ui = pd.read_csv(user_info_path)
            except Exception as e:
                GitHubLoader.logger.warning(f"[!] Failed to read user_information.csv for {patient_dir}: {e}")
                continue

            cat_col = next((c for c in ui.columns if c.lower() == "category"), None)
            if cat_col is None and len(ui.columns) >= 2:
                cat_col = ui.columns[1]
            if cat_col is None:
                cat_col = next((c for c in ui.columns if c.lower() == "label"), None)
            if cat_col is None:
                GitHubLoader.logger.warning(f"[!] No category/label column found in {user_info_path}; skipping")
                continue

            category = str(ui[cat_col].iloc[0])

            if category not in category_filter:
                continue

            try:
                spectra_df = pd.read_csv(spectra_path, header=None)
                shifts = pd.read_csv(shifts_path, header=None).to_numpy().squeeze()
            except Exception as e:
                GitHubLoader.logger.warning(f"[!] Failed to read spectra/shift for {patient_dir}: {e}")
                continue

            for _, row in spectra_df.iterrows():
                row_arr = row.to_numpy(dtype=float)
                spectra_list.append(row_arr)
                raman_shifts_list.append(shifts)
                categories.append(category)
                group_list.append(next_group_id)
            next_group_id += 1

        if len(spectra_list) == 0:
            raise Exception(f"[!] No spectra found in {dataset_dir}")

        unique_categories = sorted(set(categories))
        cat_to_idx = {lab: i for i, lab in enumerate(unique_categories)}
        targets = np.array([cat_to_idx[c] for c in categories], dtype=int)
        group_ids = np.array(group_list, dtype=int)

        first_rs = None
        if len(raman_shifts_list) > 0:
            try:
                first_rs = raman_shifts_list[0]
                all_equal = all(np.allclose(first_rs, rs) for rs in raman_shifts_list)
            except Exception:
                all_equal = False
        else:
            all_equal = False

        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra = np.stack([np.array(s, dtype=float) for s in spectra_list])
        else:
            # raman_shifts = [np.array(rs, dtype=float) for rs in raman_shifts_list]
            # spectra = [np.array(s, dtype=float) for s in spectra_list]
            raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)

        class_names = unique_categories

        return spectra, raman_shifts, targets, class_names, group_ids

    @staticmethod
    def _load_chlorinated_samples(cache_path: str):
        """Load the chlorinated-sample identification dataset.

        Source: github.com/AaronFlanagan20/Analysis-of-Data-Synthesis-for-Raman-Spectroscopy
        Single CSV with 230 spectra × 2473 wavenumbers and a trailing
        ``classAttChloroform`` label column (binary 0/1).

        Returns: spectra, raman_shifts, targets, class_names
        """
        csv_name = "original_chlorinated.csv"
        csv_path = os.path.join(cache_path, csv_name)

        if not os.path.exists(csv_path):
            os.makedirs(cache_path, exist_ok=True)
            LoaderTools.download(
                url=(
                    "https://raw.githubusercontent.com/AaronFlanagan20/"
                    "Analysis-of-Data-Synthesis-for-Raman-Spectroscopy/main/"
                    "data/chlorinated/original_chlorinated.csv"
                ),
                out_dir_path=cache_path,
                out_file_name=csv_name,
            )

        df = pd.read_csv(csv_path)

        label_col = "classAttChloroform"
        if label_col not in df.columns:
            raise FileNotFoundError(
                f"Expected label column '{label_col}' not found in {csv_path}"
            )

        raman_shifts = np.array([float(c) for c in df.columns if c != label_col], dtype=float)
        spectra = df.drop(columns=[label_col]).to_numpy(dtype=float)
        targets = df[label_col].to_numpy(dtype=int)
        class_names = ["no_chloroform", "chloroform"]

        GitHubLoader.logger.debug(
            f"Loaded chlorinated_samples: {spectra.shape[0]} spectra × {spectra.shape[1]} points, "
            f"class counts {{0: {(targets == 0).sum()}, 1: {(targets == 1).sum()}}}"
        )

        return spectra, raman_shifts, targets, class_names

    @staticmethod
    def _load_ramanbiolib(cache_path: str):
        """
        Load the RamanBioLib reference biomolecule dataset from GitHub.

        Downloads ``mteranm/ramanbiolib`` as a ZIP archive (if not cached),
        then joins ``raman_spectra_db.csv`` with ``metadata_db.csv`` to produce
        labelled spectra.

        Returns: spectra, raman_shifts, targets, class_names
        """
        shared_root = cache_path
        repo_main = os.path.join(shared_root, "ramanbiolib-main")

        if not (os.path.isdir(repo_main) and os.listdir(repo_main)):
            zip_name = "ramanbiolib.zip"
            zip_file = os.path.join(shared_root, zip_name)
            os.makedirs(shared_root, exist_ok=True)

            if not os.path.exists(zip_file):
                LoaderTools.download(
                    url="https://github.com/mteranm/ramanbiolib/archive/refs/heads/main.zip",
                    out_dir_path=shared_root,
                    out_file_name=zip_name,
                )

            LoaderTools.extract_zip_file_content(zip_file)

        db_dir = os.path.join(repo_main, "ramanbiolib", "db")
        spectra_path = os.path.join(db_dir, "raman_spectra_db.csv")
        metadata_path = os.path.join(db_dir, "metadata_db.csv")

        if not os.path.isfile(spectra_path):
            raise FileNotFoundError(f"Could not find raman_spectra_db.csv in {db_dir}")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Could not find metadata_db.csv in {db_dir}")

        import ast
        spectra_df = pd.read_csv(spectra_path)

        # Each row stores wavenumbers and intensities as JSON-encoded lists
        raman_shifts = np.array(ast.literal_eval(spectra_df["wavenumbers"].iloc[0]), dtype=float)
        spectra = np.vstack([
            np.array(ast.literal_eval(row), dtype=float)
            for row in spectra_df["intensity"]
        ])

        targets, class_names = encode_labels(spectra_df["component"])

        GitHubLoader.logger.debug(
            f"Loaded biomolecules_reference: {spectra.shape[0]} spectra × {spectra.shape[1]} points, "
            f"{len(class_names)} biomolecules"
        )

        return spectra, raman_shifts, targets, list(class_names)

    @staticmethod
    def _load_ait_glucose_blood_sers(cache_path: str):
        """
        Load the AIT-brainlab blood-SERS glucose calibration dataset.

        IMPORTANT: see the "ait_glucose_blood_sers" DatasetInfo entry above
        (metadata["description"]) for the full provenance/consent/licensing
        caveat before using or re-mirroring this dataset -- it is not a
        routine inclusion.

        Source: data/bloodSERs/5x/txt/ in
        github.com/AIT-brainlab/raman-for-glucose-measurement. Each file is
        one spectrum (1999 tab-separated wavenumber/intensity rows, 785 nm,
        5x lens, 60 s exposure). Concentration is parsed from the filename
        (``blood-<value>_...``); the same-folder ``foil_...`` blank has no
        concentration and is excluded. Downloaded as a full repo ZIP archive,
        matching this loader's other GitHub-zip entries.

        Returns: spectra, raman_shifts, targets, target_names, group_ids
        """
        shared_root = os.path.join(os.path.dirname(cache_path), "ait_glucose_shared")
        repo_main = os.path.join(shared_root, "raman-for-glucose-measurement-main")

        if not (os.path.isdir(repo_main) and os.listdir(repo_main)):
            zip_name = "raman-for-glucose-measurement.zip"
            zip_file = os.path.join(shared_root, zip_name)
            os.makedirs(shared_root, exist_ok=True)

            if not os.path.exists(zip_file):
                LoaderTools.download(
                    url="https://github.com/AIT-brainlab/raman-for-glucose-measurement/archive/refs/heads/main.zip",
                    out_dir_path=shared_root,
                    out_file_name=zip_name,
                )

            LoaderTools.extract_zip_file_content(zip_file)

        txt_dir = os.path.join(repo_main, "data", "bloodSERs", "5x", "txt")
        if not os.path.isdir(txt_dir):
            raise FileNotFoundError(f"Expected data/bloodSERs/5x/txt not found under {repo_main}")

        # Matches e.g. "blood-166_5x_0-43_600_785 nm_60 s_1_..._01.txt".
        # Deliberately does not match "foil_..." (no concentration prefix).
        pattern = re.compile(r"^blood-(\d+(?:\.\d+)?)_")

        spectra_list = []
        raman_shifts_list = []
        concentrations = []

        for fname in sorted(os.listdir(txt_dir)):
            if not fname.endswith(".txt"):
                continue
            match = pattern.match(fname)
            if match is None:
                GitHubLoader.logger.debug(f"Skipping non-blood reference file: {fname}")
                continue

            fpath = os.path.join(txt_dir, fname)
            df = pd.read_csv(fpath, sep="\t", header=None, names=["wavenumber", "intensity"])
            spectra_list.append(df["intensity"].to_numpy(dtype=float))
            raman_shifts_list.append(df["wavenumber"].to_numpy(dtype=float))
            concentrations.append(float(match.group(1)))

        if len(spectra_list) == 0:
            raise FileNotFoundError(f"No blood-*.txt spectra found in {txt_dir}")

        first_rs = raman_shifts_list[0]
        if not all(np.allclose(first_rs, rs) for rs in raman_shifts_list):
            raise ValueError("Raman shift axes differ across bloodSERs files; expected a shared axis")

        raman_shifts = np.array(first_rs, dtype=float)
        spectra = np.stack(spectra_list)
        targets = np.array(concentrations, dtype=float)
        target_names = ["glucose_concentration"]

        # Group by spiked-concentration session: every file sharing a
        # concentration value is a replicate reading of the same aliquot
        # (confirmed from filename timestamps -- see DatasetInfo caveat).
        unique_concs = sorted(set(concentrations))
        conc_to_group = {c: i for i, c in enumerate(unique_concs)}
        group_ids = np.array([conc_to_group[c] for c in concentrations], dtype=int)

        GitHubLoader.logger.debug(
            f"Loaded ait_glucose_blood_sers: {spectra.shape[0]} spectra x {spectra.shape[1]} points, "
            f"{len(unique_concs)} distinct concentration levels {unique_concs}"
        )

        return spectra, raman_shifts, targets, target_names, group_ids

    # ------------------------------------------------------------------
    # liver_cancer_raman (GitHub AzyroSH/CNN_liver-cancer_Raman, MIT)
    # ------------------------------------------------------------------
    # Huang et al. 2023, "Rapid, label-free histopathological diagnosis
    # of liver cancer based on Raman spectroscopy and deep learning"
    # (Nat. Commun. 14:48, doi:10.1038/s41467-022-35696-2).
    #
    # The paper's 12,000-spectrum database (120 carcinoma + 120
    # paracarcinoma samples, 50 spectra each, 532 nm confocal
    # micro-Raman, 500-2,000 cm^-1) was not released; the repo ships a
    # 6,000-spectrum demo as three label-first tab-separated files
    # (label 1 = carcinoma, 0 = paracarcinoma). Quirk: the train and
    # validate files carry 892 data columns while the test file
    # carries the canonical 889; the 892-column files are resampled
    # onto the 889-point grid (the extra 3 points are at the
    # high-wavenumber end -- mean-spectrum correlation with the test
    # file is 0.97 for the shared first 889 columns vs 0.93 for the
    # shifted window).
    _LIVER_REPO = "AzyroSH/CNN_liver-cancer_Raman"
    # Pinned commit (repo default-branch tip at capture time).
    _LIVER_COMMIT = "fdb3743d64763ed0a7cb7c9840c60b9150c68a74"
    # Canonical grid: 500-2,000 cm^-1, 889 points (the paper's and the
    # test file's grid).
    _LIVER_N_POINTS = 889
    _LIVER_GRID = np.arange(
        500.0, 2000.0 + 1500.0 / (_LIVER_N_POINTS - 1) / 2,
        1500.0 / (_LIVER_N_POINTS - 1),
    )
    # 50 spectra per tissue sample in the source database; the
    # train/validate demo files are ordered in contiguous
    # label-pure 50-row sample blocks (verified statistically against
    # shuffled controls: within-block std 13-17% below control).
    _LIVER_BLOCK = 50
    # (local file name, n rows, n data columns, n label-0, n label-1,
    #  group scheme: 'block' = 50-row sample blocks,
    #  'singleton' = one group per row (shuffled file)).
    _LIVER_FILES = (
        ("demo data_Raman liver_train.txt", 4000, 892, 2000, 2000,
         "block", "tr"),
        ("demo data_Raman liver_validate.txt", 1000, 892, 500, 500,
         "block", "va"),
        ("demo data_Raman liver_test.txt", 1000, 889, 500, 500,
         "singleton", "te"),
    )

    @staticmethod
    def _load_liver_cancer_raman(cache_path: str):
        """
        Download (if needed) and load the liver cancer Raman demo
        data: 6,000 spectra, binary carcinoma (1) vs paracarcinoma
        (0), on the common 500-2,000 cm^-1 grid (889 points).

        The three demo files of the authors' repository are
        concatenated in the order train, validate, test. The 892-
        column files (train, validate) are resampled onto the
        889-point grid; the 889-column file (test) is used as is.

        group_ids: 50-row sample blocks in train/validate (100
        groups), singleton groups for the shuffled test file
        (1,000 groups).

        Returns spectra, raman_shifts, targets, class_names,
        group_ids.
        """
        os.makedirs(cache_path, exist_ok=True)
        spectra_list = []
        labels = []
        group_list = []
        for (fname, n_rows, n_data_cols, n_zero, n_one,
             scheme, prefix) in GitHubLoader._LIVER_FILES:
            url = (
                f"https://raw.githubusercontent.com/"
                f"{GitHubLoader._LIVER_REPO}/{GitHubLoader._LIVER_COMMIT}/"
                f"demo%20data/{fname.replace(' ', '%20')}"
            )
            result = LoaderTools.download(
                url=url,
                out_dir_path=cache_path,
                out_file_name=fname,
            )
            if result is None:
                raise RuntimeError(
                    f"[!] liver_cancer_raman: download of {fname} "
                    "failed (network error). See logs."
                )
            arr = np.loadtxt(os.path.join(cache_path, fname), delimiter="\t")
            if arr.ndim != 2 or arr.shape != (n_rows, n_data_cols + 1):
                raise RuntimeError(
                    f"[!] liver_cancer_raman: {fname} has shape "
                    f"{arr.shape}, expected ({n_rows}, {n_data_cols + 1})."
                )
            y = arr[:, 0]
            if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
                raise RuntimeError(
                    f"[!] liver_cancer_raman: unexpected labels in "
                    f"{fname}: {sorted(set(np.unique(y).tolist()))}."
                )
            c0 = int((y == 0).sum())
            c1 = int((y == 1).sum())
            if (c0, c1) != (n_zero, n_one):
                raise RuntimeError(
                    f"[!] liver_cancer_raman: label balance in "
                    f"{fname} is ({c0}, {c1}), expected "
                    f"({n_zero}, {n_one})."
                )
            x = arr[:, 1:]
            if np.isnan(x).any():
                raise RuntimeError(
                    f"[!] liver_cancer_raman: NaN values in {fname}."
                )
            if n_data_cols != GitHubLoader._LIVER_N_POINTS:
                # 892-column files: native grid spans 500-2,000 cm^-1
                # with 892 points; resample onto the canonical
                # 889-point grid (no extrapolation -- the grid lies
                # inside the native range).
                native = np.arange(
                    500.0,
                    2000.0 + 1500.0 / (n_data_cols - 1) / 2,
                    1500.0 / (n_data_cols - 1),
                )
                x = np.vstack(
                    [
                        np.interp(GitHubLoader._LIVER_GRID, native, row)
                        for row in x
                    ]
                )
            for i in range(n_rows):
                spectra_list.append(x[i])
                labels.append(int(y[i]))
                if scheme == "block":
                    group_list.append(
                        f"{prefix}_{i // GitHubLoader._LIVER_BLOCK:02d}"
                    )
                else:
                    group_list.append(f"{prefix}_{i:04d}")

        if len(spectra_list) != 6000:
            raise RuntimeError(
                f"[!] liver_cancer_raman: expected 6000 spectra, got "
                f"{len(spectra_list)}."
            )
        spectra = np.asarray(spectra_list, dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError(
                "[!] liver_cancer_raman: NaN values after resampling."
            )
        targets = np.asarray(labels, dtype=int)
        class_names = ["paracarcinoma", "carcinoma"]
        group_ids = np.asarray(group_list)
        GitHubLoader.logger.debug(
            f"Loaded liver_cancer_raman: {spectra.shape[0]} spectra x "
            f"{spectra.shape[1]} points, "
            f"{len(set(group_ids.tolist()))} groups"
        )
        return spectra, GitHubLoader._LIVER_GRID, targets, class_names, group_ids

    # ------------------------------------------------------------------
    # sers_gna_bacteria (GitHub erkankaratas-eng/sers-gna-bacteria)
    # ------------------------------------------------------------------
    # Gulmus et al. 2026, "SERS spectra of nine bacterial strains on a
    # gold nanorod array - data and analysis code" (Zenodo
    # 10.5281/zenodo.23426967, v1.0.1, CC BY 4.0; the associated
    # article is under review, so the Zenodo record is the citation).
    #
    # 117 SERS spectra (9 strains x 13) of E. coli, P. aeruginosa and
    # S. aureus on an anisotropic gold nanorod array (WITec alpha
    # 300R, 785 nm, x50, 10 mW, ~3.03 s, 4 cm^-1). The loader reads
    # the raw instrument text files (counts, before baseline
    # correction, smoothing or normalization); all 117 share one
    # 1,024-point 247.99-1,997.9 cm^-1 axis, used as is. The
    # deposit's preprocessed CSV (Data S1: OPUS rubberband +
    # Savitzky-Golay + min-max, 1,215 points, S. aureus exported on
    # a 0-1 scale, the other species on 0-2) is not loaded; it
    # serves as the authoritative 117-row manifest.
    _GNA_BACTERIA_REPO = "erkankaratas-eng/sers-gna-bacteria"
    # Pinned commit (repo tip at capture time; the commit the Zenodo
    # v1.0.1 archive was cut from - its data are byte-identical).
    _GNA_BACTERIA_COMMIT = "1028526d22d499568c4baf135b8633b7aadb90b0"
    _GNA_BACTERIA_ZIP_URL = (
        f"https://github.com/{_GNA_BACTERIA_REPO}/archive/"
        f"{_GNA_BACTERIA_COMMIT}.zip"
    )
    _GNA_BACTERIA_ZIP = "sers-gna-bacteria.zip"
    # codeload zips of a pinned commit are byte-reproducible (verified
    # by double download).
    _GNA_BACTERIA_ZIP_SIZE = 10807998
    _GNA_BACTERIA_ZIP_SHA256 = (
        "3ecb3febcbf7b60ae4fb3f069e0c1ecb682ff9817961906a43f6a9ff751dd0ef"
    )
    _GNA_BACTERIA_ROOT = f"sers-gna-bacteria-{_GNA_BACTERIA_COMMIT}"
    _GNA_BACTERIA_N_SPECTRA = 117
    _GNA_BACTERIA_N_SPECTRA_PER_STRAIN = 13
    _GNA_BACTERIA_N_POINTS = 1024

    @staticmethod
    def _load_sers_gna_bacteria(cache_path: str):
        """
        Download (if needed) and load the SERS gold-nanorod-array
        bacteria dataset: 117 raw spectra (9 strains x 13) on the
        shared native 1,024-point 247.99-1,997.9 cm^-1 axis.

        The deposit's Data S1 CSV provides the 117-row manifest
        (species, strain, raw file number, acquisition order); each
        row joins the corresponding raw WITec instrument file in
        data/raw_spectra/<strain>/<raw_file_number>.txt. Rows are
        emitted in the manifest's (Data S1) row order.

        group_ids = strain folder name (9 groups of 13: all spectra
        of a strain come from one droplet on one substrate of one
        culture).

        Returns spectra, raman_shifts, targets, class_names,
        group_ids.
        """
        os.makedirs(cache_path, exist_ok=True)
        zip_file = os.path.join(cache_path, GitHubLoader._GNA_BACTERIA_ZIP)
        if (
            os.path.exists(zip_file)
            and os.path.getsize(zip_file)
            != GitHubLoader._GNA_BACTERIA_ZIP_SIZE
        ):
            os.remove(zip_file)
        result = LoaderTools.download(
            url=GitHubLoader._GNA_BACTERIA_ZIP_URL,
            out_dir_path=cache_path,
            out_file_name=GitHubLoader._GNA_BACTERIA_ZIP,
            hash_target=GitHubLoader._GNA_BACTERIA_ZIP_SHA256,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                "[!] sers_gna_bacteria: archive download failed "
                "(network error or checksum mismatch). See logs."
            )
        repo_root = os.path.join(cache_path, GitHubLoader._GNA_BACTERIA_ROOT)
        if not os.path.isdir(repo_root):
            LoaderTools.extract_zip_file_content(zip_file)
        manifest = os.path.join(
            repo_root, "data", "sers_spectra_preprocessed.csv"
        )
        raw_root = os.path.join(repo_root, "data", "raw_spectra")
        if not os.path.isfile(manifest) or not os.path.isdir(raw_root):
            raise RuntimeError(
                "[!] sers_gna_bacteria: the archive layout changed "
                "(data/sers_spectra_preprocessed.csv and/or "
                "data/raw_spectra/ missing)."
            )
        df = pd.read_csv(manifest)
        if df.shape[0] != GitHubLoader._GNA_BACTERIA_N_SPECTRA:
            raise RuntimeError(
                "[!] sers_gna_bacteria: expected "
                f"{GitHubLoader._GNA_BACTERIA_N_SPECTRA} manifest "
                f"rows, got {df.shape[0]}."
            )
        # Join the manifest to the strain folders: each
        # (species, strain) group must match exactly one folder by
        # its set of raw file numbers.
        folder_files = {
            f: {
                int(t[:-4])
                for t in os.listdir(os.path.join(raw_root, f))
                if t.endswith(".txt")
            }
            for f in os.listdir(raw_root)
        }
        folder_for = {}
        for (species, strain), g in df.groupby(["species", "strain"]):
            if len(g) != GitHubLoader._GNA_BACTERIA_N_SPECTRA_PER_STRAIN:
                raise RuntimeError(
                    "[!] sers_gna_bacteria: strain "
                    f"{species} {strain} has {len(g)} manifest rows, "
                    "expected "
                    f"{GitHubLoader._GNA_BACTERIA_N_SPECTRA_PER_STRAIN}."
                )
            need = {int(v) for v in g["raw_file_number"]}
            cands = [
                f for f, s in folder_files.items() if s == need
            ]
            if len(cands) != 1:
                raise RuntimeError(
                    "[!] sers_gna_bacteria: expected exactly one "
                    f"folder for {species} {strain}, matched "
                    f"{cands}."
                )
            folder_for[(species, strain)] = cands[0]
        if len(folder_for) != 9:
            raise RuntimeError(
                "[!] sers_gna_bacteria: expected 9 strains, found "
                f"{len(folder_for)}."
            )

        axis_ref = None
        spectra_list = []
        targets_list = []
        group_list = []
        class_names = []
        class_index = {}
        for row in df.itertuples(index=False):
            key = (row.species, row.strain)
            folder = folder_for[key]
            raw_path = os.path.join(
                raw_root, folder, f"{int(row.raw_file_number)}.txt"
            )
            arr = np.loadtxt(raw_path, delimiter="\t")
            if arr.shape != (
                GitHubLoader._GNA_BACTERIA_N_POINTS,
                2,
            ) or not np.isfinite(arr).all():
                raise RuntimeError(
                    "[!] sers_gna_bacteria: bad raw file "
                    f"{folder}/{int(row.raw_file_number)}.txt "
                    f"(shape {arr.shape}, non-finite values)."
                )
            if axis_ref is None:
                axis_ref = arr[:, 0]
                if not np.all(np.diff(axis_ref) > 0):
                    raise RuntimeError(
                        "[!] sers_gna_bacteria: the wavenumber axis "
                        "is not strictly increasing."
                    )
            elif not np.allclose(arr[:, 0], axis_ref, atol=1e-6):
                raise RuntimeError(
                    "[!] sers_gna_bacteria: the raw files do not "
                    "share a common wavenumber axis."
                )
            if key not in class_index:
                class_index[key] = len(class_names)
                class_names.append(folder)
            targets_list.append(class_index[key])
            group_list.append(folder)
            spectra_list.append(arr[:, 1])

        if not (
            240.0 < axis_ref[0] < 260.0
            and 1990.0 < axis_ref[-1] < 2010.0
        ):
            raise RuntimeError(
                "[!] sers_gna_bacteria: unexpected wavenumber range "
                f"({axis_ref[0]:.3f}..{axis_ref[-1]:.3f} cm^-1)."
            )
        spectra = np.vstack(spectra_list)
        targets = np.asarray(targets_list, dtype=int)
        group_ids = np.asarray(group_list)
        GitHubLoader.logger.debug(
            f"Loaded sers_gna_bacteria: {spectra.shape[0]} spectra x "
            f"{spectra.shape[1]} points, {len(class_names)} strains, "
            f"{len(set(group_ids.tolist()))} groups"
        )
        return (
            spectra,
            axis_ref,
            targets,
            class_names,
            group_ids,
        )

    # ------------------------------------------------------------------
    # sers_pesticide_produce (GitHub jianlin-cheng/SERSFormer2, GPL-3.0)
    # ------------------------------------------------------------------
    # Hegde et al. 2025, "Leveraging SERS and Transformer Models for
    # Simultaneous Detection of Multiple Pesticides in Fresh Produce"
    # (ACS Appl. Mater. Interfaces 17(1):2018-2031,
    # doi:10.1021/acsami.4c17777).
    #
    # The public repository is a pruned subset of the paper's 10+
    # mixture panel: 33 SERS spectra of pesticide-spiked fresh produce
    # (a spinach control, a spinach carbophenothion+oxamyl 50:50
    # mixture, and a strawberry carbophenothion+phosmet 50:50 mixture),
    # 3 replicate spectra per (produce, mixture, concentration)
    # condition. Each CSV is a 1,556-point native-axis export
    # (wavenumber index column + one intensity column). The loader
    # resamples every spectrum onto the common 501-1,999 cm^-1 @1 grid
    # (1,499 points) and reports the per-pesticide concentration (ppm)
    # for the three pesticides present in the deposit as a multi-output
    # regression target; the paper's multi-label presence task is
    # derivable (a pesticide is present iff its concentration > 0).
    _SFP2_REPO = "jianlin-cheng/SERSFormer2"
    _SFP2_COMMIT = "bee22a7067acc5108b0477c694d5dd3a792dd569"
    _SFP2_ZIP_URL = (
        f"https://github.com/{_SFP2_REPO}/archive/{_SFP2_COMMIT}.zip"
    )
    _SFP2_ZIP = "sersformer2.zip"
    _SFP2_ZIP_SIZE = 4019521
    _SFP2_ZIP_SHA256 = (
        "0a40e986a73a89b495633dea5e27f339ccfb3040979591a2b4310a029157e0bc"
    )
    _SFP2_ROOT = f"SERSFormer2-{_SFP2_COMMIT}"
    _SFP2_DATA_DIR = "Spinach_Strawbery"
    # Common grid: 501-1,999 cm^-1 @1 cm^-1 (1,499 points); every
    # native axis (1,556 pts, ~500-2,000 cm^-1) fully covers it, so
    # no extrapolation is involved.
    _SFP2_GRID = np.arange(501.0, 1999.0 + 0.5, 1.0)
    # The three pesticides present in the public deposit (the paper's
    # other two panel members, coumaphos and thiabendazole, have no
    # spectra here, so they are not shipped as degenerate targets).
    _SFP2_TARGETS = ("carbophenothion", "oxamyl", "phosmet")
    _SFP2_N_SPECTRA = 33
    _SFP2_N_GROUPS = 11
    _SFP2_NATIVE_POINTS = 1556

    @staticmethod
    def _load_sers_pesticide_produce(cache_path: str):
        """
        Download (if needed) and load the SERSFormer2 pesticide
        dataset: 33 SERS spectra of pesticide-spiked fresh produce on
        the common 501-1,999 cm^-1 @1 cm^-1 grid (1,499 points), with
        a 3-column regression target (per-pesticide concentration in
        ppm for carbophenothion, oxamyl and phosmet).

        group_ids = condition (produce x mixture x concentration
        level; 11 groups of 3 replicate spectra).

        Returns spectra, raman_shifts, targets, target_names,
        group_ids.
        """
        os.makedirs(cache_path, exist_ok=True)
        zip_file = os.path.join(cache_path, GitHubLoader._SFP2_ZIP)
        if (
            os.path.exists(zip_file)
            and os.path.getsize(zip_file) != GitHubLoader._SFP2_ZIP_SIZE
        ):
            os.remove(zip_file)
        result = LoaderTools.download(
            url=GitHubLoader._SFP2_ZIP_URL,
            out_dir_path=cache_path,
            out_file_name=GitHubLoader._SFP2_ZIP,
            hash_target=GitHubLoader._SFP2_ZIP_SHA256,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                "[!] sers_pesticide_produce: archive download failed "
                "(network error or checksum mismatch). See logs."
            )
        repo_root = os.path.join(cache_path, GitHubLoader._SFP2_ROOT)
        if not os.path.isdir(repo_root):
            LoaderTools.extract_zip_file_content(zip_file)
        data_root = os.path.join(repo_root, GitHubLoader._SFP2_DATA_DIR)
        if not os.path.isdir(data_root):
            raise RuntimeError(
                "[!] sers_pesticide_produce: the archive layout "
                f"changed (no {GitHubLoader._SFP2_DATA_DIR}/ folder)."
            )

        files = []
        for _dp, _dn, fn in os.walk(data_root):
            for f in fn:
                if f.upper().endswith(".CSV"):
                    files.append(os.path.join(_dp, f))
        if len(files) != GitHubLoader._SFP2_N_SPECTRA:
            raise RuntimeError(
                f"[!] sers_pesticide_produce: expected "
                f"{GitHubLoader._SFP2_N_SPECTRA} spectra, found "
                f"{len(files)} (archive layout changed?)."
            )

        grid = GitHubLoader._SFP2_GRID
        spectra_list = []
        targets_list = []
        group_list = []
        for f in sorted(files):
            parts = os.path.dirname(
                os.path.relpath(f, data_root)
            ).split(os.sep)
            produce = parts[0]
            ratio = parts[1]
            if parts[2] == "No_pest_present":
                # control: produce/ratio/No_pest_present/No_pest_present
                mixture = "No_pest_present"
                conc_tag = "control"
                conc = 0.0
                present = []
            else:
                mixture = parts[2]
                conc_dir = parts[3]
                conc_tag = conc_dir
                conc = float(
                    re.findall(r"[-+]?\d*\.\d+|\d+", conc_dir)[0]
                ) * (float(ratio.split("_")[0]) / 100.0)
                present = mixture.split("_")
            df = pd.read_csv(f, header=None, index_col=0)
            x = df.index.to_numpy(dtype=float)
            y = df.iloc[:, 0].to_numpy(dtype=float)
            if (
                x.shape[0] != GitHubLoader._SFP2_NATIVE_POINTS
                or not np.all(np.diff(x) > 0)
                or x[0] > grid[0]
                or x[-1] < grid[-1]
            ):
                raise RuntimeError(
                    f"[!] sers_pesticide_produce: {f} does not carry "
                    f"a {GitHubLoader._SFP2_NATIVE_POINTS}-point axis "
                    "covering the 501-1,999 cm^-1 grid "
                    f"({x[0]:.1f}..{x[-1]:.1f})."
                )
            if np.isnan(y).any():
                raise RuntimeError(
                    f"[!] sers_pesticide_produce: NaN values in {f}."
                )
            spectra_list.append(np.interp(grid, x, y))
            targets_list.append(
                np.array(
                    [
                        conc if p in present else 0.0
                        for p in GitHubLoader._SFP2_TARGETS
                    ],
                    dtype=float,
                )
            )
            group_list.append(f"{produce}_{mixture}_{conc_tag}")

        if len(set(group_list)) != GitHubLoader._SFP2_N_GROUPS:
            raise RuntimeError(
                "[!] sers_pesticide_produce: expected "
                f"{GitHubLoader._SFP2_N_GROUPS} conditions, found "
                f"{len(set(group_list))}."
            )
        spectra = np.asarray(spectra_list, dtype=float)
        targets = np.asarray(targets_list, dtype=float)
        group_ids = np.asarray(group_list)
        if np.isnan(spectra).any() or np.isnan(targets).any():
            raise RuntimeError(
                "[!] sers_pesticide_produce: NaN values."
            )
        GitHubLoader.logger.debug(
            f"Loaded sers_pesticide_produce: {spectra.shape[0]} "
            f"spectra x {spectra.shape[1]} points, "
            f"{targets.shape[1]} targets, "
            f"{len(set(group_ids.tolist()))} conditions"
        )
        return (
            spectra,
            grid,
            targets,
            list(GitHubLoader._SFP2_TARGETS),
            group_ids,
        )

    # ------------------------------------------------------------------
    # uraman_mofs / uraman_mof74_1000_2000 / uraman_mof74_70_3000
    # (IBM uRaman-Dataset, github.com/IBM/uRaman-Dataset)
    # ------------------------------------------------------------------
    # Standalone IBM GitHub deposit (maintainer: Matheus Esteves
    # Ferreira; CDLA-Sharing-1.0; commit 47e712d3..., 2023-06-19).
    # No indexed parent paper exists - the repository is the
    # citation.
    #
    # The 14 spectra split into three measurement campaigns:
    #   1. top level (2023-02..04): HKUST-1, Mg-MOF74, ZIF-8 - one
    #      stitched spectrum each, ~0-4,000 cm^-1, WITec UHTS300S
    #      (531.970 nm x2 / 632.810 nm x1, 1-2 mW);
    #   2. MeMOF-74/1000-2000 cm-1 (nominal window): six MOF-74
    #      metal-series samples (Mg, Zn, Zn75Mg25, Au@ variants),
    #      WITec Alpha 300 RAS, 532 nm @1 mW, 1800 g/mm;
    #   3. MeMOF-74/70-3000 cm-1 (stitched): the same series minus
    #      Au@Mg-MOF74 (no stitching: high luminescence background,
    #      per the deposit's About.md).
    # One key per campaign, each spectrum resampled onto the
    # campaign's common @1 cm^-1 grid. All 14 spectra are
    # peak-normalized (max = 1.0) as deposited; one spectrum per
    # class, so these are reference sets, not CV-learnable
    # classification data (documented in the entries).
    #
    # Each CSV: header 'Rel. Cm-1,Intensity (A.U.)', wavenumbers in
    # scientific notation. codeload zips of a pinned commit are
    # byte-reproducible (verified by double download).
    _URAMAN_COMMIT = "47e712d326fb0458ee8218ab73a2cfc269dcdfab"
    _URAMAN_ZIP_URL = (
        f"https://codeload.github.com/IBM/uRaman-Dataset/zip/"
        f"{_URAMAN_COMMIT}"
    )
    _URAMAN_ZIP = "uRaman-Dataset.zip"
    _URAMAN_ZIP_SIZE = 819336
    _URAMAN_ZIP_SHA256 = (
        "bc772edb201e4a6f0de74b1814df92080601e577ac13b67a83def1e1b520b1fd"
    )
    _URAMAN_ROOT = f"uRaman-Dataset-{_URAMAN_COMMIT}"

    @staticmethod
    def _fetch_uraman_root(cache_path: str) -> str:
        """
        Download (if needed) the commit-pinned codeload archive of
        the IBM uRaman-Dataset repository and return the path of
        the extracted repository root.
        """
        os.makedirs(cache_path, exist_ok=True)
        zip_file = os.path.join(cache_path, GitHubLoader._URAMAN_ZIP)
        if (
            os.path.exists(zip_file)
            and os.path.getsize(zip_file) != GitHubLoader._URAMAN_ZIP_SIZE
        ):
            os.remove(zip_file)
        result = LoaderTools.download(
            url=GitHubLoader._URAMAN_ZIP_URL,
            out_dir_path=cache_path,
            out_file_name=GitHubLoader._URAMAN_ZIP,
            hash_target=GitHubLoader._URAMAN_ZIP_SHA256,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                "[!] uraman: codeload archive download failed "
                "(network error or checksum mismatch). See logs."
            )
        repo_root = os.path.join(cache_path, GitHubLoader._URAMAN_ROOT)
        if not os.path.isdir(os.path.join(repo_root, "Dataset")):
            LoaderTools.extract_zip_file_content(zip_file)
            if not os.path.isdir(os.path.join(repo_root, "Dataset")):
                raise RuntimeError(
                    "[!] uraman: the pinned archive layout changed "
                    "(Dataset/ missing after extraction)."
                )
        return repo_root

    @staticmethod
    def _load_uraman_campaign(
        cache_path: str, files: dict, grid: np.ndarray
    ):
        """
        Shared reader for the three uRaman measurement campaigns:
        parses each campaign CSV (two-column
        'Rel. Cm-1,Intensity (A.U.)' WITec export, wavenumbers in
        scientific notation), verifies that the native axis covers
        the campaign's common grid, and linearly interpolates onto
        it.

        files maps class label -> repo-relative CSV path; the row
        order of the returned spectra follows that mapping.

        Returns spectra, raman_shifts, targets, class_names.
        """
        root = GitHubLoader._fetch_uraman_root(cache_path)
        spectra_list = []
        for label, rel in files.items():
            path = os.path.join(root, rel)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"[!] uraman: {rel} is missing from the pinned "
                    "archive (layout changed?)."
                )
            arr = np.genfromtxt(path, delimiter=",", skip_header=1)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise RuntimeError(
                    f"[!] uraman: {rel} is not a two-column "
                    "'Rel. Cm-1,Intensity (A.U.)' CSV."
                )
            x, y = arr[:, 0], arr[:, 1]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise RuntimeError(
                    f"[!] uraman: {rel} contains non-finite values."
                )
            if not np.all(np.diff(x) > 0):
                raise RuntimeError(
                    f"[!] uraman: {rel}: wavenumber axis is not "
                    "strictly increasing."
                )
            if x[0] > grid[0] or x[-1] < grid[-1]:
                raise RuntimeError(
                    f"[!] uraman: {rel} does not cover the common "
                    f"grid (native {x[0]:.3f}..{x[-1]:.3f} cm^-1 "
                    f"vs {grid[0]:.0f}..{grid[-1]:.0f} expected)."
                )
            spectra_list.append(np.interp(grid, x, y))
        spectra = np.asarray(spectra_list, dtype=float)
        encoded_targets, class_names = encode_labels(
            pd.Series(list(files.keys()))
        )
        GitHubLoader.logger.debug(
            f"Loaded uraman campaign: {spectra.shape[0]} spectra x "
            f"{spectra.shape[1]} points, {len(set(class_names))} "
            "classes"
        )
        return (
            spectra,
            grid,
            encoded_targets,
            list(class_names),
        )

    @staticmethod
    def _load_uraman_mofs(cache_path: str):
        """
        Load the three top-level reference spectra of the IBM
        uRaman-Dataset (HKUST-1, Mg-MOF74, ZIF-8), resampled onto
        the common 0-3,997 cm^-1 @1 cm^-1 grid (3,998 points).

        Returns spectra, raman_shifts, targets, class_names.
        """
        return GitHubLoader._load_uraman_campaign(
            cache_path,
            {
                "HKUST-1": "Dataset/HKUST-1/HKUST-1.csv",
                "ZIF-8": "Dataset/ZIF 8/ZIF-8.csv",
                "Mg-MOF74": "Dataset/Mg-MOF74/Mg-MOF74.csv",
            },
            np.arange(0.0, 3998.0),
        )

    @staticmethod
    def _load_uraman_mof74_1000_2000(cache_path: str):
        """
        Load the six MOF-74 metal-series spectra of the deposit's
        nominal 1,000-2,000 cm^-1 window, resampled onto the common
        970-1,887 cm^-1 @1 cm^-1 grid (918 points; the intersection
        of the six native axes, limited by Au@Mg-MOF74's shifted
        860.797-1,887.76 cm^-1 axis).

        The 1,000-2,000 cm^-1 file of Au@Zn75Mg25-MOF74 is spelled
        with a double dash in the deposit; the label is normalized
        to a single dash.

        Returns spectra, raman_shifts, targets, class_names.
        """
        return GitHubLoader._load_uraman_campaign(
            cache_path,
            {
                "Mg-MOF74": "Dataset/MeMOF-74/1000-2000 cm-1/Mg-MOF74.csv",
                "Zn-MOF74": "Dataset/MeMOF-74/1000-2000 cm-1/Zn-MOF74.csv",
                "Zn75Mg25-MOF74": (
                    "Dataset/MeMOF-74/1000-2000 cm-1/Zn75Mg25-MOF74.csv"
                ),
                "Au@Mg-MOF74": (
                    "Dataset/MeMOF-74/1000-2000 cm-1/Au@Mg-MOF74.csv"
                ),
                "Au@Zn-MOF74": (
                    "Dataset/MeMOF-74/1000-2000 cm-1/Au@Zn-MOF74.csv"
                ),
                "Au@Zn75Mg25-MOF74": (
                    "Dataset/MeMOF-74/1000-2000 cm-1/"
                    "Au@Zn75Mg25--MOF74.csv"
                ),
            },
            np.arange(970.0, 1888.0),
        )

    @staticmethod
    def _load_uraman_mof74_70_3000(cache_path: str):
        """
        Load the five MOF-74 metal-series spectra of the deposit's
        nominal 70-3,000 cm^-1 stitched window (native axis
        0-2,999.35 cm^-1), resampled onto the common 0-2,999 cm^-1
        @1 cm^-1 grid (3,000 points). Au@Mg-MOF74 has no stitched
        spectrum (high luminescence background, per the deposit's
        About.md).

        Returns spectra, raman_shifts, targets, class_names.
        """
        return GitHubLoader._load_uraman_campaign(
            cache_path,
            {
                "Mg-MOF74": "Dataset/MeMOF-74/70-3000 cm-1/Mg-MOF74.csv",
                "Zn-MOF74": "Dataset/MeMOF-74/70-3000 cm-1/Zn-MOF74.csv",
                "Zn75Mg25-MOF74": (
                    "Dataset/MeMOF-74/70-3000 cm-1/Zn75Mg25-MOF74.csv"
                ),
                "Au@Zn-MOF74": (
                    "Dataset/MeMOF-74/70-3000 cm-1/Au@Zn-MOF74.csv"
                ),
                "Au@Zn75Mg25-MOF74": (
                    "Dataset/MeMOF-74/70-3000 cm-1/"
                    "Au@Zn75Mg25-MOF74.csv"
                ),
            },
            np.arange(0.0, 3000.0),
        )
