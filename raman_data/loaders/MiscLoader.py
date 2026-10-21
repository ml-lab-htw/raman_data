import glob
import logging
import os
import re
from collections import Counter
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from raman_data.loaders.BaseLoader import BaseLoader
from raman_data.loaders.LoaderTools import LoaderTools
from raman_data.loaders.utils import encode_labels
from raman_data.types import RamanDataset, TASK_TYPE, DatasetInfo, CACHE_DIR, APPLICATION_TYPE, HASH_TYPE


class MiscLoader(BaseLoader):
    """
    Loader for miscellaneous Raman spectroscopy datasets from various sources
    that don't fit into a dedicated loader (Dropbox, SharePoint, email-only, etc.).
    """

    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "raman-data", "misc")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Misc)

    DATASETS = {
        "deepr_denoising": DatasetInfo(
            task_type=TASK_TYPE.Denoising,
            id="deepr_denoising",
            name="DeepeR Denoising",
            short_name="DeepeR Denoising",
            license="MIT",
            loader=lambda cache_path: MiscLoader._load_deepr_denoising(cache_path),
            metadata={
                "full_name": "DeepeR Denoising Dataset",
                "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EqZaY-_FrGdImybIGuMCvb8Bo_YD1Bc9ATBxbLxdDIv0RA?e=5%3aHhLp91&fromShare=true&at=9",
                "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
                "bibtex": "@article{Horgan_2021, title={High-Throughput Molecular Imaging via Deep-Learning-Enabled Raman Spectroscopy}, volume={93}, ISSN={1520-6882}, url={http://dx.doi.org/10.1021/acs.analchem.1c02178}, DOI={10.1021/acs.analchem.1c02178}, number={48}, journal={Analytical Chemistry}, publisher={American Chemical Society (ACS)}, author={Horgan, Conor C. and Jensen, Magnus and Nagelkerke, Anika and St-Pierre, Jean-Philippe and Vercauteren, Tom and Stevens, Molly M. and Bergholt, Mads S.}, year={2021}, month=nov, pages={15850--15860}}",
                "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
                "description": "Raman spectral denoising dataset from DeepeR paper. Contains noisy input spectra and corresponding denoised target spectra for training deep learning denoising models.",
            }
        ),
        "deepr_super_resolution": DatasetInfo(
            task_type=TASK_TYPE.SuperResolution,
            id="deepr_super_resolution",
            name="DeepeR Super-Resolution Dataset",
            short_name="DeepeR Super-Res.",
            license="MIT",
            loader=lambda cache_path: MiscLoader._load_deepr_super_resolution(cache_path),
            metadata={
                "full_name": "DeepeR Super-Resolution Dataset",
                "source": "https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EuIIZkQGtT5NgQcYO_SOzigB706Q8b0EddSLEDGUN22EbA?e=5%3axGyu4b&fromShare=true&at=9",
                "paper": "https://doi.org/10.1021/acs.analchem.1c02178",
                "bibtex": "@article{Horgan_2021, title={High-Throughput Molecular Imaging via Deep-Learning-Enabled Raman Spectroscopy}, volume={93}, ISSN={1520-6882}, url={http://dx.doi.org/10.1021/acs.analchem.1c02178}, DOI={10.1021/acs.analchem.1c02178}, number={48}, journal={Analytical Chemistry}, publisher={American Chemical Society (ACS)}, author={Horgan, Conor C. and Jensen, Magnus and Nagelkerke, Anika and St-Pierre, Jean-Philippe and Vercauteren, Tom and Stevens, Molly M. and Bergholt, Mads S.}, year={2021}, month=nov, pages={15850--15860}}",
                "citation": "Horgan et al., Analytical Chemistry 2021, 93, 48, 15850-15860.",
                "description": "Hyperspectral super-resolution dataset from DeepeR paper. Contains low-resolution input spectra and high-resolution target spectra for training super-resolution models.",
            }
        ),
        "covid19_serum": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="covid19_serum",
            name="COVID-19 Human Serum",
            short_name="COVID-19 Serum",
            license="unknown",
            loader=lambda cache_path: MiscLoader._load_covid(cache_path),
            metadata={
                "full_name": "COVID-19 in human serum using Raman spectroscopy",
                "source": "Only via E-Mail",
                "description": "Raman spectra of blood serum from 20 subjects (10 RT-PCR/ELISA-confirmed COVID-19 patients and 10 healthy controls) collected under Ethics Committee protocol 26691419.6.0000.5492 (Universidade Anhembi Morumbi). Acquired with a Dimension P1 dispersive Raman spectrometer (Lambda Solutions), 830 nm excitation, 350 mW, 400–1800 cm⁻¹, 4 cm⁻¹ resolution, 30 s acquisition (3 s × 10 accumulations), 80 µL serum per measurement, triplicate per sample. Preprocessed with 7th-order polynomial baseline correction and area normalisation (1-norm). Published PCA-DA/PLS-DA model achieved 87% sensitivity and 100% specificity.",
                "paper": "https://doi.org/10.1007/s10103-021-03488-7",
                "bibtex": "@article{Goulart_2022, title={Diagnosing COVID-19 in human serum using Raman spectroscopy}, volume={37}, ISSN={1435-604X}, url={http://dx.doi.org/10.1007/s10103-021-03488-7}, DOI={10.1007/s10103-021-03488-7}, number={4}, journal={Lasers in Medical Science}, publisher={Springer Science and Business Media LLC}, author={Goulart, Ana Cristina Castro and Silveira, Landulfo and Carvalho, Henrique Cunha and Dorta, Cristiane Bissoli and Pacheco, Marcos Tadeu T. and Zangaro, Renato Amaro}, year={2022}, month=jan, pages={2217--2226}}",
                "citation": [
                    "Goulart, Ana Cristina Castro, et al. 'Diagnosing COVID-19 in human serum using Raman spectroscopy.' Lasers in Medical Science 37.4 (2022): 2217-2226."
                ],
            }
        ),
        "bacteria_identification": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Medical,
            id="bacteria_identification",
            name="Pathogenic Bacteria",
            short_name="Pathogenic Bacteria",
            license="MIT",
            loader=lambda cache_path: MiscLoader._load_csho33_bacteria(cache_path),
            metadata={
                "full_name": "Pathogenic Bacteria Raman Dataset",
                "source": "https://github.com/csho33/bacteria-ID",
                "paper": "https://doi.org/10.1038/s41467-019-12898-9",
                "bibtex": "@article{Ho_2019, title={Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning}, volume={10}, ISSN={2041-1723}, url={http://dx.doi.org/10.1038/s41467-019-12898-9}, DOI={10.1038/s41467-019-12898-9}, number={1}, journal={Nature Communications}, publisher={Springer Science and Business Media LLC}, author={Ho, Chi-Sing and Jean, Neal and Hogan, Catherine A. and Blackmon, Lena and Jeffrey, Stefanie S. and Holodniy, Mark and Banaei, Niaz and Saleh, Amr A. E. and Ermon, Stefano and Dionne, Jennifer}, year={2019}, month=oct}",
                "citation": [
                    "Ho, C.-S., Jean, N., Hogan, C. A., et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nat Commun 10, 4927 (2019)."
                ],
                "description": "60,000 SERS spectra (2,000 per isolate across three measurement time-points) from 30 clinically relevant bacterial and yeast isolates, including an MRSA/MSSA isogenic pair. Acquired on a Horiba LabRAM HR Evolution spectrometer (633 nm, 13.17 mW, 300 l/mm grating, 1.2 cm⁻¹ dispersion, Olympus MPLAN 100× NA 0.9, ~1 µm spot) at 1 s integration time (SNR ~4.1) on gold-coated silica substrates (200 nm Au, electron beam evaporation). Published 25-layer 1D ResNet CNN (Adam, batch 10) achieved 82.2 ± 0.3% isolate accuracy, 97.0 ± 0.3% antibiotic treatment accuracy, and 99.7% treatment classification with 10 spectra in clinical validation on 50 patients (Stanford/Palo Alto VA).",
            },
            # Isolate-level structure: ~30 isolates with ~2,000 spectra each
            is_grouped=True,
            # Source preprocessing (from source paper): 5th-order polynomial baseline,
            # cropped 381.98-1792.4 cm⁻¹, min-max normalization
            embedded_preprocessing="5th-order polynomial baseline correction, cropped 381.98–1792.4 cm⁻¹, min-max normalization (source-applied)",
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        "marine_pathogens": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="marine_pathogens",
            name="Marine Pathogens (Urechis unicinctus)",
            short_name="Marine Pathogens",
            # Licensing note: the GitHub repo rehosting this CSV
            # (zshicode/Raman-Spectra-Deep-Learning) has NO LICENSE file
            # (confirmed via GitHub API: license: null) -- it grants no
            # redistribution rights of its own. The legitimizing source is
            # the underlying paper (Yu et al. 2021), which is Green Open
            # Access (confirmed via Semantic Scholar/Unpaywall metadata,
            # DOI 10.1021/acs.analchem.1c00431) under CC BY-NC-SA. This is a
            # real usage restriction: NON-COMMERCIAL USE ONLY. Cite the
            # paper, not the GitHub repo, as the license-granting source.
            license="CC BY-NC-SA (paper; non-commercial)",
            loader=lambda cache_path: MiscLoader._load_yu2021_marine_pathogens(cache_path),
            metadata={
                "full_name": "Marine Pathogen Identification via Raman Spectroscopy (Urechis unicinctus isolates)",
                "source": "https://github.com/zshicode/Raman-Spectra-Deep-Learning/blob/main/multi.csv",
                "paper": "https://doi.org/10.1021/acs.analchem.1c00431",
                "bibtex": "@article{Yu_2021, title={Analysis of Raman Spectra by Using Deep Learning Methods in the Identification of Marine Pathogens}, volume={93}, ISSN={1520-6882}, url={http://dx.doi.org/10.1021/acs.analchem.1c00431}, DOI={10.1021/acs.analchem.1c00431}, number={32}, journal={Analytical Chemistry}, publisher={American Chemical Society (ACS)}, author={Yu, Shixiang and Li, Xin and Lu, Weilai and Li, Hanfei and Fu, Y. Vincent and Liu, Fanghua}, year={2021}, month=aug, pages={11089--11098}}",
                "citation": [
                    "Yu, S., Li, X., Lu, W., Li, H., Fu, Y. V., & Liu, F. (2021). Analysis of Raman Spectra by Using Deep Learning Methods in the Identification of Marine Pathogens. Analytical Chemistry, 93(32), 11089-11098."
                ],
                "description": "1,138 Raman spectra (1200 points each, 600-1800 cm⁻¹) from eight bacterial/yeast strains (SX-1 through SX-8, labeled 0-7) isolated from the marine organism Urechis unicinctus, including four kinds of pathogens, per class counts of 149/150/150/102/137/150/150/150. Spectra are already 0-1 (min-max) normalized in the source CSV, per the rehosting repo's README. The source paper reports LSTM-based classification achieving isolate-level accuracy exceeding 94%, outperforming a standard CNN baseline. Rehosted (no header row, trailing integer label column, no sample/replicate IDs, no explicit wavenumber axis) as flat CSV by github.com/zshicode/Raman-Spectra-Deep-Learning, itself unlicensed -- see license note above.",
            },
            # Source-applied 0-1 (min-max) normalization per rehosting repo's README
            # ("Following Yu et al. (2021), the Raman spectra of each sample is
            # preprocessed by 0-1 normalization").
            embedded_preprocessing="0-1 (min-max) normalization (source-applied)",
            # NOT checked: the rehosted CSV carries no sample/isolate/replicate ID
            # column, and the source paper's full text (methods/SI, which might
            # describe how many spectra were taken per physical isolate) could not
            # be accessed -- the paywalled ACS page, the paper's own Green-OA
            # institutional repository (ir.yic.ac.cn, unreachable from this
            # network), CORE.ac.uk (metadata only, no full text), PMC (no
            # deposit), and ResearchGate (no self-archived copy found) were all
            # tried. Left as None (unknown), not assumed False, per convention --
            # do not infer grouping from the uneven per-class spectra counts
            # (149/150/150/102/137/150/150/150) alone.
            is_grouped=None,
            # Checked: no missing (NaN) label values (labels are a dense
            # 0-7 integer column with no gaps).
            has_missing_labels=False,
        ),
        "marine_pathogens_binary": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Biological,
            id="marine_pathogens_binary",
            name="Marine Pathogens (Binary: A. baumannii vs. P. nitritireducens)",
            short_name="Marine Pathogens (Binary)",
            # Same licensing situation as marine_pathogens: the GitHub repo
            # rehosting this CSV has no LICENSE file of its own (license:
            # null via GitHub API) -- the legitimizing source is the
            # underlying paper's own Green-OA CC BY-NC-SA status. See the
            # citation-confidence note in the description below: the
            # attribution to this specific paper is taken as given from the
            # rehosting repo's README, NOT independently confirmed (unlike
            # marine_pathogens/multi.csv, whose 8-strain Urechis unicinctus
            # description is directly and exactly matched by the paper's own
            # abstract).
            license="CC BY-NC-SA (paper; non-commercial)",
            loader=lambda cache_path: MiscLoader._load_yu2021_marine_pathogens_binary(cache_path),
            metadata={
                "full_name": "Marine Pathogen Identification via Raman Spectroscopy (binary: A. baumannii vs. P. nitritireducens)",
                "source": "https://github.com/zshicode/Raman-Spectra-Deep-Learning/blob/main/bin.csv",
                "paper": "https://doi.org/10.1021/acs.analchem.1c00431",
                "bibtex": "@article{Yu_2021, title={Analysis of Raman Spectra by Using Deep Learning Methods in the Identification of Marine Pathogens}, volume={93}, ISSN={1520-6882}, url={http://dx.doi.org/10.1021/acs.analchem.1c00431}, DOI={10.1021/acs.analchem.1c00431}, number={32}, journal={Analytical Chemistry}, publisher={American Chemical Society (ACS)}, author={Yu, Shixiang and Li, Xin and Lu, Weilai and Li, Hanfei and Fu, Y. Vincent and Liu, Fanghua}, year={2021}, month=aug, pages={11089--11098}}",
                "citation": [
                    "Yu, S., Li, X., Lu, W., Li, H., Fu, Y. V., & Liu, F. (2021). Analysis of Raman Spectra by Using Deep Learning Methods in the Identification of Marine Pathogens. Analytical Chemistry, 93(32), 11089-11098."
                ],
                # CITATION CONFIDENCE NOTE: this attribution is taken as given
                # from the rehosting repo's README (github.com/zshicode/
                # Raman-Spectra-Deep-Learning), which credits BOTH bin.csv and
                # multi.csv to Yu et al. 2021. It could NOT be independently
                # verified against the paper itself: the paper's own abstract
                # (confirmed via Semantic Scholar) describes only the 8-strain
                # Urechis unicinctus experiment (matching marine_pathogens/
                # multi.csv exactly) and makes no mention of Acinetobacter
                # baumannii or Pseudomonas nitritireducens. The paper's full
                # text (methods/SI, where a preliminary/pilot two-class
                # experiment might plausibly be described) could not be
                # accessed after real attempts: ACS page paywalled, the
                # paper's own institutional repository (ir.yic.ac.cn)
                # unreachable, CORE.ac.uk metadata-only (no full text), no PMC
                # deposit, no self-archived copy found on ResearchGate. No
                # independent trace of "Pseudomonas nitritireducens" combined
                # with Raman spectroscopy was found anywhere else either.
                # Bottom line: this citation reflects the source repo's stated
                # attribution, not an independently confirmed one -- treat it
                # with correspondingly lower confidence than marine_pathogens.
                "description": "261 Raman spectra (1200 points each, 600-1800 cm⁻¹) from two marine microbe species, Acinetobacter baumannii (label 0, 150 spectra) and Pseudomonas nitritireducens (label 1, 111 spectra), per the rehosting repo's README. Spectra are already 0-1 (min-max) normalized in the source CSV. The rehosting repo attributes this file to the same Yu et al. 2021 paper as marine_pathogens/multi.csv, but that attribution is UNVERIFIED -- the paper's own abstract only describes the 8-strain Urechis unicinctus experiment and does not mention either species named here; see the citation-confidence note above for the access attempts made. Rehosted (no header row, trailing integer label column, no sample/replicate IDs, no explicit wavenumber axis) as flat CSV by github.com/zshicode/Raman-Spectra-Deep-Learning, itself unlicensed -- see license note above.",
            },
            # Source-applied 0-1 (min-max) normalization per rehosting repo's README.
            embedded_preprocessing="0-1 (min-max) normalization (source-applied)",
            # NOT checked, same reasoning as marine_pathogens: no sample/
            # isolate/replicate ID column in the CSV, and the paper's full
            # text (which might describe replicate counts, IF this data is
            # even from this paper -- see citation-confidence note above)
            # could not be accessed. Left as None (unknown), not assumed
            # False -- do not infer grouping from the class counts (150/111)
            # alone.
            is_grouped=None,
            # Checked: no missing (NaN) label values (labels are a dense
            # 0-1 integer column with no gaps).
            has_missing_labels=False,
        ),
        "mlrod": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="mlrod",
            name="ML Raman Open Dataset (MLROD)",
            short_name="MLROD",
            # ODR states "License: BY-NC", without a version.
            license="CC BY-NC",
            loader=lambda cache_path: MiscLoader._load_mlrod(cache_path),
            metadata={
                "full_name": "ML Raman Open Dataset (MLROD)",
                "source": "https://odr.io/MLROD",
                "paper": "https://doi.org/10.1029/2021EA002125",
                "bibtex": "@article{Berlanga_2022, title={Convolutional Neural Networks as a Tool for Raman Spectral Mineral Classification Under Low Signal, Dusty Mars Conditions}, volume={9}, ISSN={2333-5084}, url={http://dx.doi.org/10.1029/2021EA002125}, DOI={10.1029/2021EA002125}, number={10}, journal={Earth and Space Science}, publisher={American Geophysical Union (AGU)}, author={Berlanga, Genesis and Williams, Quentin and Temiquel, Nathan}, year={2022}, month=oct}",
                "citation": [
                    "Wiens, R. C., et al. (2021). Convolutional neural networks as a tool for Raman spectral mineral classification under low signal, dusty Mars conditions. Earth and Space Science, 8, e2021EA002125."
                ],
                "description": "500,000+ point-mapping Raman spectra of Mars-analogue geological samples acquired on a Horiba LabRAM HR Evolution (532 nm, 100 mW, 1800 l/mm grating, ~1 cm⁻¹ resolution, open-electrode CCD at −60 °C, 50× LWD NA 0.5 objective, 0.1 µm XY step, 100–1800 cm⁻¹). Samples include gabbro (Madagascar), granite, Hawaiian basalt dust, 12 pure reference minerals, and 4 binary mixtures, measured at 0% and 50% basalt dust surface coverage. No cosmic ray removal, baseline correction, or other preprocessing applied. Motivated by NASA Perseverance SuperCam mineral identification.",
            },
            # Checked: no missing (NaN) label values.
            has_missing_labels=False,
        ),
        **{
            f"synthetic_organic_pigments_{process.lower().replace(' ', '_')}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.MaterialScience,
                id=f"synthetic_organic_pigments_{process.lower().replace(' ', '_')}",
                name=f"Synthetic Organic Pigments ({process})",
                short_name=f"Org. Pigments ({'Raw' if process == 'Raw' else 'Baseline'})",
                license="research use only",
                loader=lambda cache_path, p=process.lower().replace(' ', '_'): MiscLoader._load_sop_spectral_library(cache_path,
                                                                                variant=p),
                metadata={
                    "full_name": f"Synthetic Organic Pigments Raman Spectral Library - {process}",
                    "source": "https://kikirpa-my.sharepoint.com/:u:/g/personal/wim_fremout_kikirpa_be/ES5_J9PpBatLvbTe6VlFyIoBc6fFRli0YHl2qjnLxn6I8Q?download=1",
                    "paper": "https://doi.org/10.1002/jrs.4054",
                    "bibtex": "@article{Fremout_2012, title={Identification of synthetic organic pigments: the role of a comprehensive digital Raman spectral library}, volume={43}, ISSN={1097-4555}, url={http://dx.doi.org/10.1002/jrs.4054}, DOI={10.1002/jrs.4054}, number={11}, journal={Journal of Raman Spectroscopy}, publisher={Wiley}, author={Fremout, Wim and Saverwyns, Steven}, year={2012}, month=jul, pages={1536--1544}}",
                    "citation": [
                        'Fremout, Wim, and Steven Saverwyns. "Identification of synthetic organic pigments: the role of a comprehensive digital Raman spectral library." Journal of Raman Spectroscopy 43.11 (2012): 1536-1544.'
                    ],
                    "description": f"{process} Raman spectral library of approximately 270 synthetic organic pigments (~300 reference spectra, nearly 400 samples total) compiled at the Royal Institute for Cultural Heritage (KIK/IRPA), Brussels. Acquired on a Renishaw inVia dispersive Raman spectrometer (785 nm, Peltier-cooled CCD 203 K, 1200 mm⁻¹ grating, 50× LWD objective, ±0.5 cm⁻¹ calibration). Pigment classes cover azo (monoazo, disazo), heterocyclic, and polycyclic types (phthalocyanine, quinacridone, perylene, anthraquinone, dioxazine, triarylcarbonium, diketopyrrole). Validated for pigment identification on four Stedelijk Museum (Amsterdam) paintings. {'No baseline correction applied; raw fluorescence background retained.' if process == 'Raw' else 'Baseline correction applied to remove fluorescence background.'}",
                },
                # Checked (Raw variant only, via
                # raman_bench.splitting.infer_group_ids_from_targets): no
                # replicate structure found. Baseline Corrected variant not
                # checked -- same underlying spectra so *likely* the same
                # answer, but left as None (unknown) rather than assumed.
                is_grouped=(False if process == "Raw" else None),
                # Checked (Raw variant only): no missing (NaN) target values.
                # Baseline Corrected not checked -- left as None, not assumed.
                has_missing_labels=(False if process == "Raw" else None),
            )
            for process in ["Raw", "Baseline Corrected"]
        },
        "raman_spi": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="raman_spi",
            name="RaSPI: Raman Spectra for Plastics Identification",
            short_name="RaSPI Plastics",
            license="CC BY 4.0",
            loader=lambda cache_path: MiscLoader._load_raman_spi(cache_path),
            metadata={
                "full_name": "Raman Spectra for Plastics Identification (RaSPI)",
                "source": "https://doi.org/10.5683/SP3/8UQQQN",
                "paper": "https://doi.org/10.1038/s41597-026-07103-8",
                "bibtex": "@article{Hogan_2026, title={Raman spectra for plastics identification (RaSPI) and Raman maps for plastics identification (RaMPI) datasets}, volume={13}, ISSN={2052-4463}, url={http://dx.doi.org/10.1038/s41597-026-07103-8}, DOI={10.1038/s41597-026-07103-8}, number={1}, journal={Scientific Data}, publisher={Springer Nature}, author={Hogan, Una E. and Voss, H. Ben and Lei, Benjamin and Bec, Avery E. and Feng, Xinyi and Smith, Rodney}, year={2026}, month=mar, pages={765}}",
                "citation": [
                    "Hogan, Úna E., Voss, H. Ben., Lei, Benjamin, Bec, Avery E., Feng, Xinyi, Smith, Rodney (2026). Raman spectra for plastics identification (RaSPI) and Raman maps for plastics identification (RaMPI) datasets. Scientific Data, 13, 765."
                ],
                "description": "Baseline-corrected Raman spectra of 14 plastic types (polyethylene, polypropylene, PTFE, polystyrene, nylon, silicone, PMMA, POM, polyester, polyurethane, PVC, PET, polycarbonate, ABS) acquired on a Renishaw inVia Raman microscope: 226 spectra at 633 nm (1800 l/mm grating) and 176 at 532 nm (2400 l/mm grating). The deposited RaSPI_processed.csv is uniformly gridded at 1 cm^-1 over 225-3949 cm^-1 (3725 points), polynomial-baseline-subtracted and 0-1 normalized. Samples mix pristine commercial/retail plastics with environmental microplastics (bottle caps, bags, fragments) from Pollution Probe Great Lakes Seabin collections. The deposit's spectra file holds 402 rows, but 6 metadata rows lack a spectrum, 4 spectra lack metadata, and PE1a/PE1b are stored twice, leaving a fully-labeled usable set of 396 spectra from 272 physical samples (159 measured once, 104 twice, 9 three to four times). Each spectrum ID is '<plastic><number>' plus a trailing letter (PE1a, PE1b, PET4c...) marking repeated measurements of the same physical sample; the loader exposes those sample ids as group_ids. The raw (unprocessed) spectra are available in the same deposit as RaSPI_raw.csv. Same lab and much of the authorship as Lei et al. 2022 (the microplastics_training_* keys; Smith group, U. Waterloo -- Hogan and Lei co-authored both papers): the commercial-pristine subsets may draw on the same lab collection, though the deposits carry no shared sample ids.",
            },
            # Checked: the trailing-letter sample ids in RaSPI_metadata.csv
            # denote repeated measurements of the same physical plastic
            # sample (deposit README 2.2), so spectra sharing a sample stem
            # are physical replicates and must not be split across folds.
            is_grouped=True,
            # Checked: after the inner join of spectra and metadata, every
            # row has a plastic type.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Authors' pipeline (deposit README 4.2): polynomial baseline "
                "subtraction, truncation to 225-3949 cm^-1, interpolation "
                "onto a uniform 1 cm^-1 grid, per-spectrum 0-1 normalization."
            ),
        ),
        "raman_mpi": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="raman_mpi",
            name="RaMPI: Raman Maps for Plastics Identification",
            short_name="RaMPI Maps",
            license="CC BY 4.0",
            loader=lambda cache_path: MiscLoader._load_rampi(cache_path),
            metadata={
                "full_name": "Raman Maps for Plastics Identification (RaMPI)",
                "source": "https://doi.org/10.5683/SP3/8UQQQN",
                "paper": "https://doi.org/10.1038/s41597-026-07103-8",
                "bibtex": "@article{Hogan_2026, title={Raman spectra for plastics identification (RaSPI) and Raman maps for plastics identification (RaMPI) datasets}, volume={13}, ISSN={2052-4463}, url={http://dx.doi.org/10.1038/s41597-026-07103-8}, DOI={10.1038/s41597-026-07103-8}, number={1}, journal={Scientific Data}, publisher={Springer Nature}, author={Hogan, Una E. and Voss, H. Ben and Lei, Benjamin and Bec, Avery E. and Feng, Xinyi and Smith, Rodney}, year={2026}, month=mar, pages={765}}",
                "citation": [
                    "Hogan, Úna E., Voss, H. Ben., Lei, Benjamin, Bec, Avery E., Feng, Xinyi, Smith, Rodney (2026). Raman spectra for plastics identification (RaSPI) and Raman maps for plastics identification (RaMPI) datasets. Scientific Data, 13, 765."
                ],
                "description": "Per-pixel Raman spectra from 34 maps (30 pristine two-material blend maps plus 4 environmental maps of Great Lakes Seabin microplastic samples) acquired on a Renishaw inVia at 633 nm (1800 l/mm grating). Each of the 33,119 pixel spectra was manually assigned by the authors to one of 14 plastic types or 'blank'; this dataset holds the 16,598 plastic-assigned pixels. The native per-map axes (1,017 points, ~710-1,827 cm^-1, non-uniform spacing) are linearly resampled onto a common 1 cm^-1 grid (710-1,827 cm^-1, 1,118 points). Spectra are raw as exported from Renishaw WiRE -- no baseline correction or normalization. Maps per class: polyethylene 8, polyoxymethylene 7, ABS/PMMA/PS 6, nylon/PP/PTFE/PVC 5, polyester/PET 4, polycarbonate 3, polyurethane 2, silicone 2 -- the last two fall below 3 maps and cannot be split into grouped 3-fold CV by map. group_ids = map: pixels within one map are spatially correlated and must not be split across train/test. The companion 1D point-spectra dataset is raman_spi.",
            },
            # Checked: each Assig file is one Raman map; pixels within a map
            # are spatially adjacent measurements and highly correlated
            # (shared background, crosstalk), so maps are kept together.
            # Note: groups are NOT label-pure -- a mixed-material map contains
            # several plastic classes by design.
            is_grouped=True,
            # Checked: every kept row carries one of the 14 plastic labels
            # ('blank' pixels are excluded, not NaN).
            has_missing_labels=False,
            # Raw map spectra as exported from Renishaw WiRE; no baseline
            # correction, smoothing, or normalization applied.
            embedded_preprocessing=None,
        ),
        **{
            f"microplastics_training_{res}": DatasetInfo(
                task_type=TASK_TYPE.Classification,
                application_type=APPLICATION_TYPE.MaterialScience,
                id=f"microplastics_training_{res}",
                name=f"Microplastic Training Set ({res} cm^-1, Lei et al. 2022)",
                short_name=f"Microplastics {res}",
                license="CC BY 4.0",
                loader=lambda cache_path, r=res: MiscLoader._load_lei_microplastics(cache_path, r),
                metadata={
                    "full_name": f"Microplastic Training Set, {res} cm^-1 grid (Lei et al. 2022)",
                    "source": "https://doi.org/10.5683/SP3/LSN0R0",
                    "paper": "https://doi.org/10.1021/acs.analchem.2c02451",
                    "bibtex": "@article{Lei_2022, title={Customizable Machine-Learning Models for Rapid Microplastic Identification Using Raman Microscopy}, volume={94}, ISSN={0003-2700}, url={http://dx.doi.org/10.1021/acs.analchem.2c02451}, DOI={10.1021/acs.analchem.2c02451}, number={49}, journal={Analytical Chemistry}, publisher={American Chemical Society}, author={Lei, Benjamin and Bissonnette, Justine R. and Hogan, Una E. and Bec, Avery E. and Feng, Xinyi and Smith, Rodney D. L.}, year={2022}, month=dec, pages={17011-17019}}",
                    "citation": [
                        "Lei, B., Bissonnette, J. R., Hogan, U. E., Bec, A. E., Feng, X.-Y., and Smith, R. D. L. (2022). Customizable machine-learning models for rapid microplastic identification using Raman microscopy. Analytical Chemistry, 94(49), 17011-17019."
                    ],
                    "description": (
                        "AUGMENTED dataset -- the effective sample size is "
                        "186 measured spectra (108 supplier samples), not "
                        "3,720: every deposited row is a synthetic variant, "
                        "not the measured spectrum. The 3,720 non-blank rows "
                        "(of 4,520 deposited -- the 800 synthetically "
                        "generated blank spectra are excluded) are 20 "
                        "noise/scaling variants of each of 186 measurements "
                        "of 14 plastic types of commercial microplastic "
                        "particles (spheres, pellets, fragments, films; 15 "
                        "suppliers) on a Renishaw inVia Reflex (532 nm / "
                        "2,400 l/mm or 633 nm / 1,800 l/mm grating). "
                        "The authors applied polynomial baseline subtraction "
                        "and cosmic-ray removal, then a 20x augmentation "
                        "protocol per source spectrum (random noise targeting "
                        "an SNR of 3-15 on the strongest peak, amplitude "
                        "normalization so the maximum falls in 0.3-1.0). "
                        "microplastics_training_clean holds one representative "
                        "per measurement (186 spectra, 1 cm^-1 grid). Same "
                        "lab and much of the authorship as raman_spi/raman_mpi "
                        "(Smith group, U. Waterloo). "
                        "Spectra span the 720-1,800 cm^-1 fingerprint window, "
                        "pre-interpolated onto a uniform "
                        f"{res} cm^-1 grid ({1080 // int(res[:-2]) + 1} points). "
                        "The four sibling keys microplastics_training_1cm/2cm/"
                        "4cm/8cm hold the identical spectra at four "
                        "resolutions -- a built-in instrument-resolution-shift "
                        "test (the paper trains at 1 cm^-1 and evaluates at "
                        "2/4/8 cm^-1). group_ids = supplier sample id: all "
                        "20-40 augmented variants of one measurement share a "
                        "group, and the 108 groups are label-pure."
                    ),
                },
                # Checked: 108 unique supplier sample ids, each carrying
                # exactly one plastic label; all 20-40 augmented rows of a
                # sample must stay together across splits.
                is_grouped=True,
                # Checked: every non-blank row carries one of the 14 plastic
                # labels; the 800 blank rows are excluded, not NaN.
                has_missing_labels=False,
                embedded_preprocessing=(
                    "Authors' pipeline (paper + deposit README): Renishaw "
                    "WiRE 5.3 polynomial baseline subtraction and cosmic-ray "
                    "removal; each of the 186 source spectra then augmented "
                    "20x (random noise, strongest-peak SNR 3-15; amplitude "
                    "normalized so the maximum falls in 0.3-1.0); window "
                    f"truncated to 720-1800 cm^-1, pre-interpolated onto a "
                    f"uniform {res} cm^-1 grid."
                ),
            )
            for res in ("1cm", "2cm", "4cm", "8cm")
        },
        "microplastics_training_clean": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="microplastics_training_clean",
            name="Microplastic Training Set, 186 Measurements (Lei et al. 2022)",
            short_name="Microplastics Clean",
            license="CC BY 4.0",
            loader=lambda cache_path: MiscLoader._load_lei_microplastics_clean(cache_path),
            metadata={
                "full_name": "Microplastic Training Set, one spectrum per measurement (Lei et al. 2022)",
                "source": "https://doi.org/10.5683/SP3/LSN0R0",
                "paper": "https://doi.org/10.1021/acs.analchem.2c02451",
                "bibtex": "@article{Lei_2022, title={Customizable Machine-Learning Models for Rapid Microplastic Identification Using Raman Microscopy}, volume={94}, ISSN={0003-2700}, url={http://dx.doi.org/10.1021/acs.analchem.2c02451}, DOI={10.1021/acs.analchem.2c02451}, number={49}, journal={Analytical Chemistry}, publisher={American Chemical Society}, author={Lei, Benjamin and Bissonnette, Justine R. and Hogan, Una E. and Bec, Avery E. and Feng, Xinyi and Smith, Rodney D. L.}, year={2022}, month=dec, pages={17011-17019}}",
                "citation": [
                    "Lei, B., Bissonnette, J. R., Hogan, U. E., Bec, A. E., Feng, X.-Y., and Smith, R. D. L. (2022). Customizable machine-learning models for rapid microplastic identification using Raman microscopy. Analytical Chemistry, 94(49), 17011-17019."
                ],
                "description": (
                    "The un-augmented core of the Lei et al. 2022 "
                    "microplastic training set: one representative spectrum "
                    "per source measurement -- 186 measurements of 14 "
                    "plastic types from 108 supplier samples (78 samples "
                    "measured twice) of commercial microplastic particles "
                    "(spheres, pellets, fragments, films; 15 suppliers) on "
                    "a Renishaw inVia Reflex (532 nm / 2,400 l/mm or "
                    "633 nm / 1,800 l/mm grating). Each representative is "
                    "the row-wise mean of the 20 augmented variants the "
                    "authors deposited per measurement: averaging cancels "
                    "the augmentation's random noise (strongest-peak SNR "
                    "3-15) and leaves the measured spectral shape (the "
                    "absolute amplitude is arbitrary, as for all variants "
                    "of this deposit -- none of the deposited rows is the "
                    "raw measurement). 720-1,800 cm^-1 fingerprint window, "
                    "pre-interpolated onto the paper's native 1 cm^-1 "
                    "training grid (1,081 points). group_ids = supplier "
                    "sample id (label-pure; the two measurements of a "
                    "re-measured sample share a group). The augmented full "
                    "sets at four resolutions are "
                    "microplastics_training_1cm/2cm/4cm/8cm. Same lab and "
                    "much of the authorship as raman_spi/raman_mpi (Smith "
                    "group, U. Waterloo)."
                ),
            },
            # Checked: the deposited rows are laid out in 20-row blocks,
            # one per measurement (non-blank sampleid runs of 20 or 40
            # rows); 186 blocks and 108 supplier sample ids are asserted
            # at load time.
            is_grouped=True,
            # Checked: every block carries exactly one plastic label.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Authors' pipeline (paper + deposit README): Renishaw "
                "WiRE 5.3 polynomial baseline subtraction and cosmic-ray "
                "removal; window truncated to 720-1800 cm^-1, "
                "pre-interpolated onto a uniform 1 cm^-1 grid. This key "
                "additionally reduces each measurement to the row-wise "
                "mean of its 20 augmented variants (see description)."
            ),
        ),
        "yeast_bioreactor_fermentation": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Biological,
            id="10.4121/4fbcf35e-1954-48a4-a2a3-693e19c753e7",
            name="Yeast Bioreactor Fermentation (Klaverdijk et al. 2025)",
            short_name="Yeast Bioreactor",
            license="CC BY 4.0",
            loader=lambda cache_path: MiscLoader._load_yeast_bioreactor_fermentation(cache_path),
            metadata={
                "full_name": (
                    "Raman Monitoring of S. cerevisiae Bioreactor "
                    "Fermentations (Klaverdijk et al. 2025)"
                ),
                "source": "https://doi.org/10.4121/4fbcf35e-1954-48a4-a2a3-693e19c753e7",
                "paper": "https://doi.org/10.1007/s00216-025-05768-5",
                "bibtex": (
                    "@article{Klaverdijk_2025, title={Single compound data "
                    "supplementation to enhance transferability of "
                    "fermentation specific Raman spectroscopy models}, "
                    "volume={417}, ISSN={1618-2841}, url={http://dx.doi.org/"
                    "10.1007/s00216-025-05768-5}, DOI={10.1007/"
                    "s00216-025-05768-5}, number={9}, journal={Analytical "
                    "and Bioanalytical Chemistry}, publisher={Springer "
                    "Science and Business Media LLC}, author={Klaverdijk, "
                    "Maarten and Ottens, Marcel and Klijn, Marieke E.}, "
                    "year={2025}, month=feb, pages={1873}}"
                ),
                "citation": [
                    "Klaverdijk, M., Ottens, M., and Klijn, M. E. (2025). "
                    "Single compound data supplementation to enhance "
                    "transferability of fermentation specific Raman "
                    "spectroscopy models. Analytical and Bioanalytical "
                    "Chemistry, 417(9), 1873. "
                    "https://doi.org/10.1007/s00216-025-05768-5"
                ],
                "description": (
                    "Inline Raman monitoring of S. cerevisiae batch and "
                    "fed-batch bioreactor fermentations for 3-target "
                    "regression of process variables (Glucose [mM], "
                    "Ethanol [mM], Biomass [g/L]): 65 in-line measurements "
                    "from 5 fermentations -- 3 batch cultivations used as "
                    "the paper's calibration set (B1: 12, B2: 13, B3: 13 "
                    "samples), 1 unseen batch cultivation for validation "
                    "(B4: 13), and 1 fed-batch cultivation for transfer "
                    "testing (FB: 14). Kaiser Raman RXN2 analyzer with a "
                    "785 nm laser and a fiber-optic bIO-Optic immersion "
                    "probe mounted through the bioreactor head plate; each "
                    "measurement is a single 60 s acquisition (the deposit "
                    "also holds 2/4/6/8/10-min stackings of the same 60 s "
                    "blocks, which are excluded here; the paper calibrated "
                    "all models on the 1-min spectra). 100-3,425 cm^-1 at "
                    "1 cm^-1 (3,326 points; one common grid for all "
                    "samples). group_ids are the 5 fermentations (B1, B2, "
                    "B3, B4, FB): samples of one cultivation are strongly "
                    "correlated time points of the same process and must "
                    "not be split across train/test (leave-one-batch-out "
                    "is the paper's validation protocol). 5 glucose "
                    "reference values (HPLC) failed in the base set and "
                    "are NaN; all other 190 target values are complete. "
                    "Spectra are raw instrument output as deposited (strong "
                    "fluorescence background, no baseline correction or "
                    "normalization). The same 4TU deposit's other chapters "
                    "hold single-compound, synthetic, and augmented spectra "
                    "for transfer-learning studies."
                ),
            },
            # Checked: samples within one fermentation (B1..B4, FB) are
            # time points of the same cultivation -- strongly correlated,
            # so a batch must stay on one side of a split.
            is_grouped=True,
            # Checked: 5 glucose values are NaN (failed HPLC reference,
            # marked in the deposit); the other 2 targets are complete.
            has_missing_labels=True,
            # Raw spectra as deposited (Chapter 2 base/validation/test);
            # the AirPLS-processed variants live in Chapter 3 of the
            # deposit, not in the files used here.
            embedded_preprocessing=None,
        ),
        "nist_ecsers_drugs": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.Chemical,
            id="nist_ecsers_drugs",
            name="NIST EC-SERS Drug & Adulterant Library (Ott 2024)",
            short_name="NIST EC-SERS Drugs",
            license="NIST public data (fair use)",
            loader=lambda cache_path: MiscLoader._load_nist_ecsers_drugs(cache_path),
            metadata={
                "full_name": (
                    "EC-SERS (Electrochemical Surface-Enhanced Raman "
                    "Spectroscopy) Spectral Database of Drug and "
                    "Adulterant Compounds"
                ),
                "source": "https://doi.org/10.18434/mds2-3502",
                "paper": "https://doi.org/10.1016/j.forc.2024.100621",
                "bibtex": "@dataset{ott2024ecsers, title={EC-SERS (Electrochemical Surface-Enhanced Raman Spectroscopy) Spectral Database of Drug and Adulterant Compounds}, author={Ott, Colby E.}, year={2024}, publisher={National Institute of Standards and Technology (NIST) Public Data Resource}, DOI={10.18434/mds2-3502}, url={https://doi.org/10.18434/mds2-3502}}",
                "citation": [
                    "Ott, C. E. (2024). EC-SERS (Electrochemical Surface-Enhanced Raman Spectroscopy) Spectral Database of Drug and Adulterant Compounds. NIST Public Data Resource. https://doi.org/10.18434/mds2-3502",
                    "Ott, C. E., Wilcox, A., Kalb, S., McConnell, A., Sisco, E., and Arroyo, L. E. (2024). Development and evaluation of a nontargeted electrochemical surface-enhanced Raman spectroscopy (EC-SERS) screening method applied to forensic seized drug casework samples. Forensic Chemistry, 41, 100621.",
                    "Ott, C. E., Perez-Estebanez, M., Hernandez, S., Kelly, K., Dalzell, K. A., Arcos-Martinez, M. J., Heras, A., Colina, A., and Arroyo, L. E. (2022). Forensic identification of fentanyl and its analogs by electrochemical-surface enhanced Raman spectroscopy (EC-SERS) for the screening of seized drugs of abuse. Frontiers in Analytical Science, 2, 834820.",
                ],
                "description": (
                    "EC-SERS (electrochemical surface-enhanced Raman "
                    "spectroscopy) spectral library of 32 drugs of abuse "
                    "and adulterant compounds from the NIST Public Data "
                    "Resource (Ott, NIST, 2024; "
                    "doi:10.18434/mds2-3502, v1.0.1). 131 spectra, 785 nm "
                    "excitation on a DropSens SPELEC silver electrode; "
                    "each deposited CSV is the highest-intensity / "
                    "best-resolution spectrum of one replicate "
                    "acquisition. The deposit's raw potential-resolved "
                    ".exp files (118) and combined Excel sheet are not "
                    "loaded here. Values are 'Counts minus Dark (a.u.)' "
                    "-- NOT normalized: every file has slightly negative "
                    "points (dark-subtraction residuals, per-file minima "
                    "-574..-16) and per-file maxima span 5.1e4-2.0e5, so "
                    "expect the substrate/potential-dependent intensity "
                    "drift typical of EC-SERS (realistic but not 'plain "
                    "Raman'). Two acquisition batches are mixed: 96 "
                    "files on a 1,014-point grid (32.6-2,859.1 cm^-1) "
                    "and 35 on a 1,011-point grid (33.8-2,995.5 cm^-1), "
                    "both non-uniformly spaced (1.95-4.18 cm^-1 steps); "
                    "the loader interpolates all spectra onto the common "
                    "1,013-point grid (36.6-2,859.1 cm^-1) that every "
                    "file fully covers. 126 of the 131 spectra carry one "
                    "of 11 concentration labels (1, 5, 10, 25, 50, 100, "
                    "300, 333, 1,000, 1,001, 1,013 ppm) -- a built-in "
                    "concentration / detection-limit subtask; the 5 "
                    "unlabelled ones are solid-form procaine (3 "
                    "replicates) and pure-powder acryl- and "
                    "furanyl-fentanyl. Substances: alprazolam, cocaine, "
                    "codeine, dextromethorphan, diphenhydramine, "
                    "etizolam, etonitazene, fentanyl, heroin, "
                    "levamisole, lidocaine, methadone, methamphetamine, "
                    "morphine, nicotine, procaine (solid), quinine, "
                    "sufentanil, temazepam, xylazine, 4-ANPP, "
                    "buprenorphine, diltiazem, hydroxyzine, naltrexone, "
                    "and six fentanyl analogs (acetyl, acryl, furanyl, "
                    "methoxyacetyl, valeryl); the deposit's "
                    "33-compound analyte panel also lists mitragynine, "
                    "for which no CSV is deposited. Mostly the "
                    "nontargeted campaign (Ott et al. 2024, "
                    "doi:10.1016/j.forc.2024.100621); six further "
                    "spectra (spectrum numbers 292-326: the five "
                    "fentanyl analogs plus 4-ANPP) come from the "
                    "targeted method campaign (Ott et al. 2022, "
                    "doi:10.3389/frans.2022.834820). group_ids = "
                    "(substance, concentration, replicate): 118 groups "
                    "of 1-3 spectra (106 single-spectrum), and all 24 "
                    "measurements without a replicate marker at one "
                    "substance+concentration share a single group "
                    "(possibly the same acquisition session). 10 "
                    "substances have fewer than 3 groups, so grouped "
                    "k-fold CV only covers a subset of the library. "
                    "Filenames are '<Substance>[_<conc>ppm][_form_solid]"
                    "[_nontarget][_rep<N>]_Spectrum_<n>.csv'; the "
                    "deposit contains one 'Spectrum 293' with a space, "
                    "one 'nontargeT' typo and one file without a target "
                    "marker, all parsed by _nist_ecsers_parse_name."
                ),
            },
            # Checked: 118 (substance, concentration, replicate) groups
            # from all 131 filenames; the 24 measurements without a
            # replicate marker at one substance+concentration share one
            # group (possibly the same acquisition session), so no
            # correlated spectra cross a fold.
            is_grouped=True,
            # Checked: all 131 spectra carry a substance label parsed
            # from the filename.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Authors' pipeline (deposit README): modified-polynomial "
                "baseline correction in DropView on the raw "
                "potential-resolved EC-SERS scans, then the highest-"
                "intensity / best-resolution spectrum per replicate "
                "exported as CSV (Origin Pro). The CSVs are dark-"
                "subtracted counts (a.u.), NOT intensity-normalized. "
                "This loader additionally interpolates the two native "
                "grids onto the common 1,013-point grid (see "
                "description)."
            ),
        ),
        "rod_minerals": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="rod_minerals",
            name="Raman Open Database Minerals (ROD, 2026-10-20 snapshot)",
            short_name="ROD Minerals",
            license="Public domain (CC0)",
            loader=lambda cache_path: MiscLoader._load_rod_minerals(cache_path),
            metadata={
                "full_name": (
                    "Raman Open Database (ROD): experimental mineral Raman "
                    "spectra (frozen 2026-10-20 snapshot)"
                ),
                "source": "https://solsa.crystallography.net/rod/",
                "paper": "https://doi.org/10.1107/S1600576719004229",
                "bibtex": "@article{ElMendili_2019, title={Raman Open Database: first interconnected Raman--X-ray diffraction open-access resource for material identification}, volume={52}, ISSN={1600-5767}, url={http://dx.doi.org/10.1107/S1600576719004229}, DOI={10.1107/S1600576719004229}, number={3}, journal={Journal of Applied Crystallography}, publisher={International Union of Crystallography}, author={El Mendili, Yassine and Vaitkus, Antanas and Merkys, Andrius and Gra{\\=z}ulis, Saulius and Chateigner, Daniel and Mathevet, Fabrice and Gascoin, St{\'e}phanie and Petit, Sebastien}, year={2019}, month=may, pages={618-625}}",
                "citation": [
                    "El Mendili, Y., Vaitkus, A., Merkys, A., Gra\u017culis, S., Chateigner, D., Mathevet, F., Gascoin, S., and Petit, S. (2019). Raman Open Database: first interconnected Raman--X-ray diffraction open-access resource for material identification. Journal of Applied Crystallography, 52(3), 618-625.",
                ],
                "description": (
                    "Experimental Raman spectra of minerals and related "
                    "phases from the Raman Open Database (ROD, SOLSA "
                    "project; El Mendili et al. 2019, "
                    "doi:10.1107/S1600576719004229), served as a frozen "
                    "snapshot of the continuously growing live database "
                    "(bulk result.php?format=zip download of 2026-10-20: "
                    "1,133 CIF entry files; mirrored in the raman_data "
                    "HuggingFace mirror repo and pinned by size and "
                    "sha256, provenance in the snapshot MANIFEST.txt). "
                    "1,098 spectra of 432 labeled mineral/phase "
                    "entries, resampled onto the common 250-1,200 "
                    "cm^-1 @1 cm^-1 grid (951 points) that 98.5% of "
                    "the snapshot's 1,115 experimental spectra fully "
                    "cover. Excluded from the snapshot: 18 entries "
                    "with only theoretical (DFT) spectra and the 17 "
                    "experimental spectra not fully covering the "
                    "common window. Labels use _chemical_name_mineral, "
                    "falling back to _chemical_name_systematic, then "
                    "_chemical_formula_sum (the .rod files carry "
                    "values on the same line, on the next line, or in "
                    "multi-line ';'-blocks, and the loader parses all "
                    "three forms), so the 432 labels mix IMA mineral "
                    "names with systematic names and formulas for "
                    "unassigned phases and solid solutions. 108 labels "
                    "have a single spectrum and 213 have two; most "
                    "frequent: Andradite (22), Corundum (17), "
                    "Anorthite (15). Excitation wavelengths: 532 nm "
                    "(483), 780 nm (408), 633 nm (102), 785 nm (58), "
                    "514 nm (31), 488 nm (15), 632.8 nm (1); "
                    "instruments dominated by Thermo Nicolet Almega XR "
                    "(795) and DXR (154). Intensities are in arbitrary "
                    "units (per-entry normalization is not documented); "
                    "no preprocessing is applied here beyond grid "
                    "resampling. No group_ids: every spectrum is its "
                    "own ROD entry, and grouping by label would put "
                    "each class wholly into one fold. Two pairs of "
                    "entries carry byte-identical spectra under "
                    "different labels (Bismutite/Corundum, "
                    "Corundum/Shcherbianite), as deposited. "
                    "Many entries overlap with the "
                    "rruff_mineral_raw/preprocess datasets (~234 of the "
                    "1,115 experimental entries, ~21%, per the 2026-10-20 "
                    "research -- the public-domain RRUFF spectra are "
                    "mirrored in both databases); de-duplicate before "
                    "training on one and testing on the other. "
                    "License: public domain -- all ROD data have been "
                    "placed in the public domain by the contributors."
                ),
            },
            # Checked: one spectrum per ROD entry, no replicate
            # structure; group = label would make every test class
            # unseen in training, so the dataset is ungrouped.
            is_grouped=False,
            # Checked: all 1,098 loaded spectra carry a mineral /
            # systematic / formula label (full CIF parse, so
            # next-line and multi-line ';'-block values are read too).
            has_missing_labels=False,
            embedded_preprocessing=(
                "None applied by this loader beyond resampling onto the "
                "common 250-1,200 cm^-1 @1 cm^-1 grid; the per-entry "
                "device-specific preprocessing (baseline correction, "
                "normalization) varies by source and is not documented "
                "per entry in the snapshot."
            ),
        ),
        "olive_sunflower_mixtures": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            application_type=APPLICATION_TYPE.Chemical,
            id="644b79f82ef3605eb5c660aa",
            name="Olive + Sunflower Oil Mixtures, EVOO % (Picouet 2023)",
            short_name="Olive Sunflower Mixtures",
            license="lov2 (Licence Ouverte)",
            loader=lambda cache_path: MiscLoader._load_olive_sunflower_mixtures(
                cache_path
            ),
            metadata={
                "full_name": (
                    "Raman Spectra of Olive and Sunflower oil mixtures "
                    "(Picouet 2023, data.gouv.fr / ENTREPOT Recherche)"
                ),
                "source": (
                    "https://www.data.gouv.fr/datasets/spectres-raman-de-"
                    "melanges-dhuiles-dolive-et-de-tournesol-raman-"
                    "spectra-of-olive-and-sunflower-oil-mixtures"
                ),
                "doi": "10.57745/DOGT0E",
                "bibtex": (
                    "@dataset{picouet2023raman, title={Spectres Raman de "
                    "mélanges d'Huiles d'Olive et de Tournesol / Raman "
                    "Spectra of Olive and Sunflower oil mixtures}, "
                    "author={Picouet, Pierre}, year={2023}, "
                    "publisher={data.gouv.fr (ENTREPOT Recherche)}, "
                    "doi={10.57745/DOGT0E}, "
                    "url={https://doi.org/10.57745/DOGT0E}}"
                ),
                "citation": [
                    "Picouet, P. (2023). Spectres Raman de mélanges "
                    "d'Huiles d'Olive et de Tournesol / Raman Spectra "
                    "of Olive and Sunflower oil mixtures. data.gouv.fr. "
                    "https://doi.org/10.57745/DOGT0E",
                ],
                "description": (
                    "605 Raman spectra of Extra Virgin Olive Oil "
                    "(EVOO) / sunflower oil mixtures at 26 EVOO "
                    "levels (0, 10, 15, 20, 21, 23, 25, 31, 35, 47, "
                    "48, 50, 54, 57, 63, 65, 71, 74, 75, 76, 77, 81, "
                    "83, 90, 91, 100 %; sunflower % = 100 - EVOO "
                    "exactly), recorded on 16 acquisition dates "
                    "(22/11/2022 - 14/02/2023) at GRAPPE, Ecole "
                    "Superieure des Agricultures / INRAE, Angers, "
                    "France (Picouet 2023). Instrument (per the "
                    "deposit README): Ocean Insight QE Pro Raman "
                    "spectrometer, 785 nm solid-state laser diode "
                    "at 400 mW, general-purpose fiber-optic probe "
                    "with a quartz cuvette holder (working distance "
                    "0.5 mm), 1000 ms integration, 10 scans per "
                    "acquisition, ~11 cm^-1 FWHM. Task: EVOO % "
                    "regression (single target evoo_percent, "
                    "0-100). Structure: by the deposit's design of "
                    "experiment, each (date, EVOO%) combination is "
                    "one physical blend measured with 5 spectra "
                    "(S1-S5); 10 blends were re-measured once and 1 "
                    "twice on the same day, giving 109 blend groups "
                    "(98 x 5, 10 x 10, 1 x 15 spectra) - group-aware "
                    "CV is leave-one-blend-out. The deposit's own "
                    "Type column tags the spectra CALIBRATION (350) "
                    "/ VALIDATION (165) / PREDICTION (90) (9 blends "
                    "carry mixed tags); all 605 spectra are shipped "
                    "- the tags are a modelling split, not a "
                    "data-quality partition, and remain available in "
                    "the raw CSV. Native 710-point axis "
                    "500.48-2,499.31 cm^-1 (2.2-3.6 cm^-1 non-"
                    "uniform spacing), read from the CSV header (the "
                    "deposit's wavenumber cells use European decimal "
                    "commas, e.g. 500,48 = 500.48) and used as is, "
                    "so no resampling is involved. Intensities are "
                    "raw counts (645-12,922, arbitrary units), no "
                    "NaN. The EVOO is a 100% Italian cold-pressed "
                    "extra virgin olive oil (trademark PUGET, "
                    "acidity 0.3 %a, 14 g/100 g unsaturated fat); "
                    "the sunflower oil is a refined Spanish oil "
                    "(trademark FAIGES, acidity 0.2 %, 10 g/100 g "
                    "unsaturated fat, 80 mg vitamin E). License: "
                    "licence ouverte (lov2), France's open data "
                    "license. The deposit's companion README PDF "
                    "(raman-spectra-readme.pdf, 113,945 bytes, sha1 "
                    "7f9a320bb5cd7d7683b6b756636f6b60e37e3481) "
                    "documents the acquisition protocol; the CSV is "
                    "served directly from the static.data.gouv.fr "
                    "CDN (sha256 pinned; no mirror needed, the "
                    "endpoint is reachable from CI)."
                ),
            },
            # Checked: each (date, EVOO%) combination is one physical
            # blend measured with 5 scans (S1-S5; 10 blends
            # re-measured once and 1 twice on the same day); 109
            # blend groups, so grouped CV never splits a blend.
            is_grouped=True,
            # Checked: all 605 rows carry an EVOO % value and
            # EVOO % + sunflower % sums to exactly 100.
            has_missing_labels=False,
            embedded_preprocessing=(
                "None. The CSV is the deposit's raw acquisition "
                "export; intensities are raw counts as recorded (each "
                "spectrum is one acquisition of 10 scans co-added by "
                "the instrument, per the deposit README). No baseline "
                "correction, normalization or resampling is applied "
                "here."
            ),
        ),
        "infra_art_raman": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="infra-art",
            name="INFRA-ART Heritage Spectral Library, Raman (Cortea 2023)",
            short_name="INFRA-ART Raman",
            license="CC BY-NC 4.0",
            loader=lambda cache_path: MiscLoader._load_infra_art_raman(
                cache_path
            ),
            metadata={
                "full_name": (
                    "INFRA-ART Spectral Library: Raman reference spectra "
                    "of art-related materials (Cortea et al. 2023, "
                    "INOE 2000; frozen 2026-10-21 snapshot)"
                ),
                "source": "https://infra-art.eu",
                "paper": "https://doi.org/10.1145/3593427",
                "bibtex": (
                    "@article{cortea2023infraart, title={INFRA-ART: An "
                    "Open Access Spectral Library of Art-related "
                    "Materials as a Digital Support Tool for Cultural "
                    "Heritage Science}, author={Cortea, Ioana Maria "
                    "and Chiro\\u0219c{\\u0103}, Alecsandru Vladimir "
                    "and Anghelu\\u0219t{\\u0103}, Lauren\\u021b "
                    "Marian and Seri\\u0219an, George Calin}, "
                    "journal={Journal on Computing and Cultural "
                    "Heritage}, volume={16}, number={2}, year={2023}, "
                    "pages={1--11}, doi={10.1145/3593427}}"
                ),
                "citation": [
                    "Cortea, I. M., ChiroscA, A. V., Anghelu\u0219t\u0103, "
                    "L. M., and Seri\u0219an, G. C. (2023). INFRA-ART: "
                    "An Open Access Spectral Library of Art-related "
                    "Materials as a Digital Support Tool for Cultural "
                    "Heritage Science. Journal on Computing and "
                    "Cultural Heritage, 16(2), 1-11. "
                    "https://doi.org/10.1145/3593427",
                ],
                "description": (
                    "Raman reference spectra of 186 art-related "
                    "materials from the INFRA-ART Spectral Library "
                    "(Cortea et al. 2023, doi:10.1145/3593427; "
                    "FAIRsharing doi:10.25504/FAIRsharing.091c24), "
                    "operated by the National Institute for Research "
                    "and Development in Optoelectronics (INOE 2000, "
                    "Bucharest, Romania). Frozen 2026-10-21 snapshot: "
                    "at capture time the library held 1,209 reference "
                    "samples (3,161 spectra across FTIR, XRF, Raman "
                    "and SWIR), of which 186 carry a Raman spectrum "
                    "(one per sample). Each spectrum is the original "
                    "Renishaw ENLIGHTEN CSV export as served by the "
                    "site (WP 785 ER spectrometer, 785.078 nm "
                    "excitation; per-measurement parameters - "
                    "integration time 0.1-10 s, scan averaging, ROI - "
                    "are recorded in the CSV metadata block), "
                    "acquired 2022-03-30 to 2025-06-06. Task: "
                    "material-class classification - labels are the "
                    "library's 12 material classes (Special effect "
                    "pigments 61, Earth pigments 43, Resins, balsams "
                    "and wax 23, Glues, gums and binders 19, "
                    "Inorganic pigments, synthetic 18, Organic dyes "
                    "and pigments, synthetic 5, Mineral pigments 4, "
                    "Carbon-based pigments 4, Mixed pigments 3, "
                    "Unclassified materials 3, Natural organic dyes "
                    "and pigments 2, Other mixtures 1); the "
                    "per-sample material names, formulas and sources "
                    "are in the snapshot manifest (5 names are shared "
                    "by two different samples). Native 1666-point "
                    "axis 160.07-3,215.39 cm^-1 (non-uniform "
                    "spacing, identical in all files), used as is - "
                    "no resampling. Intensities are the Processed "
                    "(dark-subtracted) column in arbitrary units; "
                    "the integration time varies per sample (0.1-10 "
                    "s), so absolute intensity scales are not "
                    "comparable across spectra and models should "
                    "rely on relative peak structure (a few points "
                    "go slightly negative, minimum -13). Two CSV "
                    "header variants occur (7 columns including "
                    "Raw/Dark/Reference, 122 files; 4 columns, 64 "
                    "files); the loader parses header-driven. Served "
                    "from a frozen zip snapshot (186 CSVs + "
                    "manifest.json + README.txt, size + sha256 "
                    "pinned) mirrored at HTW-KI-Werkstatt/RamanBench "
                    "(infra_art/): the live endpoint "
                    "(https://infra-art.eu/api/samples/spectra-"
                    "file?sampleId=<id>&dataType=Raman%20Spectrum) "
                    "requires full browser request headers (the site "
                    "is behind Cloudflare) and the library is still "
                    "growing, so the snapshot is the reproducible "
                    "reference. is_grouped=False (one spectrum per "
                    "physical sample, no replicate structure). "
                    "License: CC BY-NC 4.0 (as stated on the site; "
                    "non-commercial use only)."
                ),
            },
            # Checked: one spectrum per physical sample; no replicate
            # structure exists in the library, so there is nothing to
            # keep together in a group-aware split.
            is_grouped=False,
            # Checked: all 186 samples carry a material class (the
            # "Unclassified materials" class is an actual library
            # class, not a missing label).
            has_missing_labels=False,
            embedded_preprocessing=(
                "Dark subtraction is applied by the instrument "
                "(Processed = Raw - Dark columns; the export's "
                "'Raman Intensity Corrected' flag is FALSE and no "
                "baseline correction is recorded). No baseline "
                "correction, normalization or resampling is applied "
                "here; note the per-sample integration time varies "
                "from 0.1 to 10 s, so absolute intensity scales are "
                "not comparable across spectra."
            ),
        ),
        "ocean_polymers": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="ocean_polymers",
            name=(
                "Raman Reference Library of Potential "
                "Anthropogenic and Biological Ocean Polymers "
                "(Miller 2022)"
            ),
            short_name="Ocean Polymers",
            license="CC BY 4.0",
            loader=lambda cache_path: MiscLoader._load_miller_ocean_polymers(
                cache_path
            ),
            metadata={
                "full_name": (
                    "Post-processed Raman reference library of 79 "
                    "ocean polymer specimens (61 anthropogenic "
                    "polymers + 18 biological materials), "
                    "200-3,400 cm^-1 (Miller et al. 2022, "
                    "Scientific Data 9:780)"
                ),
                "source": "https://osf.io/7cqv4/",
                "paper": "https://doi.org/10.1038/s41597-022-01883-5",
                "bibtex": (
                    "@article{miller2022ocean, "
                    "title={A Raman spectral reference library of "
                    "potential anthropogenic and biological ocean "
                    "polymers}, "
                    "author={Miller, Emily A. and Yamahara, Kevan M. "
                    "and French, Chris and Spingarn, Neil E. and "
                    "Birch, James M. and Van Houtan, Kyle S.}, "
                    "journal={Scientific Data}, volume={9}, "
                    "pages={780}, year={2022}, "
                    "doi={10.1038/s41597-022-01883-5}}"
                ),
                "citation": [
                    "Miller, E. A., Yamahara, K. M., French, C., "
                    "Spingarn, N. E., Birch, J. M., and Van Houtan, "
                    "K. S. (2022). A Raman spectral reference "
                    "library of potential anthropogenic and "
                    "biological ocean polymers. Scientific Data, "
                    "9, 780. "
                    "https://doi.org/10.1038/s41597-022-01883-5 "
                    "(data: https://osf.io/7cqv4/)",
                ],
                "description": (
                    "Post-processed Raman spectra of 79 ocean "
                    "polymer specimens from the open-access "
                    "reference library of Miller et al. 2022 "
                    "(Scientific Data 9, 780, "
                    "doi:10.1038/s41597-022-01883-5; data: OSF "
                    "project https://osf.io/7cqv4/). The library "
                    "covers (a) 40 pristine anthropogenic "
                    "polymers newly sourced from manufacturers, "
                    "(b) 22 weathered anthropogenic polymers from "
                    "used consumer, beachcast, agricultural and "
                    "fishery sources, and (c) 17 biological "
                    "polymers representing diverse marine taxa, "
                    "trophic levels and tissues; the deposit "
                    "holds 18 BIOL-coded specimens (the 17 "
                    "biological materials plus a charcoal "
                    "reference, which is carbonaceous, not a "
                    "polymer). The 79 spectra (61 PLAS-coded + "
                    "18 BIOL-coded particles, one per specimen) "
                    "carry the authors' uniform 3,201-point "
                    "200-3,400 cm^-1 @1 cm^-1 grid and are "
                    "peak-normalized (max = 1.0, min = 0) as "
                    "deposited. The deposit's long-format CSV "
                    "(Data2_processed.csv, one row per "
                    "spectrum-wavenumber point) is pivoted here; "
                    "no further preprocessing is applied. Task: "
                    "classification by the deposit's poly_lab "
                    "label - 59 classes, of which 12 have "
                    "replicates (bone x4, PETE x4, PP x4, PS x3, "
                    "chitin x3, cellulose x3, HDPE x2, "
                    "myofibrillar protein x2, PA x2, PE x2, "
                    "calcium carbonate x2) and 47 are singleton "
                    "reference entries, so most classes are "
                    "single-spectrum references; the coarse "
                    "plastic-vs-biological split (61 vs 18) is "
                    "derivable from the labels. is_grouped=False "
                    "(one spectrum per specimen). The deposit's "
                    "raw measurement files (54 distinct spectra, "
                    "532/780 nm excitation) "
                    "are served separately as the "
                    "ocean_polymers_raw key. License note: the "
                    "OSF project page sets no explicit license; "
                    "the open-access parent paper (CC BY 4.0) "
                    "publishes these data records in its Data "
                    "Records section, which is the license used "
                    "here."
                ),
            },
            # Checked: one post-processed spectrum per specimen
            # (79 IDs, 79 particles); no replicate structure.
            is_grouped=False,
            # Checked: every ID block in the pinned CSV carries a
            # complete, finite 3,201-point spectrum and a label.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Peak normalization to max = 1.0 (min = 0) applied "
                "by the authors as deposited; the post-processed "
                "grid is a uniform 200-3,400 cm^-1 @1 cm^-1 axis "
                "(3,201 points). The loader only pivots the "
                "long-format CSV into wide spectra; no baseline "
                "correction, resampling or further normalization "
                "is applied."
            ),
        ),
        "ocean_polymers_raw": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            application_type=APPLICATION_TYPE.MaterialScience,
            id="ocean_polymers_raw",
            name=(
                "Ocean Polymers Raw Measurements, 532/780 nm "
                "(Miller 2022)"
            ),
            short_name="Ocean Polymers Raw",
            license="CC BY 4.0",
            loader=lambda cache_path: (
                MiscLoader._load_miller_ocean_polymers_raw(cache_path)
            ),
            metadata={
                "full_name": (
                    "Raw Raman measurement files (54 spectra, "
                    "532/780 nm excitation) of 49 ocean polymer "
                    "specimens (Miller et al. 2022, Scientific "
                    "Data 9:780)"
                ),
                "source": "https://osf.io/7cqv4/",
                "paper": "https://doi.org/10.1038/s41597-022-01883-5",
                "bibtex": (
                    "@article{miller2022ocean, "
                    "title={A Raman spectral reference library of "
                    "potential anthropogenic and biological ocean "
                    "polymers}, "
                    "author={Miller, Emily A. and Yamahara, Kevan M. "
                    "and French, Chris and Spingarn, Neil E. and "
                    "Birch, James M. and Van Houtan, Kyle S.}, "
                    "journal={Scientific Data}, volume={9}, "
                    "pages={780}, year={2022}, "
                    "doi={10.1038/s41597-022-01883-5}}"
                ),
                "citation": [
                    "Miller, E. A., Yamahara, K. M., French, C., "
                    "Spingarn, N. E., Birch, J. M., and Van Houtan, "
                    "K. S. (2022). A Raman spectral reference "
                    "library of potential anthropogenic and "
                    "biological ocean polymers. Scientific Data, "
                    "9, 780. "
                    "https://doi.org/10.1038/s41597-022-01883-5 "
                    "(data: https://osf.io/7cqv4/)",
                ],
                "description": (
                    "The 54 distinct raw Raman measurements of the "
                    "49-specimen subset of the Miller et al. "
                    "2022 ocean-polymer library (Scientific Data "
                    "9, 780, doi:10.1038/s41597-022-01883-5; "
                    "deposit https://osf.io/7cqv4/, license note "
                    "as in ocean_polymers) - the unprocessed "
                    "counterpart of the ocean_polymers key. The "
                    "files of Data1_raw_csv.zip are Thermo/"
                    "Nicolet Omnic exports (line 1 'Created As "
                    "Omnic File,<CODE>', line 2 'Raman Shift / "
                    "cm-1,INT', data rows in DESCENDING "
                    "wavenumber order - flipped to ascending by "
                    "the loader). The zip holds 87 files, but 33 "
                    "are byte-identical copies of another file "
                    "under an instrument-run prefix (e.g. "
                    "'24544r56_1 PLAS 214 532 nm.csv' = 'PLAS 214 "
                    "532 nm.csv'); the loader keeps one of each. "
                    "Two excitation wavelengths are "
                    "present: 41 spectra at 532 nm (3,467 points, "
                    "76.55-3,418.58 cm^-1) and 13 at 780 "
                    "nm (2,700 points, 200.49-3,699.66 cm^-1); "
                    "every spectrum is linearly interpolated "
                    "onto the common 201-3,418 cm^-1 @1 cm^-1 "
                    "grid (3,218 points; the intersection of the "
                    "two native axes). 49 specimen codes: 36 "
                    "measured at 532 nm only, 8 at 780 nm only "
                    "and 5 at both -> is_grouped=True with "
                    "group_id = specimen code, so group-aware "
                    "splits keep a specimen's two spectra "
                    "together. "
                    "Task: classification of anthropogenic "
                    "polymers (PLAS-coded, 41 spectra) vs "
                    "biological materials (BIOL-coded, 13 "
                    "spectra) - the paper's central "
                    "identification question. Intensities are "
                    "raw counts with no normalization: absolute "
                    "scales differ per excitation (532 nm: "
                    "-511..29,382; 780 nm: -123..1,104), and the "
                    "8 specimens measured at 780 nm only are 5 "
                    "BIOL and 3 PLAS, so excitation and class "
                    "are partially confounded. The raw axis "
                    "extends beyond the common window at both "
                    "ends (truncation as in the ocean_polymers "
                    "key's 200-3,400 cm^-1 grid). Downloaded "
                    "directly from OSF's stable file URL (size + "
                    "sha256 pinned); no mirror required."
                ),
            },
            # Checked: after dropping the 33 byte-identical file
            # copies, 5 specimen codes have a 532 nm and a 780 nm
            # spectrum; group_id = specimen code keeps them together
            # in a group-aware split.
            is_grouped=True,
            # Checked: all 54 distinct files carry a complete, finite
            # spectrum and a parseable PLAS/BIOL specimen code.
            has_missing_labels=False,
            embedded_preprocessing=(
                "Raw Omnic counts, no normalization applied by "
                "the authors. The loader only (1) skips the two "
                "header rows, (2) flips the descending "
                "wavenumber axis to ascending, and (3) linearly "
                "interpolates onto the common 201-3,418 cm^-1 "
                "@1 cm^-1 grid (3,218 points; intersection of "
                "the 532 nm native axis 76.55-3,418.58 and the "
                "780 nm native axis 200.49-3,699.66 cm^-1). "
                "Absolute intensity scales are not comparable "
                "across excitation wavelengths (532 nm: "
                "-511..29,382; 780 nm: -123..1,104)."
            ),
        ),
    }
    logger = logging.getLogger(__name__)

    @staticmethod
    def _load_deepr_denoising(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, None]]:
        """
        Load the DeepeR denoising dataset from cache.

        Args:
            cache_path: Path to the cached dataset directory.

        Returns:
            Tuple of (spectra, raman_shifts, targets) or None if files are missing.
        """
        required_files = {
            "train_inputs": "Train_Inputs.mat",
            "train_outputs": "Train_Outputs.mat",
            "test_inputs": "Test_Inputs.mat",
            "test_outputs": "Test_Outputs.mat",
            "axis": "axis.txt",
        }

        # Check all files exist
        for key, fname in required_files.items():
            file_path = os.path.join(cache_path, fname)
            if not os.path.exists(file_path):
                MiscLoader.logger.warning(f"[!] Missing file: {fname}")
                MiscLoader.logger.warning(f"[!] Please download manually from OneDrive and extract to: {cache_path}")
                raise FileNotFoundError(f"Required file '{fname}' not found in cache path: {cache_path}")

        try:

            # Load data
            axis = np.loadtxt(os.path.join(cache_path, required_files["axis"]))
            train_inputs = loadmat(os.path.join(cache_path, required_files["train_inputs"]))["Train_Inputs"]
            train_outputs = loadmat(os.path.join(cache_path, required_files["train_outputs"]))["Train_Outputs"]
            test_inputs = loadmat(os.path.join(cache_path, required_files["test_inputs"]))["Test_Inputs"]
            test_outputs = loadmat(os.path.join(cache_path, required_files["test_outputs"]))["Test_Outputs"]

            # Combine train and test sets
            inputs = np.vstack((train_inputs, test_inputs))
            outputs = np.vstack((train_outputs, test_outputs))

            return inputs, axis, outputs, None

        except Exception as e:
            MiscLoader.logger.error(f"[!] Error loading dataset: {e}")
            raise e


    @staticmethod
    def _load_deepr_super_resolution(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load the DeepeR super-resolution dataset from cache.

        The super-resolution dataset contains hyperspectral Raman images rather
        than single spectra. It includes low-resolution input images and
        high-resolution target images for training super-resolution models.

        Files are expected to follow naming pattern: Cell-{HR/LR}_Norm_{wavelengths}-{id}.mat
        Example: Cell-HR_Norm_500-01-193-020.mat

        Args:
            cache_path: Path to the cached dataset directory.

        Returns:
            Tuple of (low_res_images, raman_shifts, high_res_images) or None if files are missing.
            Images are in format: (n_samples, height, width, n_wavelengths)
        """

        raise NotImplementedError


    @staticmethod
    def download_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Cannot download datasets from Miscellaneous loader")

    @staticmethod
    def load_dataset(
            dataset_name: str,
            cache_path: Optional[str] = None,
            load_data: bool = True,
    ) -> Optional[RamanDataset]:
        """
        Load a miscellaneous dataset from cache.

        If the dataset is not cached, attempts to download it first.

        Args:
            dataset_name: Name of the dataset to load.
            cache_path: Custom cache directory. If None, uses default location.

        Returns:
            RamanDataset object or None if loading fails.
        """
        if not LoaderTools.is_dataset_available(dataset_name, MiscLoader.DATASETS):
            raise FileNotFoundError(f"Dataset {dataset_name} is not available")

        # Get or set cache path
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Misc)

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        dataset_cache_path = os.path.join(cache_root, dataset_name)

        MiscLoader.logger.debug(f"Loading dataset from: {dataset_cache_path}")

        # Get dataset info and load data
        dataset_info = MiscLoader.DATASETS[dataset_name]

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
    def _load_csho33_bacteria(cache_path: str):
        """
        Download and load the csho33 bacteria dataset (Dropbox zip) expected layout:
          <extracted_root>/data/X_finetune.npy
                                 y_finetune.npy
                                 X_test.npy
                                 y_test.npy
                                 X_2018clinical.npy
                                 y_2018clinical.npy
                                 X_2019clinical.npy
                                 y_2019clinical.npy

        Returns spectra, raman_shifts, targets, class_names
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError("Cache root for Misc loader is not set. Cannot load bacteria_identification dataset.")

        shared_root = os.path.join(cache_root, "bacteria_identification")
        os.makedirs(shared_root, exist_ok=True)

        zip_name = "bacteria_identification.zip"
        zip_path = os.path.join(shared_root, zip_name)
        # Use Dropbox direct-download URL (dl=1)
        dl_url = "https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&dl=1"

        # Files we expect to find somewhere under the extracted tree
        required = [
            "X_finetune.npy", "y_finetune.npy",
            "X_test.npy", "y_test.npy",
            "X_2018clinical.npy", "y_2018clinical.npy",
            "X_2019clinical.npy", "y_2019clinical.npy",
        ]

        def find_dir_with_files(root_dir: str, filenames: list[str]) -> str | None:
            for root, dirs, files in os.walk(root_dir):
                if all(fname in files for fname in filenames):
                    return root
            return None

        # If the extracted folder already exists and contains our files, reuse it.
        extracted_dir = os.path.join(shared_root, "bacteria_identification")
        data_root = find_dir_with_files(extracted_dir, required) if os.path.isdir(extracted_dir) else None

        # If not found under the extracted dir, try scanning the shared_root and the wider cache_root
        if data_root is None:
            data_root = find_dir_with_files(shared_root, required)
        if data_root is None:
            data_root = find_dir_with_files(cache_root, required)

        # If no data files found yet, look for any existing zip file anywhere under the misc cache root and reuse it.
        existing_zip = None
        if data_root is None:
            for root, dirs, files in os.walk(cache_root):
                for f in files:
                    if f.lower().endswith('.zip'):
                        candidate = os.path.join(root, f)
                        if LoaderTools.is_valid_zip(candidate):
                            existing_zip = candidate
                            MiscLoader.logger.debug(f"Found existing zip at {existing_zip}; will attempt to extract")
                            break
                if existing_zip:
                    break

        # If we have an existing zip, extract it into the canonical extraction folder and search for the data
        if data_root is None and existing_zip is not None:
            try:
                LoaderTools.extract_zip_file_content(existing_zip, unzip_target_subdir="bacteria_identification")
            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to extract existing zip {existing_zip}: {e}")
            data_root = find_dir_with_files(extracted_dir, required) or find_dir_with_files(shared_root, required) or find_dir_with_files(cache_root, required)

        # If still not found, do a fresh download using the expected filename into shared_root
        if data_root is None:
            LoaderTools.download(url=dl_url, out_dir_path=shared_root, out_file_name=zip_name)
            extracted_dir = LoaderTools.extract_zip_file_content(zip_path, unzip_target_subdir="bacteria_identification")
            if extracted_dir is None:
                raise RuntimeError(f"[!] Failed to extract bacteria_identification zip after download")

            data_root = find_dir_with_files(extracted_dir, required) or find_dir_with_files(shared_root, required) or find_dir_with_files(cache_root, required)

        if data_root is None:
            raise FileNotFoundError(f"[!] Could not find required data files for bacteria_identification dataset after download and extraction. Please check the cache directory: {cache_root}")

        try:
            # Load arrays; some archives include extra files like reference or wavenumbers
            # Try to load an optional wavenumbers.npy if present
            X_f = np.load(os.path.join(data_root, "X_finetune.npy"))
            y_f = np.load(os.path.join(data_root, "y_finetune.npy"))
            X_t = np.load(os.path.join(data_root, "X_test.npy"))
            y_t = np.load(os.path.join(data_root, "y_test.npy"))
            X_2018 = np.load(os.path.join(data_root, "X_2018clinical.npy"))
            y_2018 = np.load(os.path.join(data_root, "y_2018clinical.npy"))
            X_2019 = np.load(os.path.join(data_root, "X_2019clinical.npy"))
            y_2019 = np.load(os.path.join(data_root, "y_2019clinical.npy"))

            # optional files
            X_r = None
            y_r = None
            raman_shifts = None
            if os.path.exists(os.path.join(data_root, "X_reference.npy")):
                X_r = np.load(os.path.join(data_root, "X_reference.npy"))
            if os.path.exists(os.path.join(data_root, "y_reference.npy")):
                y_r = np.load(os.path.join(data_root, "y_reference.npy"))
            if os.path.exists(os.path.join(data_root, "wavenumbers.npy")):
                raman_shifts = np.load(os.path.join(data_root, "wavenumbers.npy"))
        except Exception as e:
            raise RuntimeError(f"[!] Failed to load data arrays for bacteria_identification dataset: {e}")

        # Concatenate datasets in order finetune, (reference if present), test, 2018, 2019
        parts_X = [X_f]
        parts_y = [y_f]
        if X_r is not None and y_r is not None:
            parts_X.append(X_r)
            parts_y.append(y_r)
        parts_X.extend([X_t, X_2018, X_2019])
        parts_y.extend([y_t, y_2018, y_2019])

        X = np.vstack(parts_X)
        y = np.concatenate(parts_y)

        # If raman_shifts missing, create default
        if raman_shifts is None:
            if X.ndim == 2:
                raman_shifts = np.arange(X.shape[1], dtype=float)
            else:
                raman_shifts = [np.arange(x.shape[0], dtype=float) for x in X]

        # Determine class names from unique labels
        unique = sorted(list(map(str, np.unique(y))))

        return X.astype(float), raman_shifts, y.astype(int), unique

    @staticmethod
    def _load_yu2021_marine_pathogens(cache_path: str):
        """
        Download and load the "multi.csv" flat-CSV dataset rehosted at
        github.com/zshicode/Raman-Spectra-Deep-Learning, originally from
        Yu et al. (2021), Anal. Chem. 93(32), 11089-11098
        (doi:10.1021/acs.analchem.1c00431).

        Layout: 1138 rows x 1201 columns, no header row. The first 1200
        columns are Raman intensities (already 0-1 normalized per the
        rehosting repo's README), and the final column is an integer class
        label (0-7) for one of eight strains (SX-1..SX-8) isolated from the
        marine organism Urechis unicinctus.

        The repo's README states the spectral region is 600-1800 cm^-1
        across 1200 points -- no exact per-point wavenumber calibration is
        given by the source, so raman_shifts is reconstructed as an evenly
        spaced axis over that stated range (not an exact instrument
        calibration).

        Returns spectra, raman_shifts, targets, class_names.
        """
        return MiscLoader._load_zshicode_raman_csv(
            dataset_dir="marine_pathogens",
            csv_name="multi.csv",
            class_name_fn=lambda name: f"SX-{int(name) + 1}",
        )

    @staticmethod
    def _load_yu2021_marine_pathogens_binary(cache_path: str):
        """
        Download and load the "bin.csv" flat-CSV dataset rehosted at
        github.com/zshicode/Raman-Spectra-Deep-Learning. The rehosting
        repo's README attributes this file to the same paper as
        marine_pathogens/multi.csv (Yu et al. 2021, Anal. Chem. 93(32),
        11089-11098, doi:10.1021/acs.analchem.1c00431) -- an attribution
        taken as given, NOT independently confirmed (see the
        citation-confidence note on this dataset's DatasetInfo entry).

        Layout: 261 rows x 1201 columns, no header row. The first 1200
        columns are Raman intensities (already 0-1 normalized per the
        rehosting repo's README), and the final column is an integer class
        label: 0 = Acinetobacter baumannii (150 spectra), 1 = Pseudomonas
        nitritireducens (111 spectra).

        Same 600-1800 cm^-1 / 1200-point spectral region as multi.csv per
        the rehosting repo's README (no exact per-point calibration given
        by the source).

        Returns spectra, raman_shifts, targets, class_names.
        """
        class_names_map = {0: "Acinetobacter baumannii", 1: "Pseudomonas nitritireducens"}
        return MiscLoader._load_zshicode_raman_csv(
            dataset_dir="marine_pathogens_binary",
            csv_name="bin.csv",
            class_name_fn=lambda name: class_names_map.get(int(name), str(name)),
        )

    @staticmethod
    def _load_zshicode_raman_csv(dataset_dir: str, csv_name: str, class_name_fn):
        """
        Shared loader for the flat, headerless CSVs (spectral intensities +
        trailing integer label column) rehosted at
        github.com/zshicode/Raman-Spectra-Deep-Learning -- used by both
        marine_pathogens (multi.csv) and marine_pathogens_binary (bin.csv).
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError(f"Cache root for Misc loader is not set. Cannot load {dataset_dir} dataset.")

        shared_root = os.path.join(cache_root, dataset_dir)
        os.makedirs(shared_root, exist_ok=True)

        csv_path = os.path.join(shared_root, csv_name)
        dl_url = f"https://raw.githubusercontent.com/zshicode/Raman-Spectra-Deep-Learning/main/{csv_name}"

        if not LoaderTools._is_file_ready(csv_path):
            LoaderTools.download(url=dl_url, out_dir_path=shared_root, out_file_name=csv_name)

        try:
            df = pd.read_csv(csv_path, header=None)
        except Exception as e:
            raise RuntimeError(f"[!] Failed to read {dataset_dir} CSV at {csv_path}: {e}")

        if df.shape[1] < 2:
            raise RuntimeError(
                f"[!] {dataset_dir} CSV at {csv_path} has unexpected shape {df.shape}; expected >=2 columns "
                "(spectral intensities + trailing integer label)."
            )

        spectra = df.iloc[:, :-1].to_numpy(dtype=float)
        labels = df.iloc[:, -1].to_numpy()

        # Source paper/rehosting repo README: 1200-point spectra spanning
        # 600-1800 cm^-1 (no exact per-point calibration given).
        n_points = spectra.shape[1]
        raman_shifts = np.linspace(600.0, 1800.0, n_points, dtype=float)

        targets, class_names = encode_labels(pd.Series(labels))
        class_names = [class_name_fn(name) for name in class_names]

        return spectra, raman_shifts, targets.astype(int), class_names

    @staticmethod
    def _load_covid(cache_path):

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        dataset_cache = os.path.join(
            cache_root,
            "covid19",
        )

        if not os.path.exists(dataset_cache):
            raise Exception(f"Dataset covid19 not found in {cache_root}. Please contact the authors.")

        spectra_path = os.path.join(dataset_cache, "covid19.xls")

        xls = pd.ExcelFile(spectra_path)

        # data = pd.read_excel(xls, 'RAW')
        # data = pd.read_excel(xls, 'Poly7»RAW')
        data = pd.read_excel(xls, 'N1»Poly7»RAW')

        raman_shifts = data.iloc[:, 0].values.astype(float)
        spectra = data.iloc[:, 1:].values.astype(float)

        targets = data.keys()[1:].to_list()
        encoded_targets = np.array(["covid" in label for label in targets], dtype=int)
        target_names = ["control", "covid"]

        return spectra.T, raman_shifts, encoded_targets, target_names

    @staticmethod
    def _load_mlrod(cache_path: str):
        """
        Load the MLROD (ML Raman Open Dataset).

        This dataset consists of multiple csv files, one for each mineral class.
        Each csv file contains thousands of individual Raman spectra as columns.
        The loader will download, and process all of these into a single
        RamanDataset object.
        """
        urls = {
            "qtzalb_train_60s_3000.csv": "https://www.odr.io/view/downloadfile/63653?error_type=json",
            "qtzalb_train_30s_2000.csv": "https://www.odr.io/view/downloadfile/63652?error_type=json",
            "qtz_train_003s_4760.csv": "https://www.odr.io/view/downloadfile/63758?error_type=json",
            "qtz_train_003s_1148.csv": "https://www.odr.io/view/downloadfile/63757?error_type=json",
            "qtz_train_015s_625.csv": "https://www.odr.io/view/downloadfile/63756?error_type=json",
            "qtz_train_005s_1000.csv": "https://www.odr.io/view/downloadfile/63754?error_type=json",
            "qtz_train_005s_0900.csv": "https://www.odr.io/view/downloadfile/63753?error_type=json",
            "qtz_train_003s-1599.csv": "https://www.odr.io/view/downloadfile/63752?error_type=json",
            "ms_train_015s_5900.csv": "https://www.odr.io/view/downloadfile/63755?error_type=json",
            "mc_train_015s_3243.csv": "https://www.odr.io/view/downloadfile/63750?error_type=json",
            "mc_train_3s_2357.csv": "https://www.odr.io/view/downloadfile/63651?error_type=json",
            "hb_train_120s_1225_01.csv": "https://www.odr.io/view/downloadfile/63751?error_type=json",
            "hb_train_015s_2500.csv": "https://www.odr.io/view/downloadfile/63749?error_type=json",
            "hb_train_120s_1225_02.csv": "https://www.odr.io/view/downloadfile/63748?error_type=json",
            "hb_train_060s_104.csv": "https://www.odr.io/view/downloadfile/63650?error_type=json",
            "gyp_train_12s_3320.csv": "https://www.odr.io/view/downloadfile/63746?error_type=json",
            "gyp_train_15s_1680.csv": "https://www.odr.io/view/downloadfile/63649?error_type=json",
            "foaug_train_60s_3520.csv": "https://www.odr.io/view/downloadfile/63648?error_type=json",
            "foaug_train_30s_1640.csv": "https://www.odr.io/view/downloadfile/63647?error_type=json",
            "foalb_train_15s_5220.csv": "https://www.odr.io/view/downloadfile/63745?error_type=json",
            "fo_train_030s_5040.csv": "https://www.odr.io/view/downloadfile/63747?error_type=json",
            "fo_train_015s_1100.csv": "https://www.odr.io/view/downloadfile/63646?error_type=json",
            "en_train_015s_5600.csv": "https://www.odr.io/view/downloadfile/63744?error_type=json",
            "cal_train_1s_2732.csv": "https://www.odr.io/view/downloadfile/63743?error_type=json",
            "cal_train_1s_2300.csv": "https://www.odr.io/view/downloadfile/63645?error_type=json",
            "bt_train_120s_1225.csv": "https://www.odr.io/view/downloadfile/63741?error_type=json",
            "bt_train_015s_5600.csv": "https://www.odr.io/view/downloadfile/63644?error_type=json",
            "bt_train_015s_1100.csv": "https://www.odr.io/view/downloadfile/63643?error_type=json",
            "aug_train_20s_5103.csv": "https://www.odr.io/view/downloadfile/63738?error_type=json",
            "aug_train_15s_1000.csv": "https://www.odr.io/view/downloadfile/63642?error_type=json",
            "an_train_015s_5600.csv": "https://www.odr.io/view/downloadfile/63740?error_type=json",
            "albite_train_015s_5250.csv": "https://www.odr.io/view/downloadfile/63739?error_type=json",
            "albite_train_015s_625.csv": "https://www.odr.io/view/downloadfile/63737?error_type=json",
            "albite_train_015s_1100.csv": "https://www.odr.io/view/downloadfile/63641?error_type=json",
            "L_raw_gabbro50dust_test_015s_9740.csv": "https://www.odr.io/view/downloadfile/63659?error_type=json",
            "L_raw_granite50dust_test_015s_10000.csv": "https://www.odr.io/view/downloadfile/63245?error_type=json",
            "L_raw_granite0dust_test_015s_4084_final.csv": "https://www.odr.io/view/downloadfile/63658?error_type=json",
            "L_raw_granite0dust_test_030s_1373_final.csv": "https://www.odr.io/view/downloadfile/63657?error_type=json",
            "L_raw_granite0dust_test_015s_5571_final.csv": "https://www.odr.io/view/downloadfile/63656?error_type=json",
            "L_raw_gabbro0dust_test_015s_8906_final.csv": "https://www.odr.io/view/downloadfile/63655?error_type=json",
            "L_raw_gabbro0dust_test_015s_46_final.csv": "https://www.odr.io/view/downloadfile/63654?error_type=json",
        }

        label_map = {
            0: "Quartz",
            1: "Feldspar - Albite",
            2: "Amphibole - Hornblende",
            3: "Mica - Biotite",
            4: "Mica - Muscovite",
            5: "Olivine - Forsterite",
            6: "Pyroxene - Augite",
            7: "Pyroxene - Enstatite",
            8: "Feldspar - Anorthite",
            9: "Feldspar - Microcline",
            10: "Carbonate - Calcite",
            11: "Sulfate - Gypsum",
            12: "50% Quartz / 50% Albite Mix",
            13: "50% Forsterite / 50% Augite Mix",
            14: "50% Forsterite / 50% Albite Mix",
            15: "Granite slab - 50% dust",
        }


        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        shared_root = os.path.join(cache_root, "mlrod")
        os.makedirs(shared_root, exist_ok=True)

        # Primary source (odr.io) is frequently unreachable. Try it per file, then
        # fall back to a single Google Drive zip mirror containing all CSVs.
        gdrive_fallback_id = "1NDlDORvkoruwKMFP6y4sAf8M3KuZ6WnS"

        for filename, url in urls.items():
            file_path = os.path.join(shared_root, filename)
            if os.path.exists(file_path):
                continue
            MiscLoader.logger.debug(f"Downloading MLROD file: {filename}")
            try:
                downloaded_path = LoaderTools.download(url=url, out_dir_path=shared_root)
                if downloaded_path and downloaded_path != file_path:
                    os.rename(downloaded_path, file_path)
            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to download MLROD file {filename} from primary source: {e}")
                break

        missing = [f for f in urls if not os.path.exists(os.path.join(shared_root, f))]
        if missing:
            MiscLoader.logger.warning(
                f"[!] {len(missing)} MLROD file(s) missing from primary source; "
                f"falling back to Google Drive mirror"
            )
            import gdown
            zip_path = os.path.join(shared_root, "mlrod.zip")
            gdrive_url = f"https://drive.google.com/uc?id={gdrive_fallback_id}"
            gdown.download(gdrive_url, zip_path, quiet=False)
            if not os.path.exists(zip_path) or not LoaderTools.is_valid_zip(zip_path):
                raise RuntimeError(
                    "MLROD fallback download from Google Drive failed or produced an invalid zip."
                )
            LoaderTools.extract_zip_file_content(zip_path)

        all_spectra = []
        all_labels = []
        all_raman_shifts = []

        for filename in urls:
            file_path = os.path.join(shared_root, filename)
            if not os.path.exists(file_path):
                # Some archives may nest the CSVs in a subfolder; search recursively.
                matches = glob.glob(os.path.join(shared_root, "**", filename), recursive=True)
                if matches:
                    file_path = matches[0]
                else:
                    MiscLoader.logger.error(f"[!] MLROD file still missing after fallback: {filename}")
                    continue

            try:
                if "L_raw_granite0dust_test_015s_4084_final" in filename:
                    with open(file_path, 'r') as f:
                        file_content = f.read()

                    # Fix malformed float values in the file content
                    cleaned_content = file_content.replace('1630.89.1', '1630.90')
                    data = pd.read_csv(StringIO(cleaned_content), index_col=0)
                else:
                    data = pd.read_csv(file_path, index_col=0)

                raman_shifts = data.columns[:-1].astype(float).values
                spectra = data.iloc[:, :-1].values
                labels = data.iloc[:, -1].values

                # Map integer labels to string labels
                string_labels = [label_map.get(l, "Unknown") for l in labels]

                all_spectra.extend(spectra)
                all_raman_shifts.extend([raman_shifts] * spectra.shape[0])
                all_labels.extend(string_labels)

                # unique_labels = np.unique(labels)
                # for ul in unique_labels:
                #     print(f"{os.path.basename(file_path)}: {label_map.get(ul, 'Unknown')} (label {ul}) - {np.sum(labels == ul)} spectra")

            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to parse {file_path}: {e}")

        if not all_spectra:
            raise FileNotFoundError("No spectra were loaded. Please check download and extraction.")

        # Check if all spectra share the same wavenumber axis
        first_rs = all_raman_shifts[0]
        all_equal = (
            all(len(first_rs) == len(rs) for rs in all_raman_shifts)
            and all(np.allclose(first_rs, rs) for rs in all_raman_shifts)
        )

        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra_aligned = np.stack(all_spectra)
        else:
            # Interpolate all spectra to a common wavenumber grid
            raman_shifts, spectra_aligned = LoaderTools.align_raman_shifts(all_raman_shifts, all_spectra)

        # Encode labels
        encoded_targets, class_names = encode_labels(pd.Series(all_labels))

        return spectra_aligned, raman_shifts, encoded_targets, list(class_names)

    # Stable direct file URLs for the RaSPI deposit on Borealis
    # (University of Waterloo DataVerse), doi:10.5683/SP3/8UQQQN --
    # "api/access/datafile/<id>?format=original" addresses are immutable
    # within Borealis. MD5s are the values from the deposit's own file
    # manifest.
    _RASPI_BOREALIS_FILES = {
        "RaSPI_processed.csv": (
            "https://borealisdata.ca/api/access/datafile/1018848?format=original",
            "7796a8c14fd1a61144bd39a03c97b4f1",
        ),
        "RaSPI_metadata.csv": (
            "https://borealisdata.ca/api/access/datafile/1018849?format=original",
            "e35bbc21e33d8300b116add918caaf5b",
        ),
    }

    @staticmethod
    def _raspi_sample_stem(spectrum_id: str) -> str:
        """Map a RaSPI spectrum id to its physical sample id.

        The source names each plastic sample '<type><number>' and appends a
        trailing lowercase letter for each additional spectrum taken of the
        same sample (PE1a/PE1b are the two spectra of sample PE1; PET4c is
        the third measurement of sample PET4). Stripping the trailing letter
        yields the sample; ids without a letter map to themselves.
        """
        sid = str(spectrum_id).strip()
        stem = re.sub(r"[a-z]$", "", sid)
        return stem if stem else sid

    @staticmethod
    def _load_raman_spi(cache_path: str):
        """
        Download and load RaSPI (Hogan et al. 2026, Sci. Data 13:765,
        doi:10.1038/s41597-026-07103-8), data release doi:10.5683/SP3/8UQQQN.

        Files (Borealis / University of Waterloo DataVerse, CC BY 4.0):
          RaSPI_processed.csv  402 x 3726 wide CSV, row 0 = spectrum id,
                               columns = intensities on a uniform
                               225-3949 cm^-1 / 1 cm^-1 grid (baseline
                               subtracted, 0-1 normalized by the authors)
          RaSPI_metadata.csv   one row per spectrum: Spectra ID, Colour,
                               Origin, Additional Information, Plastic
                               Type, Laser (nm), Diffraction Grating

        The two files disagree slightly: the metadata lists 6 spectra
        (PET6a-d, PVC17, PVC18) that are absent from the spectra file, the
        spectra file holds 4 ids (PET5c, PET5d, PU15, PU16) that the
        metadata does not label, and PE1a/PE1b are stored twice in the
        spectra file. The loader deduplicates and inner-joins on the
        spectrum id, yielding the 396 fully-labeled spectra.

        Returns spectra, raman_shifts, targets, class_names, group_ids
        (group id = physical sample, see _raspi_sample_stem).
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError("Cache root for Misc loader is not set. Cannot load raman_spi dataset.")

        dataset_cache = os.path.join(cache_root, "raman_spi")
        os.makedirs(dataset_cache, exist_ok=True)

        for file_name, (url, md5) in MiscLoader._RASPI_BOREALIS_FILES.items():
            out_path = os.path.join(dataset_cache, file_name)
            if not LoaderTools._is_file_ready(out_path):
                LoaderTools.download(
                    url=url,
                    out_dir_path=dataset_cache,
                    out_file_name=file_name,
                    hash_target=md5,
                    hash_type=HASH_TYPE.md5,
                )
                if not LoaderTools._is_file_ready(out_path):
                    raise RuntimeError(
                        f"[!] Download of {file_name} for raman_spi failed "
                        "(network error or checksum mismatch). See logs."
                    )

        try:
            processed = pd.read_csv(
                os.path.join(dataset_cache, "RaSPI_processed.csv"), low_memory=False
            )
            metadata = pd.read_csv(os.path.join(dataset_cache, "RaSPI_metadata.csv"))
        except Exception as e:
            raise RuntimeError(f"[!] Failed to read raman_spi CSVs in {dataset_cache}: {e}")

        id_col = processed.columns[0]
        wav_cols = list(processed.columns[1:])
        if len(wav_cols) < 2:
            raise RuntimeError(
                f"[!] raman_spi RaSPI_processed.csv has unexpected shape {processed.shape}; "
                "expected an id column plus one column per wavenumber."
            )

        # PE1a/PE1b are stored twice in the source file; keep the first copy.
        processed = processed.drop_duplicates(subset=[id_col], keep="first")
        merged = processed.merge(metadata, left_on=id_col, right_on="Spectra ID", how="inner")
        if merged.empty:
            raise RuntimeError("[!] raman_spi: no overlap between RaSPI_processed.csv and RaSPI_metadata.csv ids.")

        # The wavenumber axis lives in the column headers (one column per
        # point on a uniform 225-3949 cm^-1 grid).
        raman_shifts = np.array([float(col) for col in wav_cols])
        # Guard: the processed file is supposed to be on an ascending grid.
        if not np.all(np.diff(raman_shifts) > 0):
            raise RuntimeError("[!] raman_spi: RaSPI_processed.csv wavenumber axis is not ascending.")

        spectra = merged[wav_cols].to_numpy(dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError("[!] raman_spi: missing intensity values.")
        plastic_types = merged["Plastic Type"]
        group_ids = merged["Spectra ID"].map(MiscLoader._raspi_sample_stem).to_numpy()

        # Integrity pin: the deposit's fully-labeled usable set is 396
        # spectra from 272 physical samples across 14 plastic types.
        n_samples = len(np.unique(group_ids))
        n_classes = len(np.unique(plastic_types))
        if spectra.shape != (396, len(raman_shifts)) or n_classes != 14 or n_samples != 272:
            raise RuntimeError(
                f"[!] raman_spi: expected 396 x {len(raman_shifts)} spectra with "
                f"14 classes / 272 physical samples, got {spectra.shape} with "
                f"{n_classes} classes / {n_samples} samples."
            )

        targets, class_names = encode_labels(plastic_types)
        return spectra, raman_shifts, targets, list(class_names), group_ids

    # RaMPI map files, same Borealis deposit as raman_spi
    # (doi:10.5683/SP3/8UQQQN). One file per Raman map; MD5s from the
    # deposit's file manifest.
    _RAMPI_BOREALIS_FILES = {
        "Assig_PC_1.csv": (
            "https://borealisdata.ca/api/access/datafile/969473?format=original",
            "e88aa4240a6349f321b8f524f41c5208",
        ),
        "Assig_PC_2.csv": (
            "https://borealisdata.ca/api/access/datafile/969527?format=original",
            "7d05ecdc542b30f02589b563939f349a",
        ),
        "Assig_PE.csv": (
            "https://borealisdata.ca/api/access/datafile/969519?format=original",
            "bc2edd0bde62e9aedcdcefe426f15d7b",
        ),
        "Assig_PET.csv": (
            "https://borealisdata.ca/api/access/datafile/969485?format=original",
            "3d2bdd0ba3b295c6bb76866416af60d0",
        ),
        "Assig_PMMA_PC.csv": (
            "https://borealisdata.ca/api/access/datafile/969489?format=original",
            "64d9ac1e4655b1e91cb75988cb1952b1",
        ),
        "Assig_PMMA_PET.csv": (
            "https://borealisdata.ca/api/access/datafile/969472?format=original",
            "583a632eb62308fa9667c9d83cfd9bb3",
        ),
        "Assig_PMMA_PET_PE_ABS.csv": (
            "https://borealisdata.ca/api/access/datafile/969535?format=original",
            "3b81f317fa81876bc751a3f23c892044",
        ),
        "Assig_PMMA_PTFE_PE.csv": (
            "https://borealisdata.ca/api/access/datafile/969495?format=original",
            "d25be8efb26dfb4a75ab28a99ab84da3",
        ),
        "Assig_POM.csv": (
            "https://borealisdata.ca/api/access/datafile/969520?format=original",
            "760a3270fcffad16e2212101fab10090",
        ),
        "Assig_POM_ABS.csv": (
            "https://borealisdata.ca/api/access/datafile/969516?format=original",
            "3906438af7333fef378dd225626a9003",
        ),
        "Assig_PP.csv": (
            "https://borealisdata.ca/api/access/datafile/969496?format=original",
            "71f0b77723cdd83dfe5b771afa1e0ef9",
        ),
        "Assig_PS.csv": (
            "https://borealisdata.ca/api/access/datafile/969492?format=original",
            "15333ea9102c1c34fa3fd50d371ccb18",
        ),
        "Assig_PS_ABS.csv": (
            "https://borealisdata.ca/api/access/datafile/969502?format=original",
            "ae5869c967b6f5737a0d6fad462a97e8",
        ),
        "Assig_PS_PE_PEs_2.csv": (
            "https://borealisdata.ca/api/access/datafile/969522?format=original",
            "752cac0ad8efc7e2d31e2eec8e8660e6",
        ),
        "Assig_PS_PTFE_POM_PMMA.csv": (
            "https://borealisdata.ca/api/access/datafile/969480?format=original",
            "e56ea675c255d7ccd60a91fa3062cf24",
        ),
        "Assig_PTFE.csv": (
            "https://borealisdata.ca/api/access/datafile/969504?format=original",
            "407d58d2f2482ae8377a33c6fc63b6d9",
        ),
        "Assig_PU.csv": (
            "https://borealisdata.ca/api/access/datafile/969534?format=original",
            "e1ee57bd86a15656b3bb5762bb429249",
        ),
        "Assig_PVC-PU-POM.csv": (
            "https://borealisdata.ca/api/access/datafile/969488?format=original",
            "1e50552adb6fc1451106881e9ba0b713",
        ),
        "Assig_PVC_1.csv": (
            "https://borealisdata.ca/api/access/datafile/969479?format=original",
            "7ff6d4c7f862f694351ef43efa6ea295",
        ),
        "Assig_PVC_2.csv": (
            "https://borealisdata.ca/api/access/datafile/969538?format=original",
            "3b46c819d7b955b911e240434d17942f",
        ),
        "Assig_PVC_ABS_PEs.csv": (
            "https://borealisdata.ca/api/access/datafile/969494?format=original",
            "e40442070d4dbee658add07db3a5d75c",
        ),
        "Assig_PVC_PET.csv": (
            "https://borealisdata.ca/api/access/datafile/969493?format=original",
            "abe6e310c5d28c8de95d9051e96b9e6b",
        ),
        "Assig_Map_PS_PE_PEs_1.csv": (
            "https://borealisdata.ca/api/access/datafile/969518?format=original",
            "f372bd6ec804eec9696ef120626ddfb7",
        ),
        "Assig_Silicone.csv": (
            "https://borealisdata.ca/api/access/datafile/969498?format=original",
            "703413ed9753b7db45d836271cd7084f",
        ),
        "Assig_Silicone_ABS.csv": (
            "https://borealisdata.ca/api/access/datafile/969477?format=original",
            "1c1bc726be57b465eebcf92c482b1f51",
        ),
        "Assig_Env_POM_PP_PE.csv": (
            "https://borealisdata.ca/api/access/datafile/969500?format=original",
            "457ad86d3ea380bf2b20786957364f73",
        ),
        "Assig_Env_PP_PE_1.csv": (
            "https://borealisdata.ca/api/access/datafile/969503?format=original",
            "0100b4f0905837e14e66a1e8c8634aed",
        ),
        "Assig_Env_PP_PE_2.csv": (
            "https://borealisdata.ca/api/access/datafile/969481?format=original",
            "c60d955986804cd8f3a92c5e29f77c1a",
        ),
        "Assig_Env_PP_PTFE.csv": (
            "https://borealisdata.ca/api/access/datafile/969510?format=original",
            "4c0b428ee99d0ad261c4c1195de3f929",
        ),
        "Assig_Nylon.csv": (
            "https://borealisdata.ca/api/access/datafile/969523?format=original",
            "afb6dd78507aff8103a02b227762887c",
        ),
        "Assig_Nylon_ABS_PMMA.csv": (
            "https://borealisdata.ca/api/access/datafile/969483?format=original",
            "09f29eaa80f8b22cc274bd529d548239",
        ),
        "Assig_Nylon_PEs_PS.csv": (
            "https://borealisdata.ca/api/access/datafile/969513?format=original",
            "ebccc8479edf0a1f3ef8c40a3f2e11ff",
        ),
        "Assig_Nylon_POM.csv": (
            "https://borealisdata.ca/api/access/datafile/969525?format=original",
            "b9a4782a500957393d976820fa6f4381",
        ),
        "Assig_Nylon_PTFE_POM.csv": (
            "https://borealisdata.ca/api/access/datafile/969491?format=original",
            "8655f68a7c3e92824c4108f309a55156",
        ),
    }

    # The 34 per-map axes are non-uniform (1,017 points each, descending,
    # ~709.9-1827.2 cm^-1). All of them cover 710-1827 cm^-1, so this
    # integer 1 cm^-1 grid (1,118 points) is a pure interpolation for every
    # map -- consistent with the 1 cm^-1 grid of the companion raman_spi.
    _RAMPI_COMMON_GRID = np.arange(710.0, 1828.0, 1.0)

    # The manual per-pixel assignments in the deposit; 'blank' marks
    # unassigned background pixels and is excluded from this dataset.
    _RAMPI_PLASTIC_LABELS = frozenset({
        "acrylonitrile butadiene styrene",
        "nylon",
        "poly (methyl methacrylate)",
        "polycarbonate",
        "polyester",
        "polyethylene",
        "polyethylene terephthalate",
        "polyoxymethylene",
        "polypropylene",
        "polystyrene",
        "polytetrafluoroethylene",
        "polyurethane",
        "polyvinyl chloride",
        "silicone",
    })

    @staticmethod
    def _rampi_map_name(file_name: str) -> str:
        """'Assig_PVC-PU-POM.csv' -> 'PVC-PU-POM' (one map per file)."""
        return file_name[len("Assig_"):-len(".csv")]

    @staticmethod
    def _load_rampi(cache_path: str):
        """
        Download and load RaMPI (Hogan et al. 2026, Sci. Data 13:765,
        doi:10.1038/s41597-026-07103-8), data release doi:10.5683/SP3/8UQQQN.

        Files: the deposit's 34 'Assig_*.csv' files, one per Raman map
        (30 pristine two-material blend maps + 4 environmental Great Lakes
        Seabin maps), 633 nm / 1800 l/mm grating. Each file's first row is
        the map's wavenumber axis ('assignment', w1, w2, ... -- descending,
        1,017 points); every following row is one pixel spectrum
        ('<label>', i1, i2, ...) with a manual plastic-type assignment or
        'blank'.

        The 33,119 pixel spectra are resampled from their per-map native
        axes onto the common 710-1,827 cm^-1 / 1 cm^-1 grid and the 16,598
        plastic-labeled pixels are kept ('blank' pixels have no class).

        Returns spectra, raman_shifts, targets, class_names, group_ids
        (group id = map name).
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError("Cache root for Misc loader is not set. Cannot load raman_mpi dataset.")

        dataset_cache = os.path.join(cache_root, "raman_mpi")
        os.makedirs(dataset_cache, exist_ok=True)

        for file_name, (url, md5) in MiscLoader._RAMPI_BOREALIS_FILES.items():
            out_path = os.path.join(dataset_cache, file_name)
            if not LoaderTools._is_file_ready(out_path):
                LoaderTools.download(
                    url=url,
                    out_dir_path=dataset_cache,
                    out_file_name=file_name,
                    hash_target=md5,
                    hash_type=HASH_TYPE.md5,
                )
                if not LoaderTools._is_file_ready(out_path):
                    raise RuntimeError(
                        f"[!] Download of {file_name} for raman_mpi failed "
                        "(network error or checksum mismatch). See logs."
                    )

        spectra_list = []
        label_list = []
        group_list = []
        n_total = 0

        for file_name in sorted(MiscLoader._RAMPI_BOREALIS_FILES):
            try:
                df = pd.read_csv(os.path.join(dataset_cache, file_name), low_memory=False)
            except Exception as e:
                raise RuntimeError(f"[!] Failed to read raman_mpi map file {file_name}: {e}")

            wav_cols = list(df.columns[1:])
            if len(wav_cols) < 2:
                raise RuntimeError(f"[!] raman_mpi {file_name} has no wavenumber columns.")

            axis = np.array([float(col) for col in wav_cols])
            # The source axes are descending; flip to ascending for interp.
            if axis[0] > axis[-1]:
                axis = axis[::-1]
                intensities = df[wav_cols].to_numpy(dtype=float)[:, ::-1]
            else:
                intensities = df[wav_cols].to_numpy(dtype=float)
            if not np.all(np.diff(axis) > 0):
                raise RuntimeError(f"[!] raman_mpi {file_name}: wavenumber axis is not monotonic.")
            if np.isnan(intensities).any():
                raise RuntimeError(f"[!] raman_mpi {file_name}: missing intensity values.")

            # Pure interpolation: the common grid lies inside every map axis.
            # (np.interp only takes 1-D fp, so resample row by row.)
            grid = MiscLoader._RAMPI_COMMON_GRID
            resampled = np.empty((intensities.shape[0], grid.size), dtype=float)
            for i in range(intensities.shape[0]):
                resampled[i] = np.interp(grid, axis, intensities[i])

            labels = df[df.columns[0]].astype(str).str.strip().to_numpy()
            keep = np.isin(labels, list(MiscLoader._RAMPI_PLASTIC_LABELS))
            n_blank = int((labels == "blank").sum())
            unknown = np.unique(labels[~keep & (labels != "blank")])
            if unknown.size:
                raise RuntimeError(f"[!] raman_mpi {file_name}: unexpected labels {unknown.tolist()}.")
            n_total += len(labels)

            spectra_list.append(resampled[keep])
            label_list.append(labels[keep])
            group_list.extend([MiscLoader._rampi_map_name(file_name)] * int(keep.sum()))

        if n_total != 33119:
            raise RuntimeError(f"[!] raman_mpi: expected 33119 pixel spectra across 34 maps, got {n_total}.")
        if not label_list:
            raise RuntimeError("[!] raman_mpi: no plastic-labeled pixels found.")

        spectra = np.concatenate(spectra_list, axis=0)
        # The 33,119 total above pins the deposit; this pins what is
        # actually returned (blank pixels excluded).
        if spectra.shape[0] != 16598:
            raise RuntimeError(f"[!] raman_mpi: expected 16598 plastic-labeled pixels, got {spectra.shape[0]}.")
        targets, class_names = encode_labels(pd.Series(np.concatenate(label_list)))
        group_ids = np.array(group_list)
        return spectra, MiscLoader._RAMPI_COMMON_GRID.copy(), targets, list(class_names), group_ids

    # Lei et al. 2022 (doi:10.1021/acs.analchem.2c02451) training files,
    # Borealis doi:10.5683/SP3/LSN0R0. One CSV per resolution grid; MD5s
    # from the deposit's file manifest.
    _LEI_MPLASTICS_FILES = {
        "1cm": (
            "2022-03-31-20aug-1cm-720-1800-20baseline.csv",
            "https://borealisdata.ca/api/access/datafile/406825?format=original",
            "20fa0433fe10c77884a36b76587f0d45",
        ),
        "2cm": (
            "2022-03-31-20aug-2cm-720-1800-20baseline.csv",
            "https://borealisdata.ca/api/access/datafile/406811?format=original",
            "99b53eb93f81f7d0803dbe9355d9e954",
        ),
        "4cm": (
            "2022-03-31-20aug-4cm-720-1800-20baseline.csv",
            "https://borealisdata.ca/api/access/datafile/406820?format=original",
            "db25a54e27b70b0db9de119121547347",
        ),
        "8cm": (
            "2022-03-31-20aug-8cm-720-1800-20baseline.csv",
            "https://borealisdata.ca/api/access/datafile/406819?format=original",
            "f9a0c1b44de2803300f601c35f0f4364",
        ),
    }

    @staticmethod
    def _lei_read_csv(dataset_cache: str, resolution: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Download (if not already cached) and read one Lei et al. 2022
        microplastic training CSV (see _load_lei_microplastics).

        Returns (df, raman_shifts): one row per spectrum, first three
        columns 'plastic', 'supplier', 'sampleid', then one column per
        wavenumber.
        """
        if resolution not in MiscLoader._LEI_MPLASTICS_FILES:
            raise ValueError(f"Unknown microplastics training resolution: {resolution}")
        file_name, url, md5 = MiscLoader._LEI_MPLASTICS_FILES[resolution]

        out_path = os.path.join(dataset_cache, file_name)
        if not LoaderTools._is_file_ready(out_path):
            LoaderTools.download(
                url=url,
                out_dir_path=dataset_cache,
                out_file_name=file_name,
                hash_target=md5,
                hash_type=HASH_TYPE.md5,
            )
            if not LoaderTools._is_file_ready(out_path):
                raise RuntimeError(
                    f"[!] Download of {file_name} for microplastics_training "
                    "failed (network error or checksum mismatch). See logs."
                )

        try:
            df = pd.read_csv(out_path, low_memory=False)
        except Exception as e:
            raise RuntimeError(f"[!] Failed to read microplastics training CSV {file_name}: {e}")

        for col in ("plastic", "supplier", "sampleid"):
            if col not in df.columns:
                raise RuntimeError(f"[!] Lei microplastics {file_name}: missing column '{col}'.")

        wav_cols = list(df.columns[3:])
        if len(wav_cols) < 2:
            raise RuntimeError(f"[!] Lei microplastics {file_name}: no wavenumber columns.")
        raman_shifts = np.array([float(col) for col in wav_cols])
        if not np.all(np.diff(raman_shifts) > 0):
            raise RuntimeError(f"[!] Lei microplastics {file_name}: wavenumber axis is not ascending.")
        return df, raman_shifts

    @staticmethod
    def _load_lei_microplastics(cache_path: str, resolution: str):
        """
        Download and load the Lei et al. 2022 (Anal. Chem. 94(49),
        17011-17019, doi:10.1021/acs.analchem.2c02451) microplastic
        training CSV for one resolution grid, data release
        doi:10.5683/SP3/LSN0R0 (Borealis, CC BY 4.0).

        File layout: one row per spectrum; columns 'plastic', 'supplier',
        'sampleid', then one column per wavenumber (720-1800 cm^-1 at the
        requested grid). The deposit's 4,520 rows comprise 186 unique
        plastic measurements (x20 augmented) and 40 synthetic baseline
        spectra (x20 augmented); the 800 'blank' rows are artificial
        (flat-line / broad inverted parabola) and carry no measurement,
        so they are excluded.

        Returns spectra, raman_shifts, targets, class_names, group_ids
        (group id = supplier sample id).
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError("Cache root for Misc loader is not set. Cannot load microplastics training dataset.")

        dataset_cache = os.path.join(cache_root, "microplastics_training")
        os.makedirs(dataset_cache, exist_ok=True)

        df, raman_shifts = MiscLoader._lei_read_csv(dataset_cache, resolution)

        spectra = df[list(df.columns[3:])].to_numpy(dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError(f"[!] microplastics_training_{resolution}: missing intensity values.")

        plastic = df["plastic"].astype(str).str.strip()
        # The 800 'blank' rows are the authors' synthetic baseline spectra
        # (20 flat-line + 20 broad inverted parabola, each x20 augmented),
        # not measurements -- exclude them.
        keep = plastic != "blank"
        if len(df) != 4520 or int(keep.sum()) != 3720:
            raise RuntimeError(
                f"[!] microplastics_training_{resolution}: expected 4520 rows with "
                f"3720 non-blank, got {len(df)} with {int(keep.sum())}."
            )

        targets, class_names = encode_labels(plastic[keep])
        group_ids = df["sampleid"].astype(str).str.strip()[keep].to_numpy()
        # Integrity pin (the clean-key loader pins its derived counts
        # too): 108 supplier samples across 14 plastic classes.
        if len(np.unique(group_ids)) != 108 or len(class_names) != 14:
            raise RuntimeError(
                f"[!] microplastics_training_{resolution}: expected 108 supplier sample "
                f"ids / 14 plastic classes, got {len(np.unique(group_ids))} / {len(class_names)}."
            )
        return spectra[keep.to_numpy()], raman_shifts, targets, list(class_names), group_ids

    @staticmethod
    def _lei_measurement_blocks(sid, labels):
        """
        Group the non-blank rows of a Lei et al. 2022 training CSV into
        20-row measurement blocks.

        The deposited rows are laid out in 20-row blocks, one block per
        source measurement (the 20 augmented variants of that spectrum);
        a sample measured twice contributes two consecutive 20-row blocks
        under the same 'sampleid' (a 40-row run).

        Args:
            sid: array of 'sampleid' strings, one per kept row.
            labels: array of plastic labels, one per kept row.

        Returns a list of (start_row, sampleid, plastic_label), one per
        20-row block, in row order.
        """
        blocks = []
        i, n = 0, len(sid)
        while i < n:
            j = i
            while j < n and sid[j] == sid[i]:
                j += 1
            if (j - i) % 20 != 0:
                raise RuntimeError(
                    f"[!] Lei microplastics: sampleid run of {j - i} rows "
                    f"at row {i} is not a multiple of the 20-row block."
                )
            if len(np.unique(labels[i:j])) != 1:
                raise RuntimeError(
                    f"[!] Lei microplastics: sampleid run at row {i} carries "
                    "mixed plastic labels."
                )
            for k in range(i, j, 20):
                blocks.append((k, sid[k], labels[k]))
            i = j
        return blocks

    @staticmethod
    def _load_lei_microplastics_clean(cache_path: str):
        """
        Load microplastics_training_clean: the un-augmented core of the
        Lei et al. 2022 training set, one representative spectrum per
        source measurement on the paper's native 1 cm^-1 grid.

        Each representative is the row-wise mean of the 20 augmented
        variants the authors deposited per measurement: averaging
        cancels the augmentation's random noise (strongest-peak SNR
        3-15) and leaves the measured spectral shape (absolute amplitude
        is arbitrary, as for all variants of this deposit).

        Returns spectra (186 x 1081), raman_shifts (1081,), targets
        (186,), class_names, group_ids (supplier sample id; 108).
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError("Cache root for Misc loader is not set. Cannot load microplastics training dataset.")

        dataset_cache = os.path.join(cache_root, "microplastics_training")
        os.makedirs(dataset_cache, exist_ok=True)

        df, raman_shifts = MiscLoader._lei_read_csv(dataset_cache, "1cm")

        plastic = df["plastic"].astype(str).str.strip()
        keep = plastic != "blank"
        if len(df) != 4520 or int(keep.sum()) != 3720:
            raise RuntimeError(
                f"[!] microplastics_training_clean: expected 4520 rows with "
                f"3720 non-blank, got {len(df)} with {int(keep.sum())}."
            )

        X = df[list(df.columns[3:])].to_numpy(dtype=float)[keep.to_numpy()]
        sid = df["sampleid"].astype(str).str.strip().to_numpy()[keep.to_numpy()]
        labels = plastic.to_numpy()[keep.to_numpy()]

        # 20-row blocks, one per source measurement (see
        # _lei_measurement_blocks); the representative is the row-wise
        # mean of the block's 20 augmented variants.
        blocks = MiscLoader._lei_measurement_blocks(sid, labels)
        reps = [X[k:k + 20].mean(axis=0) for k, _, _ in blocks]
        groups = [s for _, s, _ in blocks]
        block_labels = [l for _, _, l in blocks]

        if len(reps) != 186:
            raise RuntimeError(
                f"[!] microplastics_training_clean: expected 186 measurement "
                f"blocks, got {len(reps)}."
            )
        if len(set(groups)) != 108:
            raise RuntimeError(
                f"[!] microplastics_training_clean: expected 108 supplier "
                f"sample ids, got {len(set(groups))}."
            )

        spectra = np.vstack(reps)
        if np.isnan(spectra).any():
            raise RuntimeError("[!] microplastics_training_clean: missing intensity values.")

        targets, class_names = encode_labels(np.array(block_labels))
        return spectra, raman_shifts, targets, list(class_names), np.array(groups)

    # yeast_bioreactor_fermentation (4TU.ResearchData, Klaverdijk et al. 2025)
    _YEAST_BIOREACTOR_ZIP = "Chapter 2.zip"
    _YEAST_BIOREACTOR_MD5 = "cd5b3b78afe641320974193129d0bc27"
    _YEAST_BIOREACTOR_URL = (
        "https://data.4tu.nl/file/4fbcf35e-1954-48a4-a2a3-693e19c753e7/"
        "892f9b68-04d7-441c-8da8-07a01002b215"
    )
    # (reference prefix, folder, expected batch -> expected sample count)
    _YEAST_BIOREACTOR_PARTS = {
        "Base": ("Base dataset", {"B1": 12, "B2": 13, "B3": 13}),
        "Validation": ("Validation dataset", {"B4": 13}),
        "Test": ("Test dataset", {"FB": 14}),
    }
    _YEAST_BIOREACTOR_TARGETS = ["Glucose [mM]", "Ethanol [mM]", "Biomass [g/L]"]

    @staticmethod
    def _load_yeast_bioreactor_fermentation(cache_path: str):
        """
        Download and load the yeast bioreactor fermentation dataset
        (4TU.ResearchData doi:10.4121/4fbcf35e-1954-48a4-a2a3-693e19c753e7,
        CC BY 4.0, data for the Klaverdijk et al. 2025 ABAC paper,
        doi:10.1007/s00216-025-05768-5).

        Uses the Chapter 2 base/validation/test datasets: per part, one
        '..._references.xlsx' (Sample, Glucose [mM], Ethanol [mM],
        Biomass [g/L]; 5 failed HPLC glucose values are empty) and one
        '..._spectra_1m.xlsx' (single 60 s acquisition per sample; the
        2m-10m files are stackings of the same 60 s blocks). Sample
        names are like 'B1 S1 ' (trailing space); the batch prefix is the
        group id. All files share the 100-3,425 cm^-1 / 1 cm^-1 header.

        Returns spectra, raman_shifts, targets (65 x 3, NaN where a
        reference value is missing), target_names, group_ids.
        """
        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise ValueError("Cache root for Misc loader is not set. Cannot load yeast_bioreactor_fermentation dataset.")

        dataset_cache = os.path.join(cache_root, "yeast_bioreactor_fermentation")
        os.makedirs(dataset_cache, exist_ok=True)

        zip_path = os.path.join(dataset_cache, MiscLoader._YEAST_BIOREACTOR_ZIP)
        if not LoaderTools._is_file_ready(zip_path):
            LoaderTools.download(
                url=MiscLoader._YEAST_BIOREACTOR_URL,
                out_dir_path=dataset_cache,
                out_file_name=MiscLoader._YEAST_BIOREACTOR_ZIP,
                hash_target=MiscLoader._YEAST_BIOREACTOR_MD5,
                hash_type=HASH_TYPE.md5,
            )
            if not LoaderTools._is_file_ready(zip_path):
                raise RuntimeError(
                    "[!] Download of Chapter 2.zip for yeast_bioreactor_fermentation "
                    "failed (network error or checksum mismatch). See logs."
                )
        if not os.path.isdir(os.path.join(dataset_cache, "Chapter 2", "Chapter 2")):
            if LoaderTools.extract_zip_file_content(zip_path, "Chapter 2") is None:
                raise RuntimeError(
                    "[!] yeast_bioreactor_fermentation: could not extract Chapter 2.zip."
                )
        ch2_root = os.path.join(dataset_cache, "Chapter 2", "Chapter 2")

        spectra_list = []
        raman_shifts_ref = None
        targets_list = []
        group_ids = []
        for prefix, (folder, expected_batches) in MiscLoader._YEAST_BIOREACTOR_PARTS.items():
            ref = pd.read_excel(os.path.join(ch2_root, folder, f"{prefix}_dataset_references.xlsx"))
            sp = pd.read_excel(os.path.join(ch2_root, folder, f"{prefix}_dataset_spectra_1m.xlsx"))
            if len(ref.columns) < 4:
                raise RuntimeError(
                    f"[!] yeast_bioreactor_fermentation: unexpected reference columns for {prefix}."
                )
            ref_cols = ["Sample"] + MiscLoader._YEAST_BIOREACTOR_TARGETS
            for col in ref_cols:
                if col not in ref.columns:
                    raise RuntimeError(
                        f"[!] yeast_bioreactor_fermentation: missing column '{col}' in {prefix} references."
                    )
            ref = ref[ref_cols].copy()
            ref["Sample"] = ref["Sample"].astype(str).str.strip()
            if not ref["Sample"].is_unique:
                raise RuntimeError(f"[!] yeast_bioreactor_fermentation: duplicate samples in {prefix} references.")

            label_col = sp.columns[0]
            sp = sp.copy()
            sp["Sample"] = sp[label_col].astype(str).str.strip()
            if not sp["Sample"].is_unique or set(sp["Sample"]) != set(ref["Sample"]):
                raise RuntimeError(
                    f"[!] yeast_bioreactor_fermentation: spectra/reference sample mismatch for {prefix}."
                )
            ref = ref.set_index("Sample").loc[sp["Sample"].to_list()].reset_index()

            wn_cols = [c for c in sp.columns if c != label_col]
            try:
                wn = np.array([float(c) for c in wn_cols])
            except (TypeError, ValueError) as e:
                raise RuntimeError(f"[!] yeast_bioreactor_fermentation: non-numeric wavenumber header in {prefix} spectra: {e}")
            if raman_shifts_ref is None:
                raman_shifts_ref = wn
            elif wn.shape != raman_shifts_ref.shape or not np.allclose(wn, raman_shifts_ref):
                raise RuntimeError(f"[!] yeast_bioreactor_fermentation: wavenumber header differs in {prefix} spectra.")

            spectra_list.append(sp[wn_cols].to_numpy(dtype=float))
            targets_list.append(ref[MiscLoader._YEAST_BIOREACTOR_TARGETS].to_numpy(dtype=float))
            for sample in sp["Sample"]:
                batch = sample.split()[0]
                if batch not in expected_batches:
                    raise RuntimeError(
                        f"[!] yeast_bioreactor_fermentation: unexpected batch '{batch}' in {prefix}."
                    )
                group_ids.append(batch)

        if raman_shifts_ref is None:
            raise RuntimeError("[!] yeast_bioreactor_fermentation: no spectra parsed.")
        if len(raman_shifts_ref) != 3326 or raman_shifts_ref[0] != 100 or raman_shifts_ref[-1] != 3425:
            raise RuntimeError(
                f"[!] yeast_bioreactor_fermentation: expected the 100-3,425 cm^-1 grid, got "
                f"{len(raman_shifts_ref)} points ({raman_shifts_ref[0]}-{raman_shifts_ref[-1]})."
            )

        spectra = np.vstack(spectra_list)
        targets = np.vstack(targets_list)
        group_ids = np.array(group_ids)

        if len(spectra) != 65:
            raise RuntimeError(f"[!] yeast_bioreactor_fermentation: expected 65 samples, got {len(spectra)}.")
        if np.isnan(spectra).any() or (spectra < 0).any():
            raise RuntimeError("[!] yeast_bioreactor_fermentation: missing or negative intensity values.")
        batch_counts = Counter(group_ids.tolist())
        expected_counts = {}
        for _, counts in MiscLoader._YEAST_BIOREACTOR_PARTS.values():
            expected_counts.update(counts)
        if dict(batch_counts) != expected_counts:
            raise RuntimeError(
                f"[!] yeast_bioreactor_fermentation: unexpected per-batch counts {dict(batch_counts)} (expected {expected_counts})."
            )
        n_nan = int(np.isnan(targets).sum())
        if n_nan != 5 or np.isnan(targets[:, 1:]).any():
            raise RuntimeError(
                f"[!] yeast_bioreactor_fermentation: expected 5 missing glucose values (in column 0), got {n_nan} NaNs total."
            )

        return spectra, raman_shifts_ref, targets, list(MiscLoader._YEAST_BIOREACTOR_TARGETS), group_ids

    @staticmethod
    def _load_sop_spectral_library(cache_path: str, variant: str = "baseline_corrected"):
        """
        Load the SOP (Synthetic Organic Pigments) Spectral Library.

        The library contains ~300 Raman reference spectra of synthetic organic pigments.
        Two variants are available: 'baseline_corrected' and 'raw'.

        The data is downloaded from SharePoint as a ZIP archive containing one
        TXT file per spectrum. Each TXT file has two columns (wavenumber, intensity)
        and the pigment name is encoded in the filename.

        Args:
            cache_path: Path to the cached dataset directory.
            variant: Either 'baseline_corrected' or 'raw'.

        Returns:
            Tuple of (spectra, raman_shifts, targets, class_names) or None if loading fails.

        Classification target: each filename encodes a specific pigment (e.g.
        "PB12_A_785"), essentially unique per sample -- using that directly as the
        label would give ~300 singleton classes, useless for any train/test split.
        The leading letters (before the first digit) encode a coarser color-family
        grouping shared by many pigments instead (PR=red, PY=yellow, PO=orange,
        PV=violet, PB=blue, PBR=brown, PG=green, PBK=black) -- that's the real,
        learnable classification target used here.
        """
        urls = {
            "baseline_corrected": "https://kikirpa-my.sharepoint.com/:u:/g/personal/wim_fremout_kikirpa_be/ES5_J9PpBatLvbTe6VlFyIoBc6fFRli0YHl2qjnLxn6I8Q?download=1",
            "raw": "https://kikirpa-my.sharepoint.com/:u:/g/personal/wim_fremout_kikirpa_be/EdVQqPRI0l1FgNy1lemrYwwBaNcH2Jan0ZuKa04UnBqAWA?download=1",
        }

        if variant not in urls:
            raise Exception(f"Unknown variant: {variant}")

        cache_root = LoaderTools.get_cache_root(CACHE_DIR.Misc)
        if cache_root is None:
            raise Exception(f"No cache root found for {cache_path}")

        shared_root = os.path.join(cache_root, "sop_spectral_library")
        os.makedirs(shared_root, exist_ok=True)

        zip_name = f"sop_{variant}.zip"
        zip_path = os.path.join(shared_root, zip_name)
        extracted_dir = os.path.join(shared_root, f"sop_{variant}")

        # Check if already extracted
        if not os.path.isdir(extracted_dir) or not os.listdir(extracted_dir):
            # Try to find an existing zip if the expected one doesn't exist
            if not os.path.exists(zip_path):
                existing_zips = glob.glob(os.path.join(shared_root, f"*{variant}*.zip"))
                if existing_zips:
                    zip_path = existing_zips[0]
                    MiscLoader.logger.debug(f"Found existing SOP zip: {zip_path}")

            # Download if not present
            if not os.path.exists(zip_path):
                try:
                    LoaderTools.download(
                        url=urls[variant],
                        out_dir_path=shared_root,
                        out_file_name=zip_name,
                    )
                except Exception as e:
                    MiscLoader.logger.error(
                        f"[!] Failed to download SOP Spectral Library ({variant}): {e}\n"
                        f"    Please download manually from:\n"
                        f"    {urls[variant]}\n"
                        f"    and place the ZIP at: {zip_path}"
                    )
                    raise e

            if not os.path.exists(zip_path):
                raise Exception(f"Failed to obtain SOP zip file for variant '{variant}'. Expected at: {zip_path}")

            # Extract
            LoaderTools.extract_zip_file_content(
                zip_path,
                unzip_target_subdir=f"sop_{variant}"
            )

        if not os.path.isdir(extracted_dir):
            raise FileNotFoundError(f"Failed to extract SOP Spectral Library ({variant})")

        # Collect all TXT files recursively
        txt_files = glob.glob(os.path.join(extracted_dir, "**", "*.txt"), recursive=True)
        if not txt_files:
            raise FileNotFoundError(f"Failed to extract SOP Spectral Library ({variant})")

        spectra_list = []
        raman_shifts_list = []
        pigment_labels = []

        txt_file = ""
        for txt_file in sorted(txt_files):
            try:
                # Try common delimiters for two-column data
                data = None
                for sep in ["\t", ",", ";", r"\s+"]:
                    try:
                        candidate = pd.read_csv(
                            txt_file, sep=sep, header=None, comment="#",
                            engine="python", skip_blank_lines=True
                        )
                        if candidate.shape[1] >= 2:
                            data = candidate
                            break
                    except Exception:
                        continue

                if data is None or data.shape[1] < 2:
                    MiscLoader.logger.warning(f"[!] Skipping unparseable file: {txt_file}")
                    continue

                wavenumbers = data.iloc[:, 0].values.astype(float)
                intensities = data.iloc[:, 1].values.astype(float)

                # Sort by ascending wavenumber
                sort_idx = np.argsort(wavenumbers)
                wavenumbers = wavenumbers[sort_idx]
                intensities = intensities[sort_idx]

                # Extract pigment name from filename, dropping institution
                # suffix. E.g. "PO14_A_785_kikirpa (original).txt" -> "PO14_A_785"
                pigment_name = os.path.splitext(os.path.basename(txt_file))[0]
                pigment_name = pigment_name.split("_kikirpa")[0].strip("_ ")

                spectra_list.append(intensities)
                raman_shifts_list.append(wavenumbers)
                pigment_labels.append(pigment_name)

            except Exception as e:
                MiscLoader.logger.warning(f"[!] Failed to parse {txt_file}: {e}")
                continue

        if len(spectra_list) == 0:
            raise FileNotFoundError(f"Failed to parse {txt_file}")

        # Check if all spectra share the same wavenumber axis
        first_rs = raman_shifts_list[0]
        all_equal = (
            all(len(first_rs) == len(rs) for rs in raman_shifts_list)
            and all(np.allclose(first_rs, rs) for rs in raman_shifts_list)
        )

        if all_equal:
            raman_shifts = np.array(first_rs, dtype=float)
            spectra = np.stack(spectra_list)
        else:
            # Interpolate all spectra to a common wavenumber grid
            raman_shifts, spectra = LoaderTools.align_raman_shifts(raman_shifts_list, spectra_list)

        # Color-family grouping (see this method's docstring) -- the leading
        # alphabetic run of each pigment code, e.g. "PB12_A_785" -> "PB",
        # "PBR14_A_785" -> "PBR". A plain regex match is safe here even though
        # "PB" is itself a prefix of "PBR"/"PBK": it greedily consumes every
        # leading letter before the first digit, so "PBR14..." matches "PBR"
        # in full, never truncating to "PB".
        color_families = pd.Series(pigment_labels).str.extract(r"^([A-Za-z]+)", expand=False)
        encoded_targets, class_names = encode_labels(color_families)

        MiscLoader.logger.debug(
            f"Loaded SOP library ({variant}): {spectra.shape[0]} spectra, "
            f"{spectra.shape[1]} wavenumber points, "
            f"{len(class_names)} unique pigments"
        )

        return spectra, raman_shifts, encoded_targets, list(class_names)

    # ------------------------------------------------------------------
    # NIST EC-SERS drug & adulterant library (doi:10.18434/mds2-3502)
    # ------------------------------------------------------------------
    # (filename, size in bytes, sha256) for all 131 deposited CSV files,
    # pinned from the NERD record v1.0.1 (modified 2024-08-28) of
    # https://doi.org/10.18434/mds2-3502. The NIST Open Data API no
    # longer serves the file list for this record, so the manifest is
    # embedded here; the direct file URLs remain the download source.
    _NIST_ECSERS_BASE_URL = (
        "https://data.nist.gov/od/ds/ark:/88434/mds2-3502/"
    )
    _NIST_ECSERS_FILES = (
        ("4-ANPP_5ppm_nontarget_Spectrum_326.csv", 30116, "d07d5bfec130127c115610c902d97fc9deaec1a4fda976b1f13ad1286c3b0f98"),
        ("Acetyl_fentanyl_10ppm_nontarget_Spectrum_295.csv", 30253, "9b89eea3456912f6783407b2c1f2820466f312ceda5005528f1e6e6b6018deac"),
        ("Acryl_fentanyl_nontarget_Spectrum_292.csv", 30455, "3163da524abbe419f7d19897fed14bf463cbf666c714649ac16f65daa82a0cd5"),
        ("Alprazolam_100ppm_nontarget_Spectrum_6.csv", 31225, "90c8981ed4d3478749a88f58317fdba53171b569e99ba186ed63e22ddd0be661"),
        ("Alprazolam_100ppm_nontarget_Spectrum_7.csv", 31481, "5f19c1938a58cf29e40f7feb406a5574bf5d104e10d04b1eb3bd82dbc2c0fa51"),
        ("Alprazolam_10ppm_nontarget_rep1_Spectrum_6.csv", 31472, "2f788cde4db78b856078c6334efbf210d9e38cb1b019d098a9ac0a1b45b5ae9a"),
        ("Alprazolam_10ppm_nontarget_rep1_Spectrum_7.csv", 31621, "356cf02ceeafabb03e6f91c30556bddeed7e11f3f80841c00ce04628c61a31d2"),
        ("Alprazolam_10ppm_nontarget_rep2_Spectrum_6.csv", 31441, "9e04eccdc8d93b1f972ec073287bc2e7c103eced99a0a9b8dd2632e518f5aadc"),
        ("Alprazolam_10ppm_nontarget_rep2_Spectrum_7.csv", 31567, "d523e2a8214d9a39746b417f2b1e2c81a957009a480d36d6964dfeb267cb15f2"),
        ("Alprazolam_10ppm_nontarget_rep3_Spectrum_6.csv", 31418, "6a6fd978576d7802222f7ecfd751cb28583bcd1b78247c0a2dc7d1dd4c0ec205"),
        ("Alprazolam_10ppm_nontarget_rep3_Spectrum_7.csv", 31612, "2c303773033beaa36a79afccc467a1056d91d21fe89fbb4fbc29147104c6969a"),
        ("Alprazolam_333ppm_nontarget_Spectrum_6.csv", 31411, "eefdfc069600b6f7f3e035b0d1f9998b2de249d9ebdb80203e876249d73ddf4c"),
        ("Alprazolam_333ppm_nontarget_Spectrum_7.csv", 31603, "6b040d9fae635fcf085133932f5bc5b697776347418ebf6224f57722decf0659"),
        ("Alprazolam_5ppm_nontarget_rep1_Spectrum_7.csv", 31359, "72d22527c638a366e5cb11f980f82a99a7bdf872ec5c22d1395a228a19d8031e"),
        ("Alprazolam_5ppm_nontarget_rep2_Spectrum_7.csv", 31316, "bab5c89fe71152061ba49ed0ddb26b0ddf573b492bbaa8f222d1611131bab4f3"),
        ("Alprazolam_5ppm_nontarget_rep3_Spectrum_7.csv", 31508, "8577c482b7af772597fc777312f310a1218aa82c9fbe2c5f1b58538e36ff7282"),
        ("Buprenorphine_100ppm_nontarget_Spectrum_6.csv", 31058, "fc4084f6c4d7c576b583f675d07ec3af3e5c113e96354a1b8438699200ab5a8b"),
        ("Cocaine_1000ppm_nontarget_rep1_Spectrum_6.csv", 31330, "56d5ec068ddd4e7fa6e2074448ed21bf7533ba3c406b5c59a5aad436797faae7"),
        ("Cocaine_1000ppm_nontarget_rep1_Spectrum_7.csv", 31332, "596ac3bbb1b7c2f2bc1b1227256119f3e1b4cd2f12a8e7f051ceb4b8852c85ed"),
        ("Cocaine_1000ppm_nontarget_rep2_Spectrum_6.csv", 31363, "cf75244764cfb20a96a6f05939fd474bdb0828e2f33793911d415f8c02424ca2"),
        ("Cocaine_1000ppm_nontarget_rep2_Spectrum_7.csv", 31352, "4c03c70676ab52147e3d26fdd943b581e6d126691f6262eb8b1c1ce9b0ebac70"),
        ("Cocaine_1000ppm_nontarget_rep3_Spectrum_6.csv", 31314, "ada3ada75b26f81472f64ebff7854d4d32f6bb34c2c68f1a3b2f728afc3c4e82"),
        ("Cocaine_1000ppm_nontarget_rep3_Spectrum_7.csv", 31314, "0bf49a6332766e3ff88fd13e300979503c64299d0c6dd36559eca49257473cde"),
        ("Cocaine_100ppm_nontarget_rep1_Spectrum_5.csv", 31097, "f9791296931e50bf95d8f37c9eb178e9682c7c0133a83a34cb15f10872b89cca"),
        ("Cocaine_100ppm_nontarget_rep2_Spectrum_5.csv", 31096, "14adcf3365ce99f20d0b8f53179bd2e1da823046c990607f29259becde225031"),
        ("Cocaine_100ppm_nontarget_rep3_Spectrum_5.csv", 31141, "6b7b8abfaf178b0eb1f2f092f7a3126920129c477a41de231ae90744c2d20f1a"),
        ("Cocaine_50ppm_nontarget_rep1_Spectrum_5.csv", 30986, "9046eb4ac83dea68dc73b5798ba25ca5c6112efb952223697b3e88b4ed74f19e"),
        ("Cocaine_50ppm_nontarget_rep2_Spectrum_5.csv", 31036, "01798448d15c76c9510e066f6cb18d9c353f71554bb6cfd172407301eaafe771"),
        ("Cocaine_50ppm_nontarget_rep3_Spectrum_5.csv", 31158, "ecd72152fd14642af2645b58f63f105389baa1a0626e3bd59905fa8a4da64365"),
        ("Codeine_1000_ppm_nontarget_rep1_Spectrum_7.csv", 31237, "b11f421c53a54e8fe0004198e345249c8415792d20e7f700f3524dc132939b2a"),
        ("Codeine_1000_ppm_nontarget_rep2_Spectrum_7.csv", 31359, "c5f77e9dd2ddabd2394d409337037f0db59ff1b935bbd905cd9bfd366f4c97af"),
        ("Codeine_1000_ppm_nontarget_rep3_Spectrum_7.csv", 31321, "3e75321cbdf8c7362783671fc94d477d38feb1c52e476dafe772d4e2b5fdf190"),
        ("Codeine_100ppm_nontarget_rep1_Spectrum_5.csv", 31138, "3793afe78aa544d42edb4374be8fc1c84af2a9d79c7cdf7d88739fad665d9375"),
        ("Codeine_100ppm_nontarget_rep2_Spectrum_5.csv", 31189, "41626473df48e301296110adec583e2eff505e3f140767d604c6747bab02063c"),
        ("Codeine_100ppm_nontarget_rep3_Spectrum_5.csv", 31240, "320da487be6639dbb6b0728bcabfecc35211606749b75c524f513df09e6e1d2e"),
        ("Codeine_300ppm_nontarget_rep1_Spectrum_6.csv", 31311, "4829790278fa0ed48f9af2264a1bc6fc05c7836403e3b0cdb4583be5356cd9a6"),
        ("Codeine_300ppm_nontarget_rep2_Spectrum_6.csv", 31285, "004007ce712b06bf9b061960135ff466cc07f5f5136241f3d8e68fe7fdad36b6"),
        ("Codeine_300ppm_nontarget_rep3_Spectrum_6.csv", 31319, "a2b0b17fabe50073aae24f12f7a2680b6a3a731f94798872c6601cd0462e3ed1"),
        ("Dextromethorphan_100ppm_nontarget_rep1_Spectrum_6.csv", 31377, "9b9ec05f577823edd5428f8abd9e2f846d07a2b6f0e4c06e93960d5d3bc7587b"),
        ("Dextromethorphan_100ppm_nontarget_rep2_Spectrum_6.csv", 31296, "bd409e05941cf6b37b9482d6a0c6c196ac83a7f6c6a481519ba142dcb2f86604"),
        ("Dextromethorphan_100ppm_nontarget_rep3_Spectrum_6.csv", 31365, "45385eea38b70131c7c52c61d3c0d194a4ac4caa649d2af221220411a44afdbb"),
        ("Dextromethorphan_10ppm_nontarget_Spectrum_6.csv", 31319, "26e99520ab469b83fba5fc1949ebd3797e6c83141ee4679f4332ec172e09cf1b"),
        ("Diltiazem_100ppm_nontarget_Spectrum_5.csv", 31357, "ccfd4da6f2ea5e9a14ac23c11e26614d4eeeb5ed871c5aef73e9dbc3f4b4a630"),
        ("Diltiazem_10ppm_nontarget_Spectrum_5.csv", 31480, "626a910329e5b571944f1273e7471426564fad18762deac188f1c2bd9fc4b550"),
        ("Diphenhydramine_10ppm_nontarget_rep1_Spectrum_9.csv", 31250, "bd75c9253b2cfcf0afa03eec0de2fa463dbb2b83b030bf1d56dfa33cf8ffc3aa"),
        ("Diphenhydramine_10ppm_nontarget_rep2_Spectrum_9.csv", 31253, "71d1627165d63a7babc576da063a154cf05825672625b217efb0c0a5ba95fadb"),
        ("Diphenhydramine_10ppm_nontarget_rep3_Spectrum_9.csv", 31330, "6ed0408904f18fd5d8bfa788603dcfc525750a4b3083c218f5332d115ae6d097"),
        ("Diphenhydramine_1ppm_nontarget_Spectrum_5.csv", 31139, "c316d4f1a24800d31611f4cd03f07f5f36f63fe322e6f61816bfd710ec9207dd"),
        ("Etizolam_50ppm_nontarget_rep1_Spectrum_6.csv", 31501, "4be649ff2eade589416c3a260b620da5d25dc80e34a2cbc5ea24bcbfd7d54681"),
        ("Etizolam_50ppm_nontarget_rep2_Spectrum_6.csv", 31470, "418fecd22b747a8de78c24e2e711510c7e2de641a6df7d548d3df37298c1fd7e"),
        ("Etizolam_50ppm_nontarget_rep3_Spectrum_6.csv", 31497, "1955094395428c5d844bdcb9ec2797cfdec48119dd456c9a9e255c652b0aa40b"),
        ("Etonitazene_100ppm_nontarget_rep1_Spectrum_6.csv", 31520, "e5de60063f37e4e5918d5a5fcbbb359878831c71ed60a2cb34ce4be4cbd3ac6f"),
        ("Etonitazene_100ppm_nontarget_rep2_Spectrum_6.csv", 31539, "864cf1fd82f2f98929de89401997eb654d97044c4a1f018180b673ad587b93e7"),
        ("Etonitazene_100ppm_nontarget_rep3_Spectrum_6.csv", 31488, "1d3922c81ba9bc1f07c0bf78b165ea4007b8de5546793fb416212804c1891a42"),
        ("Fentanyl_1000ppm_nontarget_rep1_Spectrum_6.csv", 31167, "cb15e0680fd0b2cb081a73f5c897ff21715493a10b21524aca19e6e9f6538c68"),
        ("Fentanyl_1000ppm_nontarget_rep1_Spectrum_7.csv", 31417, "4d7538a455b05dfa3a7151dbf7cfe48344983c7fa8e872256c6668f4eb436dfb"),
        ("Fentanyl_1000ppm_nontarget_rep2_Spectrum_6.csv", 31183, "8b8b8c0915b47e3f5fb4241b9bc507373aaaaa059cdedc90a3a66659af68a736"),
        ("Fentanyl_1000ppm_nontarget_rep2_Spectrum_7.csv", 31384, "52d8183ae1d64365b0ae3eeab70092fc281dd97577b1677cd06b96b83cae72f4"),
        ("Fentanyl_1000ppm_nontarget_rep3_Spectrum_6.csv", 31102, "25aa11df66fff333e09560a2934baf05d5b07574cf6fc954cb1afbb471de2018"),
        ("Fentanyl_1000ppm_nontarget_rep3_Spectrum_7.csv", 31362, "044642b06b2f63878087b747a3c3a2d5e0587939258618e98c3e36b741d73313"),
        ("Fentanyl_100ppm_nontarget_Spectrum_5.csv", 31447, "b19d0153976f6f4e61c7c9c1cc0db48aa40412d9dd8bbcf02b44bb197854490e"),
        ("Fentanyl_10ppm_nontarget_Spectrum_7.csv", 31336, "6f3d9e27f52e7140da57756871a7d1776e3ea5b8888cb046a9c0c667048e2225"),
        ("Fentanyl_1ppm_nontargeT_rep3_Spectrum_5.csv", 31116, "f016dd4597d8517fbc7c157c73a301242b5f6c32c1f3f80980792746ef477bb2"),
        ("Fentanyl_1ppm_nontarget_rep1_Spectrum_5.csv", 31184, "c96fa6b9729c7c61782112f1c80ba7863d4245c171008b7c7446c75b3871ccf3"),
        ("Fentanyl_1ppm_nontarget_rep2_Spectrum_5.csv", 31316, "b1991f0b0c3985167e8a368f4a2207ab0362d89e29da4d8b0802f50aeda26952"),
        ("Fentanyl_1ppm_nontarget_rep4_Spectrum_5.csv", 31189, "ee85fbc28641022c04c1d7bfab0bb5c7319204629a8e66fa8cd205bf2f0eb301"),
        ("Fentanyl_1ppm_nontarget_rep5_Spectrum_5.csv", 31116, "e2e17e139b1159ac0606828c69d73dd270982fc0be2e88e914ba89520130127e"),
        ("Furanyl_fentanyl_nontarget_Spectrum_293.csv", 30442, "cc53061337f9041dc0528c6d16a54b83bcf5385d22dd8413d73c05bbfee987c8"),
        ("Heroin_1001ppm_nontarget_rep1_Spectrum_6.csv", 31003, "0312c6c89a157d360288052ede2212ff28c29be9a517141daafa7f1e35c4871c"),
        ("Heroin_1001ppm_nontarget_rep2_Spectrum_6.csv", 30951, "b78f68c6a02894913ea5fb36a2bce355d304372b14ba195ec243841592197938"),
        ("Heroin_1001ppm_nontarget_rep3_Spectrum_6.csv", 31082, "14aff06269c7758f49ff66aee23829ea082a2dd0f5f15b14dfe3566387a70e53"),
        ("Heroin_1001ppm_nontarget_rep4_Spectrum_6.csv", 31052, "58aaebc388882b8442d7f16850fd7040805c63fd2d89b24f1c4c4016b69429d8"),
        ("Heroin_100ppm_nontarget_rep1_Spectrum_5.csv", 31071, "9b15ff5d28514f669a126b7053cddd079c9d1f1ece73b7a42eb283a996fdebcf"),
        ("Heroin_100ppm_nontarget_rep2_Spectrum_5.csv", 31086, "39d4c1f96899e2ae6b968ae599bbf9140e3f08d64b0bf96f9441865297bebf08"),
        ("Heroin_100ppm_nontarget_rep3_Spectrum_5.csv", 31090, "6333c493955034f8b22c6b1b7ee2c83d53e3cbeccb16fb055806407b3900e727"),
        ("Heroin_300ppm_nontarget_rep1_Spectrum_6.csv", 31198, "b4e453eafa1383494f3aa8c11b3bf6fde284d4d8561cfeaf2352fc36ccc8d234"),
        ("Heroin_300ppm_nontarget_rep2_Spectrum_6.csv", 31190, "2c0a89fb066dc8e728f02ac23c70020f68a078382cf046f2fe835e741f89bd53"),
        ("Heroin_300ppm_nontarget_rep3_Spectrum_6.csv", 31159, "3e1254076610be1227660c5fcfdf93fd4d975900c83a48acdb0eb62a6b62a9aa"),
        ("Hydroxyzine_5ppm_nontarget_Spectrum_6.csv", 31467, "e1d62110b49e003999eab37212c12a3925e3a356f1daa93af63b6569392558b5"),
        ("Levamisole_1001_ppm_nontarget_rep1_Spectrum_6.csv", 31400, "b6f237d569caa5927081e32c359bcbc244406b6754e17ba7aaa87563b7c57c22"),
        ("Levamisole_1001_ppm_nontarget_rep2_Spectrum_6.csv", 31452, "c16c3b894b9da61cf9c0ac8cd3a9964d2b54fc5c56344b5ac750cf1dcf81c9d7"),
        ("Levamisole_1001_ppm_nontarget_rep3_Spectrum_6.csv", 31432, "8de60d8ec33e9c98f70e5817d05766dcf1d3f51f24edcd3040bdfce63ea59ef8"),
        ("Levamisole_100ppm_nontarget_Spectrum_6.csv", 31564, "995952226dc400c3eb77a0ccffc0dbacd6b9f0646a9ffb829e364eadf72cea8b"),
        ("Levamisole_10ppm_nontarget_rep1_Spectrum_6.csv", 31222, "dfaf7b01741f4d5cbd8966ba012f392f571708599b4b0d4963077daae5dcb2b1"),
        ("Levamisole_10ppm_nontarget_rep2_Spectrum_6.csv", 31086, "17bcf23ceaec5929789718512c1db0d436b416035a4d283bc5859cf28edf632c"),
        ("Levamisole_10ppm_nontarget_rep3_Spectrum_6.csv", 31199, "d6b5c1fa93d117e42bad3f951947be4196395c25552f014846c16c80ac5914f1"),
        ("Lidocaine_1001_ppm_nontarget_rep1_Spectrum_6.csv", 31327, "0d7ddab73bfd3d04f8e75b968ca0fc40bceeb46cfc20f564467c995ff451e03b"),
        ("Lidocaine_1001_ppm_nontarget_rep2_Spectrum_6.csv", 31371, "5d648186a45962438bb5c6e3efe8e36f3d364809bc7598084c8accc614eb22d3"),
        ("Lidocaine_1001_ppm_nontarget_rep3_Spectrum_6.csv", 31428, "05af2f490b5bba111e6f83f37e8e951a533a93a5042c3701bd044eb0df0f5b5e"),
        ("Methadone_50ppm_nontarget_rep1_Spectrum_6.csv", 31312, "f3ba1058283deecb13054d5d24f1e7450f5b761cc867133f5e5ca0d884fb7f9e"),
        ("Methadone_50ppm_nontarget_rep2_Spectrum_6.csv", 31371, "83b245ced8f1875eab5eaa5fd85c3594c054982863205a192d448b847f75c82d"),
        ("Methadone_50ppm_nontarget_rep3_Spectrum_6.csv", 31268, "691db6bb166b46492388c49b6433aacdfa41d055953ca37c5e2f86ac8a109fe0"),
        ("Methamphetamine_1000ppm_nontarget_rep1_Spectrum_5.csv", 30938, "2b659e71ca77ef1f43f62f2229b89e37eab09265bddcc3ef6ae720bd462f31a9"),
        ("Methamphetamine_1000ppm_nontarget_rep1_Spectrum_6.csv", 30854, "921eda55116482a4ea66e5403fa37301da7c4c2c3c7ac6600af66f5bbe4d71de"),
        ("Methamphetamine_1000ppm_nontarget_rep1_Spectrum_7.csv", 30799, "3eeeb0996d3c292f7033703a93346fd92052c61b805aabb5b4c271fa7ee205f0"),
        ("Methamphetamine_1000ppm_nontarget_rep2_Spectrum_5.csv", 30925, "d4e510e8b84872436e6ace2542077d0912604a8a89fa1e1b91ffc80e71c6ed58"),
        ("Methamphetamine_1000ppm_nontarget_rep3_Spectrum_5.csv", 30920, "354b85224f9285a01db8f295cd695dcc8febeee8c97ae116bb2f2ef56d4fb381"),
        ("Methoxyacetyl_fentanyl_25ppm_nontarget_Spectrum_293.csv", 30185, "0ca5bb1e7b7010e679d118e197413071087bce8ea6c8430825d8aa23df6bbd99"),
        ("Morphine_1000_ppm_nontarget_rep1_Spectrum_7.csv", 31337, "f43fa9161f1c78776fe0f06bee3f4284f003a68636d9bb9f12f7bfb174832128"),
        ("Morphine_1000_ppm_nontarget_rep2_Spectrum_7.csv", 31331, "5f8011850995113619785b4000724f28020cf03daf541676c468a352be84c7a4"),
        ("Morphine_1000_ppm_nontarget_rep3_Spectrum_7.csv", 31286, "649d3071aa962ca2a03658931f25883a6cf5cb4b842176150bfc45866fa87d4c"),
        ("Morphine_100ppm_nontarget_Spectrum_5.csv", 31068, "ad9c758058d89534907e4f357c32943296be4d0e4daf382f26904197a28facbe"),
        ("Morphine_333ppm_nontarget_Spectrum_5.csv", 31168, "a769c3322bdb7e05f9fb01507ccedabe80cba22a9cac6b445220ef1450c3869d"),
        ("Naltrexone_1000ppm_Spectrum_7.csv", 31247, "3d9be1afaef94fc36fe8539e32162ccbf96c7038e524adf8c297aabbcbdb2768"),
        ("Nicotine_100ppm_nontarget_rep1_Spectrum_7.csv", 31204, "fd8161c3f7aa349f487e40b39b1d3f42869bccb51fa80ba3465b2dcf4bf8a559"),
        ("Nicotine_100ppm_nontarget_rep2_Spectrum_7.csv", 31133, "d6ef1731452659c09f33468f245e956667e5009a06ab38a4e05e391a0ef8720b"),
        ("Nicotine_100ppm_nontarget_rep3_Spectrum_7.csv", 31133, "928df46291916489e84a2515703a7a7c9403ff670f7ca2ff9336b6aef835b2e6"),
        ("Npiperidinylmetonitazene_100ppm_nontarget_rep1_Spectrum_6.csv", 31321, "f90bc656fecb0ff2f22a9746541fb479cff1734224150bb15d3a659381f94d53"),
        ("Npiperidinylmetonitazene_100ppm_nontarget_rep2_Spectrum_6.csv", 31216, "2da6096a7c331abc0011241ccad3799326f1a80089d696aa7f442ee9f9bcfc11"),
        ("Npiperidinylmetonitazene_100ppm_nontarget_rep3_Spectrum_6.csv", 31159, "b5fb6788e430b0cc206dae9449f16aeee0472273dc862b0f336356263a6b7c7e"),
        ("Npyrrolidinoetonitazene_100ppm_nontarget_rep1_Spectrum_6.csv", 31314, "68cf1b0355809b847c2c345e796b41aaa7ddb641175993351a7fe6d55843a52b"),
        ("Npyrrolidinoetonitazene_100ppm_nontarget_rep2_Spectrum_6.csv", 31292, "a3151aa2ed1551c6ddf680bda2f82f1494df972dbceeb7a65f4372bfee1f96a5"),
        ("Npyrrolidinoetonitazene_100ppm_nontarget_rep3_Spectrum_6.csv", 31302, "a5720a637b3ed8266ec48fcad05dcc4f3c040da9fa59a981134e1fd6358511ac"),
        ("Procaine_form_solid_nontarget_rep1_Spectrum_7.csv", 20361, "32dfc81f72d7d98e0ae429d96ecada5d4e7a4f1518ea1c9989578e198367e274"),
        ("Procaine_form_solid_nontarget_rep2_Spectrum_7.csv", 20673, "4fd3430e73e44a27f2f6c542ca054c3e4e341e23934623773786f94130cce2cc"),
        ("Procaine_form_solid_nontarget_rep3_Spectrum_7.csv", 20656, "a23eb4de89c8b600215063bb64ea4d7456c72ba0b91a551c3469e71d37c18e95"),
        ("Quinine_1013ppm_nontarget_rep1_Spectrum_6.csv", 31122, "c520d36644837b0927408ac2773f9d4ed61bb91671ca6d0d7ec1cd8f4eb1063b"),
        ("Quinine_1013ppm_nontarget_rep2_Spectrum_6.csv", 31110, "bf1234b8a2e63b49f17c73696a61f8b6a22ccd9bcc49b8ab0641d079e59a1fc9"),
        ("Quinine_1013ppm_nontarget_rep3_Spectrum_6.csv", 31042, "7ffe3788f86c5178c6748c0a293c149dde8f22fdd032e87578e4d63cc35ca5f2"),
        ("Sufentanil_100ppm_nontarget_Spectrum_6.csv", 31419, "e916464bb54d40c4c72074e87878bab8db6810d1f50c1e14667eab32479c1494"),
        ("Sufentanil_10ppm_nontarget_rep1_Spectrum_6.csv", 31087, "5aeda819df7c37ae6d752a16ca829d57675c7c5234bbccb5354d00ea1f331f0f"),
        ("Sufentanil_10ppm_nontarget_rep2_Spectrum_6.csv", 31229, "77e0228eb90ad58911b57c9cafacc611399447d4365106b8111d8830860eb260"),
        ("Sufentanil_10ppm_nontarget_rep3_Spectrum_6.csv", 31318, "980a1e8fbf365b20fe4f9e41838c55897a6cc1767897244c5c6b7e46de8c11b1"),
        ("Sufentanil_333ppm_nontarget_Spectrum_6.csv", 31426, "7a6f5cf8d70503555f9164d365504beccb3d1b608b3142f770dc18408233d5ab"),
        ("Temazepam_100ppm_nontarget_rep1_Spectrum_7.csv", 31194, "2953ef4e84cf55f16820c8603d5bebc9ae11e8d69e177f970673a0afb25c9760"),
        ("Temazepam_100ppm_nontarget_rep2_Spectrum_7.csv", 31240, "810b26c8eca8ae2518a82bf26c76eb2ed3d34f538b2a27c3f5a9c4f12990803c"),
        ("Temazepam_100ppm_nontarget_rep3_Spectrum_7.csv", 30965, "7096a0c9335093020a40856d6047662ff67b863db5c71cc36218ee77a402f578"),
        ("Valeryl_fentanyl_25ppm_nontarget_Spectrum 293.csv", 30241, "f77fe22d40760193de549e81f780e2fee435b9fae22030f2ec79406aee9b789d"),
        ("Xylazine_1000ppm_nontarget_rep1_Spectrum_5.csv", 31321, "7c8a3bfc965ad2170e694015d2af89c77e545186107c73a2856d4443a9c3f5cf"),
        ("Xylazine_1000ppm_nontarget_rep2_Spectrum_5.csv", 31449, "7725c36d238eb229a7d8d4fda86f40d71fab02dccc98704649a9b987d82f15fd"),
        ("Xylazine_1000ppm_nontarget_rep3_Spectrum_5.csv", 31436, "c13f4e4ed1d7c984b4372fdee04b5bcc1cd27e823b70c6a88f5869097e2473a0"),
    )

    @staticmethod
    def _nist_ecsers_parse_name(filename: str) -> dict:
        """
        Parse a NIST EC-SERS CSV filename into its metadata parts.

        Filenames follow
        '<Substance>[_<conc>ppm][_form_solid][_nontarget|target][_rep<N>]
        [_Spectrum <n>].csv' with three irregularities in the deposit:
        one 'Spectrum 293' name uses a space instead of an underscore
        (Valeryl_fentanyl), one fentanyl name carries a 'nontargeT'
        typo, and one file (Naltrexone_1000ppm) has no target marker.
        The suffixes may appear in the order above only; stripping is
        therefore done from the end in reverse order.

        Returns a dict with keys: substance (str), conc (int ppm or
        None), form ('solid' or None), target ('nontarget'/'target' or
        None), rep (int or None), spectrum (int).
        """
        if not filename.endswith(".csv"):
            raise ValueError(f"nist_ecsers_drugs: not a CSV filename: {filename}")
        n = filename[:-4]
        m = re.search(r"[_ ]Spectrum[_ ](\d+)\s*$", n)
        if m is None:
            raise ValueError(f"nist_ecsers_drugs: unrecognized filename: {filename}")
        spectrum = int(m.group(1))
        n = n[:m.start()]
        m = re.search(r"_rep(\d+)$", n)
        rep = int(m.group(1)) if m else None
        if m:
            n = n[:m.start()]
        m = re.search(r"_(nontarge[tT]|target)$", n)
        target = None
        if m:
            target = "nontarget" if m.group(1).lower().startswith("non") else "target"
            n = n[:m.start()]
        form = None
        if n.endswith("_form_solid"):
            form = "solid"
            n = n[: -len("_form_solid")]
        m = re.search(r"_(\d{1,4})_?ppm$", n)
        conc = None
        if m:
            conc = int(m.group(1))
            n = n[:m.start()]
        if not n:
            raise ValueError(f"nist_ecsers_drugs: empty substance in {filename}")
        return {
            "substance": n,
            "conc": conc,
            "form": form,
            "target": target,
            "rep": rep,
            "spectrum": spectrum,
        }

    @staticmethod
    def _nist_ecsers_read_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read one deposited NIST EC-SERS CSV. Layout: line 1 = sample
        name, line 2 = column titles ('Raman Shift (cm-1)' /
        'Counts minus Dark (a.u.)'), then quoted ';'-separated
        (wavenumber, intensity) rows with thousands separators inside
        the quotes. The three procaine solid files end with a stray
        Origin 'Secondary axis;' footer line, which is skipped like
        any non-numeric line.

        Returns: (wavenumber axis, intensity), both 1-D float arrays.
        """
        rows = []
        with open(path, encoding="latin-1") as fh:
            lines = fh.read().splitlines()
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            cells = [c.strip().strip('"').replace(",", "") for c in line.split(";")]
            try:
                rows.append((float(cells[0]), float(cells[1])))
            except (ValueError, IndexError):
                continue
        if not rows:
            raise RuntimeError(f"nist_ecsers_drugs: no data rows in {path}")
        d = np.asarray(rows, dtype=float)
        return d[:, 0], d[:, 1]

    @staticmethod
    def _load_nist_ecsers_drugs(cache_path: str):
        """
        Download (if needed) and load the NIST EC-SERS drug &
        adulterant library (doi:10.18434/mds2-3502, v1.0.1): 131
        EC-SERS spectra of 32 substances (drugs of abuse and
        adulterants), 785 nm excitation on a DropSens SPELEC silver
        electrode. Each CSV is the highest-intensity / best-resolution
        spectrum of one replicate; values are 'Counts minus Dark
        (a.u.)' (dark-subtracted, NOT normalized).

        The deposit mixes two acquisition batches with different
        native grids (96 files: 1,014 points 32.6-2,859.1 cm^-1; 35
        files: 1,011 points 33.8-2,995.5 cm^-1), both non-uniformly
        spaced. All spectra are interpolated onto the common 1,013-
        point grid (36.6-2,859.1 cm^-1) that every file fully covers,
        so no extrapolation occurs.

        Returns spectra, raman_shifts, targets, class_names, group_ids
        (group id = substance | concentration-or-solid | replicate;
        unmarked measurements at one substance+concentration share a
        group).
        """
        os.makedirs(cache_path, exist_ok=True)
        for fname, size, sha256 in MiscLoader._NIST_ECSERS_FILES:
            path = os.path.join(cache_path, fname)
            if os.path.exists(path) and os.path.getsize(path) != size:
                # Truncated or replaced cache entry: force re-download.
                os.remove(path)
            result = LoaderTools.download(
                url=MiscLoader._NIST_ECSERS_BASE_URL + fname,
                out_dir_path=cache_path,
                out_file_name=fname,
                hash_target=sha256,
                hash_type=HASH_TYPE.sha256,
            )
            if result is None:
                raise RuntimeError(
                    f"[!] nist_ecsers_drugs: download of {fname} failed "
                    f"(network error or checksum mismatch). See logs."
                )

        spectra_list = []
        targets = []
        group_ids = []
        common_axis = None
        for fname, size, sha256 in MiscLoader._NIST_ECSERS_FILES:
            axis, inten = MiscLoader._nist_ecsers_read_csv(
                os.path.join(cache_path, fname)
            )
            if axis.size not in (1011, 1014):
                raise RuntimeError(
                    f"[!] nist_ecsers_drugs: {fname} has {axis.size} points, "
                    f"expected 1011 or 1014 (deposit layout changed?)."
                )
            if not np.all(np.diff(axis) > 0):
                raise RuntimeError(
                    f"[!] nist_ecsers_drugs: non-monotonic wavenumber axis "
                    f"in {fname}."
                )
            if common_axis is None:
                # First manifest entry is a 1,014-point batch file; drop
                # its single point below the other batch's range (33.8
                # cm^-1) to form the common grid.
                common_axis = axis[1:]
                if common_axis.size != 1013:
                    raise RuntimeError(
                        "[!] nist_ecsers_drugs: unexpected reference grid "
                        f"({common_axis.size} points)."
                    )
            if axis[0] > common_axis[0] or axis[-1] < common_axis[-1]:
                raise RuntimeError(
                    f"[!] nist_ecsers_drugs: {fname} does not cover the "
                    f"common grid {common_axis[0]:.2f}-{common_axis[-1]:.2f} "
                    f"cm^-1."
                )
            spectra_list.append(np.interp(common_axis, axis, inten))
            meta = MiscLoader._nist_ecsers_parse_name(fname)
            targets.append(meta["substance"])
            if meta["conc"] is not None:
                conc = meta["conc"]
            elif meta["form"] == "solid":
                conc = "solid"
            else:
                conc = "na"
            rep = meta["rep"] if meta["rep"] is not None else "unmarked"
            group_ids.append(f'{meta["substance"]}|{conc}|{rep}')

        encoded_targets, class_names = encode_labels(pd.Series(targets))
        return (
            np.asarray(spectra_list),
            common_axis,
            encoded_targets,
            list(class_names),
            group_ids,
        )

    # ------------------------------------------------------------------
    # Raman Open Database (ROD / SOLSA) -- frozen snapshot
    # ------------------------------------------------------------------
    # The live ROD (https://solsa.crystallography.net/rod/) grows
    # continuously, so this loader serves a frozen snapshot: the bulk
    # result.php?format=zip download of 2026-10-20 (1,133 CIF entry
    # files), mirrored on the raman_data HuggingFace mirror and pinned
    # by size + sha256. See croissant_files/../MANIFEST in the mirror
    # repo for the snapshot provenance.
    _ROD_MINERALS_SNAPSHOT_URL = (
        "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanBench/"
        "resolve/main/rod_minerals/snapshot/rod_all_2026-10-20.zip"
    )
    _ROD_MINERALS_SNAPSHOT_ZIP = "rod_all_2026-10-20.zip"
    _ROD_MINERALS_SNAPSHOT_SIZE = 24783194
    _ROD_MINERALS_SNAPSHOT_SHA256 = (
        "adc56ad40b9eefffb437c3ea3e79c9068276bafa75b682e962a578d44d4c8e48"
    )
    # Common analysis window (lo, hi, step) in cm^-1: 250-1,200 @1
    # (951 points) is fully covered by 98.5% of the snapshot's
    # experimental spectra; the spectra that do not fully cover it are
    # excluded rather than extrapolated.
    _ROD_MINERALS_GRID = (250.0, 1200.0, 1.0)

    @staticmethod
    def _rod_parse_cif(text: str) -> Tuple[dict, list]:
        """
        Minimal CIF parser for the .rod entry files.

        Returns (values, loops): values maps each top-level tag to
        its first value. The .rod files place a tag's value on the
        same line ('plain', 'quoted' or ';'-block, with or without
        inline content after the opening ';') or on the next line
        (quoted single line or ';'-block); all forms are handled.
        loops is a list of (column_tags, data_rows) pairs with the
        rows kept as stripped strings (cells are whitespace-separated).
        """
        values = {}
        loops = []
        lines = text.splitlines()
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if line.startswith("loop_"):
                i += 1
                cols = []
                while i < n and lines[i].startswith("_"):
                    cols.append(lines[i].strip())
                    i += 1
                rows = []
                while (
                    i < n
                    and lines[i].strip()
                    and not lines[i].startswith(("loop_", "_", "data_"))
                ):
                    rows.append(lines[i].strip())
                    i += 1
                loops.append((cols, rows))
                continue
            if line.startswith("_"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    tag, val = parts[0], parts[1].strip()
                    if val == ";":
                        # Multi-line ';'-block starting on the same
                        # line.
                        buf = []
                        i += 1
                        while i < n and lines[i].strip() != ";":
                            buf.append(lines[i])
                            i += 1
                        val = "\n".join(buf).strip()
                    elif val.startswith(";") and len(val) > 1:
                        # Multi-line block whose opening ';' carries
                        # inline content on the same line.
                        buf = [val[1:]]
                        i += 1
                        while i < n and lines[i].strip() != ";":
                            buf.append(lines[i])
                            i += 1
                        val = "\n".join(buf).strip()
                    elif (
                        len(val) >= 2
                        and val.startswith("'")
                        and val.endswith("'")
                    ):
                        val = val[1:-1]
                    values.setdefault(tag, val)
                    i += 1
                    continue
                # Bare tag: the value (if any) sits on the next
                # non-empty line -- a quoted single line or a
                # multi-line ';'-block (with or without inline
                # content after the opening ';').
                tag = line.strip()
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j < n:
                    nx = lines[j].strip()
                    if nx == ";":
                        i = j + 1
                        buf = []
                        while i < n and lines[i].strip() != ";":
                            buf.append(lines[i])
                            i += 1
                        values.setdefault(tag, "\n".join(buf).strip())
                        continue
                    if nx.startswith(";") and len(nx) > 1:
                        buf = [nx[1:]]
                        i = j + 1
                        while i < n and lines[i].strip() != ";":
                            buf.append(lines[i])
                            i += 1
                        values.setdefault(tag, "\n".join(buf).strip())
                        continue
                    if (
                        len(nx) >= 2
                        and nx.startswith("'")
                        and nx.endswith("'")
                    ):
                        values.setdefault(tag, nx[1:-1])
                        i = j
                        continue
            i += 1
        return values, loops

    @staticmethod
    def _load_rod_minerals(cache_path: str):
        """
        Download (if needed) and load the frozen 2026-10-20 snapshot
        of the Raman Open Database (ROD): 1,098 experimental mineral
        Raman spectra of 432 labeled mineral/phase entries, resampled
        onto the common 250-1,200 cm^-1 @1 cm^-1 grid (951 points).

        Excluded from the snapshot's 1,133 entries: 18 entries that
        only contain theoretical (DFT) spectra and the 17 experimental
        spectra that do not fully cover the common window. No
        group_ids (one spectrum per entry).

        Returns spectra, raman_shifts, targets, class_names.
        """
        os.makedirs(cache_path, exist_ok=True)
        zip_path = os.path.join(cache_path, MiscLoader._ROD_MINERALS_SNAPSHOT_ZIP)
        if (
            os.path.exists(zip_path)
            and os.path.getsize(zip_path) != MiscLoader._ROD_MINERALS_SNAPSHOT_SIZE
        ):
            os.remove(zip_path)
        result = LoaderTools.download(
            url=MiscLoader._ROD_MINERALS_SNAPSHOT_URL,
            out_dir_path=cache_path,
            out_file_name=MiscLoader._ROD_MINERALS_SNAPSHOT_ZIP,
            hash_target=MiscLoader._ROD_MINERALS_SNAPSHOT_SHA256,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                "[!] rod_minerals: snapshot download failed (network error "
                "or checksum mismatch). See logs."
            )
        extract_root = os.path.join(cache_path, "extracted")
        if not glob.glob(os.path.join(extract_root, "*.rod")):
            if LoaderTools.extract_zip_file_content(zip_path, "extracted") is None:
                raise RuntimeError(
                    "[!] rod_minerals: could not extract the snapshot zip."
                )
        rod_files = sorted(glob.glob(os.path.join(extract_root, "*.rod")))
        if len(rod_files) != 1133:
            raise RuntimeError(
                f"[!] rod_minerals: expected 1133 .rod files in the snapshot, "
                f"found {len(rod_files)} (snapshot layout changed?)."
            )

        lo, hi, step = MiscLoader._ROD_MINERALS_GRID
        grid = np.arange(lo, hi + step / 2, step)
        spectra_list = []
        labels = []
        n_theoretical_only = 0
        n_unnamed = 0
        n_partial = 0
        for path in rod_files:
            with open(path, encoding="latin-1") as fh:
                values, loops = MiscLoader._rod_parse_cif(fh.read())
            spec = None
            for cols, rows in loops:
                if cols == [
                    "_raman_spectrum.raman_shift",
                    "_raman_spectrum.intensity",
                ]:
                    pts = []
                    for r in rows:
                        p = r.split()
                        if len(p) >= 2:
                            try:
                                pts.append((float(p[0]), float(p[1])))
                            except ValueError:
                                pass
                    if pts:
                        spec = np.asarray(pts, dtype=float)
            if spec is None:
                # No experimental spectrum (theoretical-only entry or
                # malformed loop).
                n_theoretical_only += 1
                continue
            label = (
                values.get("_chemical_name_mineral")
                or values.get("_chemical_name_systematic")
                or values.get("_chemical_formula_sum")
                or ""
            )
            # Multi-line ';'-block label values may contain embedded
            # newlines; collapse all whitespace runs to single spaces.
            label = re.sub(r"\s+", " ", label).strip()
            if not label:
                n_unnamed += 1
                continue
            if spec[0, 0] > lo or spec[-1, 0] < hi:
                n_partial += 1
                continue
            if not (np.all(np.diff(spec[:, 0]) < 0) or np.all(np.diff(spec[:, 0]) > 0)):
                raise RuntimeError(
                    f"[!] rod_minerals: non-monotonic wavenumber axis in "
                    f"{os.path.basename(path)}."
                )
            spectra_list.append(np.interp(grid, spec[:, 0], spec[:, 1]))
            labels.append(label)
        if len(spectra_list) != 1098:
            raise RuntimeError(
                f"[!] rod_minerals: expected 1098 spectra, got "
                f"{len(spectra_list)} (theoretical-only: {n_theoretical_only}, "
                f"unnamed: {n_unnamed}, partial window: {n_partial})."
            )
        encoded_targets, class_names = encode_labels(pd.Series(labels))
        return (
            np.asarray(spectra_list),
            grid,
            encoded_targets,
            list(class_names),
        )

    # ------------------------------------------------------------------
    # olive_sunflower_mixtures (data.gouv.fr 644b79f82ef3605eb5c660aa)
    # ------------------------------------------------------------------
    # Picouet (2023), "Spectres Raman de mélanges d'Huiles d'Olive et
    # de Tournesol / Raman Spectra of Olive and Sunflower oil
    # mixtures" (data.gouv.fr / ENTREPOT Recherche,
    # doi:10.57745/DOGT0E, licence ouverte lov2).
    #
    # 605 Raman spectra of EVOO/sunflower oil mixtures at 26 EVOO
    # levels (0-100 %), Ocean Insight QE Pro, 785 nm / 400 mW,
    # 1000 ms, 10 scans, 5 spectra per blend (design of experiment;
    # per the deposit README). The deposit CSV is ';'-separated
    # (719 fields per row: 9 metadata + 710 intensities) and the
    # wavenumber header cells use European decimal commas
    # ("500,48" -> 500.48 cm^-1), so the header row splits into 711
    # comma-fields while each data row has 1 comma-field plus 710
    # trailing empties (an Excel export quirk) - parse with
    # split(';'), never split(','). Served directly from the
    # static.data.gouv.fr CDN (no WAF, CI-reachable); sha256 pinned.
    _OLIVE_SUNFLOWER_CSV = (
        "https://static.data.gouv.fr/resources/spectres-raman-de-"
        "melanges-dhuiles-dolive-et-de-tournesol-raman-spectra-of-"
        "olive-and-sunflower-oil-mixtures/20230428-094746/"
        "raman-oils-raw.csv",
        "raman-oils-raw.csv",
        2173967,
        "c426967faac3005b1145385d5f8da5bfa85078084f7c5931b30a9bf9fcb11758",
    )

    @staticmethod
    def _load_olive_sunflower_mixtures(cache_path: str):
        """
        Download (if needed) and load the olive/sunflower mixture
        Raman dataset: 605 spectra of EVOO/sunflower blends at 26
        EVOO levels (0-100 %), with the EVOO percentage as the
        single regression target.

        Returns spectra, raman_shifts, targets, target_names,
        group_ids (group = (date, EVOO %) physical blend).
        """
        os.makedirs(cache_path, exist_ok=True)
        url, fname, size, sha = MiscLoader._OLIVE_SUNFLOWER_CSV
        fpath = os.path.join(cache_path, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) != size:
            os.remove(fpath)
        result = LoaderTools.download(
            url=url,
            out_dir_path=cache_path,
            out_file_name=fname,
            hash_target=sha,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                "[!] olive_sunflower_mixtures: CSV download failed "
                "(network error or checksum mismatch). See logs."
            )

        with open(fpath, encoding="utf-8-sig") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        if len(lines) != 606:
            raise RuntimeError(
                f"[!] olive_sunflower_mixtures: expected 605 data "
                f"rows (+ header), found {len(lines) - 1} (deposit "
                "changed?)."
            )
        header = lines[0].split(";")
        if len(header) != 719:
            raise RuntimeError(
                f"[!] olive_sunflower_mixtures: expected 719 "
                f"semicolon header fields (9 meta + 710 wavenumbers), "
                f"got {len(header)}."
            )
        # The wavenumber header cells use European decimal commas.
        axis = np.array([float(h.replace(",", ".")) for h in header[9:]])
        if (
            not np.all(np.diff(axis) > 0)
            or not (500.0 < axis[0] < 510.0)
            or not (2400.0 < axis[-1] < 2500.0)
        ):
            raise RuntimeError(
                "[!] olive_sunflower_mixtures: unexpected wavenumber "
                f"axis ({axis.shape[0]} points, "
                f"{axis[0]:.3f}..{axis[-1]:.3f} cm^-1)."
            )
        rows = []
        for i, ln in enumerate(lines[1:], start=2):
            fields = ln.split(";")
            if len(fields) != 719:
                raise RuntimeError(
                    f"[!] olive_sunflower_mixtures: row {i} has "
                    f"{len(fields)} semicolon fields (expected 719)."
                )
            rows.append(fields)
        meta = np.array([f[:9] for f in rows], dtype=str)
        spectra = np.array([f[9:] for f in rows], dtype=float)
        if np.isnan(spectra).any():
            raise RuntimeError(
                "[!] olive_sunflower_mixtures: NaN intensity values."
            )

        def _pct(col: int) -> np.ndarray:
            vals = meta[:, col].astype(str)
            if not np.all(np.char.endswith(vals, "%")):
                raise RuntimeError(
                    f"[!] olive_sunflower_mixtures: column {col} "
                    "cells are not percentage values."
                )
            out = np.array([float(v.rstrip("%")) for v in vals])
            if np.isnan(out).any():
                raise RuntimeError(
                    f"[!] olive_sunflower_mixtures: unparseable "
                    f"percentage in column {col}."
                )
            return out

        targets = _pct(3)  # EVOO (%)
        sunflower = _pct(4)  # Sunflower (%)
        if not (targets.min() >= 0.0 and targets.max() <= 100.0):
            raise RuntimeError(
                f"[!] olive_sunflower_mixtures: EVOO % out of range "
                f"({targets.min():g}..{targets.max():g})."
            )
        if not np.allclose(targets + sunflower, 100.0):
            raise RuntimeError(
                "[!] olive_sunflower_mixtures: EVOO % + sunflower % "
                "does not sum to 100."
            )
        # Acquisition parameters are constant across the deposit
        # (per the README: 1000 ms, 10 scans, 400 mW).
        if (
            set(meta[:, 5].tolist()) != {"1000"}
            or set(meta[:, 6].tolist()) != {"10"}
            or set(meta[:, 7].tolist()) != {"400"}
        ):
            raise RuntimeError(
                "[!] olive_sunflower_mixtures: Tps/Scans/P are not "
                "the documented constant 1000 ms / 10 / 400 mW."
            )
        # Group = (date, EVOO %): one physical blend measured with
        # 5 scans (S1-S5); a few blends were re-measured the same
        # day, so groups hold 5, 10 or 15 spectra.
        group_ids = np.array(
            [f"{d}_{v}" for d, v in zip(meta[:, 0], meta[:, 3])],
            dtype=str,
        )
        n_groups = len(set(group_ids.tolist()))
        if n_groups != 109:
            raise RuntimeError(
                f"[!] olive_sunflower_mixtures: expected 109 blend "
                f"groups, found {n_groups}."
            )
        MiscLoader.logger.debug(
            f"Loaded olive_sunflower_mixtures: {spectra.shape[0]} "
            f"spectra x {spectra.shape[1]} points, EVOO % "
            f"{targets.min():g}..{targets.max():g}, {n_groups} blends"
        )
        return spectra, axis, targets, ["evoo_percent"], group_ids

    # ------------------------------------------------------------------
    # infra_art_raman (INFRA-ART Spectral Library, infra-art.eu)
    # ------------------------------------------------------------------
    # Cortea, ChiroscA, Angheluta, Seritan (2023), "INFRA-ART: An
    # Open Access Spectral Library of Art-related Materials as a
    # Digital Support Tool for Cultural Heritage Science", Journal
    # on Computing and Cultural Heritage 16(2), 1-11,
    # doi:10.1145/3593427 (FAIRsharing
    # doi:10.25504/FAIRsharing.091c24). National Institute for
    # Research and Development in Optoelectronics (INOE 2000),
    # Bucharest; CC BY-NC 4.0.
    #
    # The library is a live Next.js/Strapi web app behind
    # Cloudflare; the Raman spectra of its 186 reference samples
    # (one per sample) are Renishaw ENLIGHTEN CSV exports served
    # from
    #   /api/samples/spectra-file?sampleId=<string id>&dataType=Raman%20Spectrum
    # (requires a full browser request header set). The live
    # endpoint is not reachable from CI and the library is still
    # growing, so this key reads a frozen 2026-10-21 snapshot zip
    # (186 CSVs + manifest.json + README.txt, size + sha256
    # pinned) mirrored at HTW-KI-Werkstatt/RamanBench (infra_art/).
    #
    # Each CSV: a Key,Value metadata block, then a header row
    # (Pixel,Wavelength,Wavenumber,Processed[,Raw,Dark,Reference]
    # - both variants occur: 7 columns in 122 files, 4 columns in
    # 64) and one row per detector pixel. Wavenumber column in
    # cm^-1 (1666 points, 160.07-3215.39, common to all files,
    # non-uniform spacing); Processed = dark-subtracted intensity
    # in arbitrary units (slightly negative values possible).
    _INFRA_ART_SNAPSHOT = (
        "https://huggingface.co/datasets/HTW-KI-Werkstatt/RamanBench/"
        "resolve/main/infra_art/infra_art_raman_snapshot_2026-10-21.zip",
        "infra_art_raman_snapshot_2026-10-21.zip",
        4930163,
        "506040528ef866bf2cb5ac543599ebabc3e56ea6c9c84d140dedba147d0b34bb",
    )

    @staticmethod
    def _load_infra_art_raman(cache_path: str):
        """
        Download (if needed) and load the frozen INFRA-ART Raman
        snapshot: 186 reference spectra of art-related materials
        (one per sample) on the native 1666-point 160.07-3,215.39
        cm^-1 axis, with the library's material class as the
        classification label.

        Returns spectra, raman_shifts, targets, class_names.
        """
        import hashlib
        import json
        import shutil
        import zipfile

        os.makedirs(cache_path, exist_ok=True)
        url, fname, size, sha = MiscLoader._INFRA_ART_SNAPSHOT
        zip_path = os.path.join(cache_path, fname)
        if os.path.exists(zip_path) and os.path.getsize(zip_path) != size:
            os.remove(zip_path)
        result = LoaderTools.download(
            url=url,
            out_dir_path=cache_path,
            out_file_name=fname,
            hash_target=sha,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                "[!] infra_art_raman: snapshot zip download failed "
                "(network error or checksum mismatch). See logs."
            )

        extract_dir = os.path.join(cache_path, "infra_art_raman_snapshot")
        spectra_dir = os.path.join(extract_dir, "spectra")
        if not (
            os.path.isdir(spectra_dir)
            and len(os.listdir(spectra_dir)) == 186
            and os.path.isfile(os.path.join(extract_dir, "manifest.json"))
        ):
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        with open(
            os.path.join(extract_dir, "manifest.json"), encoding="utf-8"
        ) as fh:
            manifest = json.load(fh)
        records = manifest.get("samples", [])
        if manifest.get("n_samples") != 186 or len(records) != 186:
            raise RuntimeError(
                f"[!] infra_art_raman: expected 186 samples in the "
                f"snapshot manifest, found {len(records)} (snapshot "
                "changed?)."
            )

        axis = None
        spectra_list = []
        labels = []
        for rec in records:
            fpath = os.path.join(extract_dir, "spectra", rec["csv"])
            if not os.path.isfile(fpath):
                raise RuntimeError(
                    f"[!] infra_art_raman: {rec['csv']} is missing "
                    "from the snapshot."
                )
            data = open(fpath, "rb").read()
            if hashlib.sha256(data).hexdigest() != rec["csv_sha256"]:
                raise RuntimeError(
                    f"[!] infra_art_raman: checksum mismatch for "
                    f"{rec['csv']} (snapshot corrupted or altered)."
                )
            lines = data.decode("utf-8", errors="replace").splitlines()
            hi = None
            for i, ln in enumerate(lines):
                if ln.startswith("Pixel,"):
                    hi = i
                    break
            if hi is None:
                raise RuntimeError(
                    f"[!] infra_art_raman: {rec['csv']} has no data "
                    "header row."
                )
            header = [c.strip() for c in lines[hi].split(",")]
            if not all(n in header for n in ("Wavenumber", "Processed")):
                raise RuntimeError(
                    f"[!] infra_art_raman: {rec['csv']} lacks the "
                    "Wavenumber/Processed columns."
                )
            ix = header.index("Wavenumber")
            ip = header.index("Processed")
            rows = []
            for ln in lines[hi + 1:]:
                if not ln.strip():
                    continue
                p = ln.split(",")
                if len(p) <= max(ix, ip):
                    raise RuntimeError(
                        f"[!] infra_art_raman: short row in "
                        f"{rec['csv']}: {ln[:60]!r}."
                    )
                rows.append((float(p[ix]), float(p[ip])))
            wn = np.array([r[0] for r in rows], dtype=float)
            sp = np.array([r[1] for r in rows], dtype=float)
            if (
                wn.shape[0] != 1666
                or not np.all(np.diff(wn) > 0)
                or not (155.0 < wn[0] < 165.0)
                or not (3200.0 < wn[-1] < 3220.0)
            ):
                raise RuntimeError(
                    f"[!] infra_art_raman: {rec['csv']} has an "
                    f"unexpected axis ({wn.shape[0]} points, "
                    f"{wn[0]:.2f}..{wn[-1]:.2f} cm^-1)."
                )
            if np.isnan(sp).any():
                raise RuntimeError(
                    f"[!] infra_art_raman: {rec['csv']} has "
                    "non-numeric intensity values."
                )
            if axis is None:
                axis = wn
            elif not np.array_equal(axis, wn):
                raise RuntimeError(
                    f"[!] infra_art_raman: {rec['csv']} deviates "
                    "from the common 1666-point axis."
                )
            spectra_list.append(sp)
            labels.append(rec["materialClassName"])

        spectra = np.array(spectra_list, dtype=float)
        encoded_targets, class_names = encode_labels(pd.Series(labels))
        MiscLoader.logger.debug(
            f"Loaded infra_art_raman: {spectra.shape[0]} spectra x "
            f"{spectra.shape[1]} points, {len(set(class_names))} "
            "material classes"
        )
        return (
            spectra,
            axis,
            encoded_targets,
            list(class_names),
        )

    # ------------------------------------------------------------------
    # ocean_polymers / ocean_polymers_raw
    # (Miller et al. 2022, Scientific Data 9:780, OSF osf.io/7cqv4)
    # ------------------------------------------------------------------
    # "A Raman spectral reference library of potential
    # anthropogenic and biological ocean polymers" (Miller,
    # Yamahara, French, Spingarn, Birch & Van Houtan 2022,
    # Scientific Data 9, 780, doi:10.1038/s41597-022-01883-5,
    # open access CC BY 4.0). The OSF project itself sets no
    # explicit license; the open-access parent paper publishes
    # the data records (Data Records section) under CC BY 4.0,
    # which is the license used here.
    #
    # The public OSF node 7cqv4 (no child components) holds
    # exactly two files:
    #   Data2_processed.csv (19,921,069 B) - the
    #     post-processed reference library in long format, one
    #     row per (spectrum, wavenumber); columns ID, SAMPLE,
    #     INTENSITY, WAVE, PARTICLE, poly_lab. 79 spectra (61
    #     PLAS-coded + 18 BIOL-coded particles, one per
    #     specimen), each a uniform 3,201-point 200-3,400 cm^-1
    #     @1 grid, peak-normalized 0-1. poly_lab = 59 labels
    #     (12 multi-member, 47 singletons).
    #   Data1_raw_csv.zip (1,452,113 B) - 87 raw Omnic
    #     measurement files of 49 specimen codes: 74 x 532 nm
    #     (3,467 pts, 76.55-3,418.58 cm^-1) + 13 x 780 nm
    #     (2,700 pts, 200.49-3,699.66 cm^-1); raw counts, axis
    #     stored descending. 28 specimen codes have a repeated
    #     532 nm measurement, 5 have a repeated 532 nm
    #     measurement plus one 780 nm measurement.
    #
    # OSF serves the files from stable URLs that 302-redirect
    # to a short-lived signed object-storage URL; no browser
    # headers are required (CI-reachable), so both keys
    # download directly with size + sha256 pinning - no mirror.
    _MILLER_DATA2_URL = (
        "https://files.osf.io/v1/resources/7cqv4/providers/"
        "osfstorage/61cffbeace4dbc0c5dc73a40"
    )
    _MILLER_DATA2 = "Data2_processed.csv"
    _MILLER_DATA2_SIZE = 19921069
    _MILLER_DATA2_SHA256 = (
        "4d063b8bf95d87ba3936aa65c0dbc2968ffba4e27e0debb7dae17a16840af6d0"
    )
    _MILLER_DATA1_URL = (
        "https://files.osf.io/v1/resources/7cqv4/providers/"
        "osfstorage/61cfff382962ce0c76b00abc"
    )
    _MILLER_DATA1 = "Data1_raw_csv.zip"
    _MILLER_DATA1_SIZE = 1452113
    _MILLER_DATA1_SHA256 = (
        "033cc6fa9243c72591ab67f8cf2204d982e04e8ae920b5a2e343e45e0308c54d"
    )
    _MILLER_DATA1_ROOT = "Data1_raw_csv"
    _MILLER_DATA1_N_FILES = 87
    # 33 of the 87 files are byte-identical copies of another file
    # under an instrument-run prefix ('24544r56_1 PLAS 214 532 nm.csv'
    # = 'PLAS 214 532 nm.csv'), so 54 distinct measurements remain.
    _MILLER_DATA1_N_UNIQUE = 54
    _MILLER_DATA2_N_SPECTRA = 79
    _MILLER_DATA2_N_POINTS = 3201
    _MILLER_DATA2_N_CLASSES = 59

    @staticmethod
    def _download_miller_osf_file(
        cache_path: str,
        url: str,
        fname: str,
        size: int,
        sha: str,
        key: str,
    ) -> str:
        """
        Download (if needed) one pinned OSF file of the Miller
        2022 ocean-polymer deposit; returns the local file path.
        """
        os.makedirs(cache_path, exist_ok=True)
        fpath = os.path.join(cache_path, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) != size:
            os.remove(fpath)
        result = LoaderTools.download(
            url=url,
            out_dir_path=cache_path,
            out_file_name=fname,
            hash_target=sha,
            hash_type=HASH_TYPE.sha256,
        )
        if result is None:
            raise RuntimeError(
                f"[!] {key}: OSF download of {fname} failed "
                "(network error or checksum mismatch). See logs."
            )
        return fpath

    @staticmethod
    def _load_miller_ocean_polymers(cache_path: str):
        """
        Load the 79 post-processed reference spectra (poly_lab
        classification, 59 classes) on the authors' uniform
        3,201-point 200-3,400 cm^-1 @1 cm^-1 grid.

        Returns spectra, raman_shifts, targets, class_names.
        """
        fpath = MiscLoader._download_miller_osf_file(
            cache_path,
            MiscLoader._MILLER_DATA2_URL,
            MiscLoader._MILLER_DATA2,
            MiscLoader._MILLER_DATA2_SIZE,
            MiscLoader._MILLER_DATA2_SHA256,
            "ocean_polymers",
        )
        df = pd.read_csv(fpath)
        expected_cols = {
            "ID", "SAMPLE", "INTENSITY", "WAVE", "PARTICLE",
            "poly_lab",
        }
        if not expected_cols.issubset(df.columns):
            raise RuntimeError(
                "[!] ocean_polymers: Data2_processed.csv columns "
                f"changed ({list(df.columns)})."
            )
        if df[["INTENSITY", "WAVE"]].isna().any().any():
            raise RuntimeError(
                "[!] ocean_polymers: Data2_processed.csv "
                "contains missing values."
            )
        # One row per (spectrum, wavenumber) point; group by ID,
        # preserving first appearance (deterministic; the file
        # is sha256-pinned).
        groups = df.groupby("ID", sort=False)
        if len(groups) != MiscLoader._MILLER_DATA2_N_SPECTRA:
            raise RuntimeError(
                "[!] ocean_polymers: expected "
                f"{MiscLoader._MILLER_DATA2_N_SPECTRA} spectra, "
                f"found {len(groups)}."
            )
        spectra_list = []
        labels = []
        axis = None
        for sid, block in groups:
            if (
                block["PARTICLE"].nunique() != 1
                or block["poly_lab"].nunique() != 1
            ):
                raise RuntimeError(
                    f"[!] ocean_polymers: ID {sid!r} maps to more "
                    "than one particle or label."
                )
            wave = block["WAVE"].to_numpy(dtype=float)
            inten = block["INTENSITY"].to_numpy(dtype=float)
            if axis is None:
                axis = wave
                if (
                    len(wave)
                    != MiscLoader._MILLER_DATA2_N_POINTS
                    or wave[0] != 200.0
                    or wave[-1] != 3400.0
                    or not np.all(np.diff(wave) > 0)
                ):
                    raise RuntimeError(
                        "[!] ocean_polymers: the common "
                        "3,201-point 200-3,400 cm^-1 axis is no "
                        "longer present."
                    )
            elif not np.array_equal(wave, axis):
                raise RuntimeError(
                    f"[!] ocean_polymers: ID {sid!r} carries a "
                    "different wavenumber axis."
                )
            if not np.isfinite(inten).all():
                raise RuntimeError(
                    f"[!] ocean_polymers: ID {sid!r} has "
                    "non-finite intensities."
                )
            spectra_list.append(inten)
            labels.append(block["poly_lab"].iat[0])
        spectra = np.asarray(spectra_list, dtype=float)
        encoded_targets, class_names = encode_labels(
            pd.Series(labels)
        )
        if len(set(class_names)) != MiscLoader._MILLER_DATA2_N_CLASSES:
            raise RuntimeError(
                f"[!] ocean_polymers: expected "
                f"{MiscLoader._MILLER_DATA2_N_CLASSES} poly_lab "
                f"classes, found {len(set(class_names))}."
            )
        MiscLoader.logger.debug(
            f"Loaded ocean_polymers: {spectra.shape[0]} spectra "
            f"x {spectra.shape[1]} points, "
            f"{len(set(class_names))} poly_lab classes"
        )
        return (
            spectra,
            axis,
            encoded_targets,
            list(class_names),
        )

    @staticmethod
    def _load_miller_ocean_polymers_raw(cache_path: str):
        """
        Load the 54 distinct raw Omnic measurement spectra (the
        zip's 87 files minus 33 byte-identical copies) of the 49
        specimen codes (anthropogenic vs biological
        classification, group_id = specimen code), each
        resampled onto the common 201-3,418 cm^-1 @1 cm^-1
        grid (3,218 points).

        Returns spectra, raman_shifts, targets, class_names,
        group_ids.
        """
        import hashlib
        import zipfile

        zpath = MiscLoader._download_miller_osf_file(
            cache_path,
            MiscLoader._MILLER_DATA1_URL,
            MiscLoader._MILLER_DATA1,
            MiscLoader._MILLER_DATA1_SIZE,
            MiscLoader._MILLER_DATA1_SHA256,
            "ocean_polymers_raw",
        )
        grid = np.arange(201.0, 3419.0)  # 201..3418 @1 = 3,218
        with zipfile.ZipFile(zpath) as zf:
            names = sorted(
                n
                for n in zf.namelist()
                if n.startswith(MiscLoader._MILLER_DATA1_ROOT + "/")
                and n.endswith(".csv")
            )
            if len(names) != MiscLoader._MILLER_DATA1_N_FILES:
                raise RuntimeError(
                    "[!] ocean_polymers_raw: expected "
                    f"{MiscLoader._MILLER_DATA1_N_FILES} CSVs in "
                    f"{MiscLoader._MILLER_DATA1_ROOT}/, found "
                    f"{len(names)}."
                )
            # Drop byte-identical copies, keeping the shortest name
            # (the one without the instrument-run prefix).
            by_hash = {}
            for name in sorted(names, key=lambda n: (len(n), n)):
                by_hash.setdefault(
                    hashlib.sha256(zf.read(name)).hexdigest(), name
                )
            names = sorted(by_hash.values())
            if len(names) != MiscLoader._MILLER_DATA1_N_UNIQUE:
                raise RuntimeError(
                    "[!] ocean_polymers_raw: expected "
                    f"{MiscLoader._MILLER_DATA1_N_UNIQUE} distinct "
                    f"CSVs, found {len(names)}."
                )
            entries = []
            for name in names:
                fname = name.rsplit("/", 1)[-1]
                m = re.search(r"((?:PLAS|BIOL) \d+) (\d+) nm", fname)
                if not m:
                    raise RuntimeError(
                        "[!] ocean_polymers_raw: cannot parse a "
                        f"specimen code from {fname!r}."
                    )
                code, ex = m.group(1), int(m.group(2))
                if ex not in (532, 780):
                    raise RuntimeError(
                        "[!] ocean_polymers_raw: unexpected "
                        f"excitation wavelength {ex} in {fname!r}."
                    )
                text = zf.read(name).decode("utf-8-sig", "replace")
                # deterministic row order: specimen code,
                # excitation, filename
                entries.append(((code, ex, fname), code, ex, text))
        entries.sort(key=lambda e: e[0])

        spectra_list = []
        labels = []
        group_ids = []
        for _, code, ex, text in entries:
            # line 1: 'Created As Omnic File,<CODE>'
            # line 2: 'Raman Shift / cm-1,INT'
            pts = []
            for ln in text.splitlines()[2:]:
                if not ln.strip():
                    continue
                a, b = ln.split(",", 1)
                pts.append((float(a), float(b)))
            arr = np.asarray(pts, dtype=float)
            n_native = 3467 if ex == 532 else 2700
            if arr.shape[0] != n_native:
                raise RuntimeError(
                    f"[!] ocean_polymers_raw: {code} ({ex} nm) "
                    f"has {arr.shape[0]} points (expected "
                    f"{n_native})."
                )
            x, y = arr[:, 0], arr[:, 1]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise RuntimeError(
                    f"[!] ocean_polymers_raw: {code} ({ex} nm) "
                    "contains non-finite values."
                )
            if not np.all(np.diff(x) < 0):
                raise RuntimeError(
                    f"[!] ocean_polymers_raw: {code} ({ex} nm) "
                    "wavenumber axis is not strictly descending "
                    "as expected."
                )
            # the common grid must lie inside the native range
            if x[0] < grid[-1] or x[-1] > grid[0]:
                raise RuntimeError(
                    f"[!] ocean_polymers_raw: {code} ({ex} nm) "
                    f"does not cover the common grid (native "
                    f"{x[-1]:.2f}..{x[0]:.2f} cm^-1 vs "
                    f"{grid[0]:.0f}..{grid[-1]:.0f} expected)."
                )
            # flip the descending axis to ascending, then
            # interpolate onto the common grid
            spectra_list.append(np.interp(grid, x[::-1], y[::-1]))
            labels.append(
                "anthropogenic" if code.startswith("PLAS")
                else "biological"
            )
            group_ids.append(code)
        spectra = np.asarray(spectra_list, dtype=float)
        encoded_targets, class_names = encode_labels(
            pd.Series(labels)
        )
        census = Counter(labels)
        if (
            spectra.shape != (MiscLoader._MILLER_DATA1_N_UNIQUE, grid.size)
            or census != {"anthropogenic": 41, "biological": 13}
            or len(set(group_ids)) != 49
        ):
            raise RuntimeError(
                f"[!] ocean_polymers_raw: unexpected census "
                f"(shape {spectra.shape}, labels "
                f"{dict(census)}, {len(set(group_ids))} groups)."
            )
        MiscLoader.logger.debug(
            f"Loaded ocean_polymers_raw: {spectra.shape[0]} "
            f"spectra x {spectra.shape[1]} points, "
            f"{len(set(group_ids))} specimen groups"
        )
        return (
            spectra,
            grid,
            encoded_targets,
            list(class_names),
            group_ids,
        )
