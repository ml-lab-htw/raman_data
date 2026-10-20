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
                "description": "Baseline-corrected Raman spectra of 14 plastic types (polyethylene, polypropylene, PTFE, polystyrene, nylon, silicone, PMMA, POM, polyester, polyurethane, PVC, PET, polycarbonate, ABS) acquired on a Renishaw inVia Raman microscope: 226 spectra at 633 nm (1800 l/mm grating) and 176 at 523 nm (2400 l/mm grating). The deposited RaSPI_processed.csv is uniformly gridded at 1 cm^-1 over 225-3949 cm^-1 (3725 points), polynomial-baseline-subtracted and 0-1 normalized. Samples mix pristine commercial/retail plastics with environmental microplastics (bottle caps, bags, fragments) from Pollution Probe Great Lakes Seabin collections. The deposit's spectra file holds 402 rows, but 6 metadata rows lack a spectrum, 4 spectra lack metadata, and PE1a/PE1b are stored twice, leaving a fully-labeled usable set of 396 spectra from 272 physical samples (159 measured once, 104 twice, 9 three to four times). Each spectrum ID is '<plastic><number>' plus a trailing letter (PE1a, PE1b, PET4c...) marking repeated measurements of the same physical sample; the loader exposes those sample ids as group_ids. The raw (unprocessed) spectra are available in the same deposit as RaSPI_raw.csv. Same lab and much of the authorship as Lei et al. 2022 (the microplastics_training_* keys; Smith group, U. Waterloo -- Hogan and Lei co-authored both papers): the commercial-pristine subsets may draw on the same lab collection, though the deposits carry no shared sample ids.",
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
                "subtraction, truncation to 225-3950 cm^-1, interpolation "
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
                "description": "Per-pixel Raman spectra from 34 maps (30 pristine two-material blend maps plus 4 environmental maps of Great Lakes Seabin microplastic samples) acquired on a Renishaw inVia at 633 nm (1800 l/mm grating). Each of the 33,119 pixel spectra was manually assigned by the authors to one of 14 plastic types or 'blank'; this dataset holds the 16,598 plastic-assigned pixels. The native per-map axes (1,017 points, ~710-1,827 cm^-1, non-uniform spacing) are linearly resampled onto a common 1 cm^-1 grid (710-1,827 cm^-1, 1,118 points). Spectra are raw as exported from Renishaw WiRE -- no baseline correction or normalization. Maps per class: polyethylene 8, polyoxymethylene 7, ABS/PMMA/PS/PTFE 6, nylon/PP/PVC 5, polyester/PET 4, polycarbonate 3, polyurethane 2, silicone 2 -- the last two fall below 3 maps and cannot be split into grouped 3-fold CV by map. group_ids = map: pixels within one map are spatially correlated and must not be split across train/test. The companion 1D point-spectra dataset is raman_spi.",
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

        Files (Borealis / University of Waterloo DataVerse, CC-BY 4.0):
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
        plastic_types = merged["Plastic Type"]
        group_ids = merged["Spectra ID"].map(MiscLoader._raspi_sample_stem).to_numpy()

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
        doi:10.5683/SP3/LSN0R0 (Borealis, CC-BY 4.0).

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

        spectra = df[wav_cols].to_numpy(dtype=float)
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
