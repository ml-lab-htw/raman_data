# raman-data

[![PyPI version](https://badge.fury.io/py/raman-data.svg)](https://pypi.org/project/raman-data/)
[![License](https://img.shields.io/github/license/ml-lab-htw/raman_data)](https://github.com/ml-lab-htw/raman_data/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/raman-data)](https://pypi.org/project/raman-data/)

A unified Python library for accessing public Raman spectroscopy datasets.
`raman-data` is the dataset layer of [RamanBench](https://github.com/ml-lab-htw/RamanBench), a large-scale benchmark for machine learning on Raman spectroscopy data.
It provides a single API to discover, download, and load <!-- DATASET_COUNT_START -->**90 datasets**<!-- DATASET_COUNT_END --> — covering classification, regression, denoising, and super-resolution tasks — from diverse sources (Kaggle, HuggingFace, Zenodo, Figshare, GitHub) in a standardized, ML-ready format.
Of these, 75 meet the inclusion criteria of [RamanBench](https://github.com/ml-lab-htw/RamanBench) and are used for benchmarking.

## Installation

```bash
pip install raman-data
```

> **Kaggle datasets** require API credentials. Follow the [Kaggle API setup guide](https://www.kaggle.com/docs/api) and place your `kaggle.json` in `~/.kaggle/`.

## Quick Start

```python
from raman_data import raman_data, TASK_TYPE, APPLICATION_TYPE

# List all available datasets
raman_data()

# Filter by task type or application domain
raman_data(task_type=TASK_TYPE.Classification)
raman_data(application_type=APPLICATION_TYPE.Medical)
raman_data(task_type=TASK_TYPE.Regression, application_type=APPLICATION_TYPE.Biological)

# Load a dataset by ID
dataset = raman_data("bioprocess_substrates")

# Access data
X = dataset.spectra          # np.ndarray (n_samples × n_wavenumbers)
w = dataset.raman_shifts     # np.ndarray of wavenumber values in cm⁻¹
y = dataset.targets          # np.ndarray of labels or values
print(dataset.metadata)

# Convert to pandas DataFrame
df = dataset.to_dataframe()
```

See [`examples/demo.ipynb`](examples/demo.ipynb) for a full walkthrough.

## RamanDataset API

Every `raman_data(dataset_id)` call returns a `RamanDataset` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `spectra` | `np.ndarray` | Intensity matrix (samples × wavenumbers) |
| `raman_shifts` | `np.ndarray` | Wavenumber axis in cm⁻¹ |
| `targets` | `np.ndarray` | Labels (classification) or values (regression) |
| `target_names` | `list` | Target column names (regression) or class names (classification) |
| `metadata` | `dict` | Source, paper reference, description |
| `name` | `str` | Dataset identifier |
| `task_type` | `TASK_TYPE` | `Classification`, `Regression`, `Denoising`, or `SuperResolution` |
| `application_type` | `APPLICATION_TYPE` | `Medical`, `Biological`, `Chemical`, or `MaterialScience` |
| `n_spectra` | `int` | Number of spectra |
| `n_frequencies` | `int` | Number of wavenumber points |
| `n_classes` | `int \| None` | Number of classes (classification only) |
| `target_range` | `tuple \| None` | (min, max) of target values (regression only) |
| `min_shift` / `max_shift` | `float` | Spectral range in cm⁻¹ |

**Variable-length spectra:** If spectra share the same wavenumber axis, `spectra` and `raman_shifts` are standard 2D/1D arrays.
If the axis differs per sample, both are returned as object arrays of 1D arrays.
For machine learning, interpolate or pad to a common grid as needed.

## Available Datasets

<details>
<summary>89 datasets across Material Science, Biological & Biotechnological, Medical & Clinical, and Chemical & Industrial domains (click to expand)</summary>

<!-- DATASETS_TABLE_START -->
<!-- AUTO-GENERATED: START - datasets table. Do not edit manually. -->

| Dataset Name | Application | Task Type | Description |
|--------------|-------------|-----------|-------------|
| `acetic_acid_species` | Chemical & Industrial | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `adenine_colloidal_gold` | Chemical & Industrial | Regression | Quantitative SERS spectra of adenine measured using colloidal gold substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_colloidal_silver` | Chemical & Industrial | Regression | Quantitative SERS spectra of adenine measured using colloidal silver substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_solid_gold` | Chemical & Industrial | Regression | Quantitative SERS spectra of adenine measured using solid gold substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_solid_silver` | Chemical & Industrial | Regression | Quantitative SERS spectra of adenine measured using solid silver substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `alzheimer` | Medical & Clinical | Classification | Raman spectra from dried saliva drops targeting Alzheimer's Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment. |
| `amino_acids_glycine` | Chemical & Industrial | Regression | Time-resolved LC-Raman spectra for Glycine elution using the vertical flow method (Lo, Hiramatsu lab, NCTU Taiwan). Glycine injected at 100 mM into an HPLC system (hydrophobic resin column, H₂O→ACE gradient, 7 mL/min, 50 µL injection). Time-resolved acquisition at 0.2 s/frame. Preprocessing: solvent |
| `amino_acids_leucine` | Chemical & Industrial | Regression | Time-resolved LC-Raman spectra for Leucine elution using the vertical flow method (Lo, Hiramatsu lab, NCTU Taiwan). Leucine injected at 100 mM into an HPLC system (hydrophobic resin column, H₂O→ACE gradient, 7 mL/min, 50 µL injection). Time-resolved acquisition at 0.2 s/frame. Preprocessing: solvent |
| `amino_acids_phenylalanine` | Chemical & Industrial | Regression | Time-resolved LC-Raman spectra for Phenylalanine elution using the vertical flow method (Lo, Hiramatsu lab, NCTU Taiwan). Phenylalanine injected at 55 mM into an HPLC system (hydrophobic resin column, H₂O→ACE gradient, 7 mL/min, 50 µL injection). Time-resolved acquisition at 0.2 s/frame. Preprocessi |
| `amino_acids_tryptophan` | Chemical & Industrial | Regression | Time-resolved LC-Raman spectra for Tryptophan elution using the vertical flow method (Lo, Hiramatsu lab, NCTU Taiwan). Tryptophan injected at 55 mM into an HPLC system (hydrophobic resin column, H₂O→ACE gradient, 7 mL/min, 50 µL injection). Time-resolved acquisition at 0.2 s/frame. Preprocessing: so |
| `bacteria_identification` | Medical & Clinical | Classification | 60,000 SERS spectra (2,000 per isolate across three measurement time-points) from 30 clinically relevant bacterial and yeast isolates, including an MRSA/MSSA isogenic pair. Acquired on a Horiba LabRAM HR Evolution spectrometer (633 nm, 13.17 mW, 300 l/mm grating, 1.2 cm⁻¹ dispersion, Olympus MPLAN 1 |
| `biomolecules_reference` | Biological & Biotechnological | Classification | Reference Raman spectra (450–1800 cm⁻¹, 1 cm⁻¹ resolution) of ~140 pure biomolecules including amino acids, nucleotides, lipids, and sugars. Each spectrum is labelled by biomolecule name. Useful for spectral assignment and as a reference library for classification benchmarks. |
| `bioprocess_analytes_anton_532` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_anton_785` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_kaiser` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_metrohm` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_mettler_toledo` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_tec5` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_timegate` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_analytes_tornado` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. It is part of a series of 8 datasets that use eight different spectrometers that measure nearly the same samples. Some datasets have a bit more samples than others. Each spectrum is paired with ground |
| `bioprocess_substrates` | Biological & Biotechnological | Regression | A benchmark dataset of 6,960 spectra featuring eight key metabolites (glucose, glycerol, acetate, etc.) sampled via a statistically independent uniform distribution. Designed to evaluate regression robustness against common bioprocess correlations, including background effects from mineral salts and |
| `cancer_cell_(cooh)2` | Biological & Biotechnological | Classification | SERS spectra of conditioned cell culture media metabolites entrapped on gold multibranched nanoparticles (AuMs, ~50 nm) functionalized with (COOH)2. 12 sample categories from Table 1 (Erzina et al. 2020): A2058/G361 melanoma cells, HPM melanocytes, HF skin fibroblasts, ZAM tumour-associated fibrobla |
| `cancer_cell_cooh` | Biological & Biotechnological | Classification | SERS spectra of conditioned cell culture media metabolites entrapped on gold multibranched nanoparticles (AuMs, ~50 nm) functionalized with COOH. 12 sample categories from Table 1 (Erzina et al. 2020): A2058/G361 melanoma cells, HPM melanocytes, HF skin fibroblasts, ZAM tumour-associated fibroblasts |
| `cancer_cell_nh2` | Biological & Biotechnological | Classification | SERS spectra of conditioned cell culture media metabolites entrapped on gold multibranched nanoparticles (AuMs, ~50 nm) functionalized with NH2. 12 sample categories from Table 1 (Erzina et al. 2020): A2058/G361 melanoma cells, HPM melanocytes, HF skin fibroblasts, ZAM tumour-associated fibroblasts, |
| `chembl_molecules` | Chemical & Industrial | Regression | 140k DFT-computed Raman spectra for ChEMBL drug-like molecules. Targets: HOMO-LUMO gap, HOMO/LUMO energies, isotropic polarizability, heat capacity, dipole moment. |
| `chlorinated_samples` | Chemical & Industrial | Classification | Binary Raman classification task: detect the presence of chloroform in a sample. 230 spectra across 2473 wavenumbers (350–3500 cm⁻¹). Class balance {0: 76, 1: 154}. Data provided by Analyze IQ Limited; predefined 3-fold splits ship with the source repository. |
| `citric_acid_species` | Chemical & Industrial | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `comfile_stroke` | Medical & Clinical | Classification | SERS serum spectra for binary stroke vs. healthy-control classification. 20 tab-separated files (10 stroke, 10 healthy control), each containing ~201 spectra across 723 wavenumber points (202.985–1999.92 cm⁻¹). ~4,020 spectra total. |
| `covid19_salvia` | Medical & Clinical | Classification | Curated for non-invasive SARS-CoV-2 screening. Includes ~25 spectral replicates per subject from 101 patients (positive, negative symptomatic, and healthy controls) acquired from dried saliva drops using a 785 nm spectrometer. |
| `covid19_serum` | Medical & Clinical | Classification | Raman spectra of blood serum from 20 subjects (10 RT-PCR/ELISA-confirmed COVID-19 patients and 10 healthy controls) collected under Ethics Committee protocol 26691419.6.0000.5492 (Universidade Anhembi Morumbi). Acquired with a Dimension P1 dispersive Raman spectrometer (Lambda Solutions), 830 nm exc |
| `deepr_denoising` | Unknown | Denoising | Raman spectral denoising dataset from DeepeR paper. Contains noisy input spectra and corresponding denoised target spectra for training deep learning denoising models. |
| `deepr_super_resolution` | Unknown | SuperResolution | Hyperspectral super-resolution dataset from DeepeR paper. Contains low-resolution input spectra and high-resolution target spectra for training super-resolution models. |
| `diabetes_skin_ages` | Medical & Clinical | Classification | In vivo portable Raman spectra of human skin at the AGEs site for DM2 screening. 11 type 2 diabetes patients (7F, 49.5±6.7 y) and 9 healthy controls (7F, 33.2±4.9 y), University of Guanajuato, Mexico, IRB approved. PEK-785 spectrometer (Agiltron), 785 nm, 90 mW, 200 µm spot, 12 cm⁻¹ resolution, 5 sc |
| `diabetes_skin_ear_lobe` | Medical & Clinical | Classification | In vivo portable Raman spectra of human skin at the Ear Lobe site for DM2 screening. 11 type 2 diabetes patients (7F, 49.5±6.7 y) and 9 healthy controls (7F, 33.2±4.9 y), University of Guanajuato, Mexico, IRB approved. PEK-785 spectrometer (Agiltron), 785 nm, 90 mW, 200 µm spot, 12 cm⁻¹ resolution, |
| `diabetes_skin_inner_arm` | Medical & Clinical | Classification | In vivo portable Raman spectra of human skin at the Inner Arm site for DM2 screening. 11 type 2 diabetes patients (7F, 49.5±6.7 y) and 9 healthy controls (7F, 33.2±4.9 y), University of Guanajuato, Mexico, IRB approved. PEK-785 spectrometer (Agiltron), 785 nm, 90 mW, 200 µm spot, 12 cm⁻¹ resolution, |
| `diabetes_skin_thumbnail` | Medical & Clinical | Classification | In vivo portable Raman spectra of human skin at the Thumbnail site for DM2 screening. 11 type 2 diabetes patients (7F, 49.5±6.7 y) and 9 healthy controls (7F, 33.2±4.9 y), University of Guanajuato, Mexico, IRB approved. PEK-785 spectrometer (Agiltron), 785 nm, 90 mW, 200 µm spot, 12 cm⁻¹ resolution, |
| `diabetes_skin_vein` | Medical & Clinical | Classification | In vivo portable Raman spectra of human skin at the Vein site for DM2 screening. 11 type 2 diabetes patients (7F, 49.5±6.7 y) and 9 healthy controls (7F, 33.2±4.9 y), University of Guanajuato, Mexico, IRB approved. PEK-785 spectrometer (Agiltron), 785 nm, 90 mW, 200 µm spot, 12 cm⁻¹ resolution, 5 sc |
| `ecoli_fermentation` | Biological & Biotechnological | Regression | Spectra captured during batch and fed-batch fermentation of E. coli. Measurements were performed on the supernatant using a 785 nm spectrometer to track glucose and acetate concentrations in a dynamic, high-throughput bioprocess environment. |
| `ecoli_metabolites` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose and sodium acetate which are the most important metabolites during Ecoli Fermentations. The spectra were measured with a liquid handling station and a system for automatic Raman spectra measurements used in  High-Throughput Experimentation |
| `ecoli_metabolites_dig4bio` | Biological & Biotechnological | Regression | This dataset contains Raman spectra of mixtures of glucose, sodium acetate, and magnesium sulfate. These components are important during E. Coli fermentation processes. The spectra were measured with a liquid handling station and a system for automatic Raman spectra measurements used in  High-Throug |
| `flow_microgel_synthesis` | Chemical & Industrial | Regression | This data set contains in-line Raman spectroscopy measurements and predicted microgel sizes from Dynamic Light Scattering (DLS).The Raman spectroscopy measurements were conducted inside a customized measurement cell for monitoring in a tubular flow reactor.Inside the flow reactor, the microgel synth |
| `formic_acid_species` | Chemical & Industrial | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `fuel_benchtop` | Chemical & Industrial | Regression | Raman spectra from 179 commercial gasoline samples recorded using a benchtop 1064 nm FT-Raman system. Targets include Research Octane Number (RON), Motor Octane Number (MON), and oxygenated additive concentrations. |
| `fuel_handheld` | Chemical & Industrial | Regression | Counterpart to the benchtop fuel dataset, acquired from the same 179 samples using a handheld 785 nm spectrometer. Used for benchmarking model transferability across different hardware and wavelengths. |
| `hair_dyes_sers` | Chemical & Industrial | Classification | SERS spectra of commercial hair dye products acquired with a portable Raman spectrometer. Each spectrum is labelled by brand, permanence (permanent/semi-permanent/temporary), and colour. Target: brand identity (classification). |
| `head_neck_cancer` | Medical & Clinical | Classification | Raman spectra of blood plasma and saliva samples from head and neck cancer patients and healthy controls. Acquired for non-invasive liquid biopsy screening. Target: cancer vs. control (binary classification). |
| `ht_raman_bio_catalysis_axp` | Biological & Biotechnological | Regression | This dataset consists of Raman spectra tailored for the real-time monitoring of biocatalytic reactions. A key feature of this data is the use of Deep Eutectic Solvents (DES) as the reaction medium. |
| `illicit_adulterants_ft_raman` | Medical & Clinical | Classification | FT-Raman reference spectra (1064 nm, 50–3600 cm⁻¹, 4 cm⁻¹ resolution, 32 scans) of 11 SERS-active pharmaceutically active adulterants found in dietary supplements, acquired with a benchtop Bruker RAM II FT-IR Raman module (OPUS v7.2, 300 mW). Compounds selected from RASFF portal EU alerts. Library s |
| `illicit_adulterants_sers` | Medical & Clinical | Classification | Portable SERS spectra (785 nm, 400–2300 cm⁻¹, 8 cm⁻¹ FWHM, 3 scans) of 11 SERS-active illicit adulterants in dietary supplements. Acquired with a Metrohm Instant SERS analyzer (MISA) using silver printed-SERS (p-SERS) substrates and Orbital Raster Scan (ORS™) technology (30 µm spot, ~2 mm raster, 10 |
| `itaconic_acid_species` | Chemical & Industrial | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `kaiser_ecoli_fermentation` | Biological & Biotechnological | Regression | Raman spectra collected during E. coli fermentation using a Kaiser spectrometer. |
| `kaiser_ecoli_fermentation_supernatant` | Biological & Biotechnological | Regression | Raman spectra collected during E. coli fermentation, measured on the supernatant using a Kaiser spectrometer. |
| `levulinic_acid_species` | Chemical & Industrial | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `microgel_size_lf_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Linear Fit, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_lf_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Linear Fit, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_lf_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Linear Fit, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_lf_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Linear Fit, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_rb_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Rubber Band, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_rb_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Rubber Band, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_raw_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Raw, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_raw_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Raw, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_rb_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Rubber Band, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_rb_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Rubber Band, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_lf_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Linear Fit, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_lf_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Linear Fit, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_rb_fingerprint` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Rubber Band, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_rb_global` | Chemical & Industrial | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Rubber Band, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_synthesis` | Chemical & Industrial | Regression | This data set contains in-line Raman spectroscopy measurements inside a customized measurement cell for monitoring in a tubular flow reactor. The setup aims at monitoring the microgel synthesis in a flow reactor while aiming at a high measurement precision. The measurements include a systematic accu |
| `microplastics_weathered` | Material Science | Classification | Raman spectra of field-collected weathered microplastic debris (155 samples) and unweathered standard plastics (18 samples), acquired from sediments near waste plastics recycling industries in Laizhou City, Shandong Province, China (August 2018). WITec alpha300-R confocal Raman (532 nm, 5 mW typical |
| `mlrod` | Material Science | Classification | 500,000+ point-mapping Raman spectra of Mars-analogue geological samples acquired on a Horiba LabRAM HR Evolution (532 nm, 100 mW, 1800 l/mm grating, ~1 cm⁻¹ resolution, open-electrode CCD at −60 °C, 50× LWD NA 0.5 objective, 0.1 µm XY step, 100–1800 cm⁻¹). Samples include gabbro (Madagascar), grani |
| `organic_compounds_preprocess` | Chemical & Industrial | Classification | Preprocess Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data. |
| `organic_compounds_raw` | Chemical & Industrial | Classification | Raw Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data. |
| `parkinson` | Medical & Clinical | Classification | Raman spectra from dried saliva drops targeting Parkinson's Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment. |
| `pharmaceutical_ingredients` | Medical & Clinical | Classification | A Raman spectral dataset comprising 3,510 spectra from 32 chemical substances. This dataset includes organic solvents and reagents commonly used in API development, along with information regarding the products in the XLSX, and code to visualise and perform technical validation on the data. |
| `ralstonia_fermentations` | Biological & Biotechnological | Regression | Monitoring of P(HB-co-HHx) copolymer synthesis in Ralstonia eutropha batch cultivations. Includes a hybrid mix of experimental and high-fidelity synthetic data to handle high multicollinearity between process variables. |
| `rruff_mineral_preprocess` | Material Science | Classification | Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm). |
| `rruff_mineral_raw` | Material Science | Classification | Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm). |
| `serum_alzheimer_disease` | Medical & Clinical | Classification | SERS serum metabolite spectra for classifying Alzheimer's disease (AD), mild cognitive impairment (MCI), and healthy controls. 139 serum samples (57 male, 82 female) collected at Rui Jin Hospital, Shanghai Jiao Tong University. Organized as SERSomes (200 spectra per sample, 638 nm laser, quartz capi |
| `serum_prostate_cancer` | Medical & Clinical | Classification | SERS serum metabolite spectra for classifying prostate cancer (PCa), benign prostatic hyperplasia (BPH), and healthy controls. 424 serum samples from male participants (ages 41–89) collected at Ren Ji Hospital, Shanghai Jiao Tong University. Organized as SERSomes (200 spectra per sample, 638 nm lase |
| `streptococcus_thermophilus_fermentation_kaiser` | Biological & Biotechnological | Regression | This dataset contains offline Raman spectra collected during batch cultivations of Streptococcus thermophilus. The spectra were measured using a Kaiser RXN1. The dataset includes two distinct fermentation runs conducted in shake flasks over a 24-hour period. |
| `streptococcus_thermophilus_fermentation_timegate` | Biological & Biotechnological | Regression | This dataset contains offline Raman spectra collected during batch cultivations of Streptococcus thermophilus. Unlike conventional continuous-wave Raman, these measurements were captured using Time-Gated Raman Spectroscopy. The dataset includes two distinct fermentation runs conducted in shake flask |
| `succinic_acid_species` | Chemical & Industrial | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `sugar_mixtures_high_snr` | Chemical & Industrial | Regression | The high signal-to-noise ratio subset of the Sugar Mixtures benchmark (7,680 measurements at 0.5 s integration). Used for evaluating the noise-robustness of hyperspectral unmixing and quantification algorithms. |
| `sugar_mixtures_low_snr` | Chemical & Industrial | Regression | The low signal-to-noise ratio subset of the Sugar Mixtures benchmark (7,680 measurements at 0.5 s integration). Used for evaluating the noise-robustness of hyperspectral unmixing and quantification algorithms. |
| `synthetic_organic_pigments_baseline_corrected` | Material Science | Regression | Baseline Corrected Raman spectral library of approximately 270 synthetic organic pigments (~300 reference spectra, nearly 400 samples total) compiled at the Royal Institute for Cultural Heritage (KIK/IRPA), Brussels. Acquired on a Renishaw inVia dispersive Raman spectrometer (785 nm, Peltier-cooled |
| `synthetic_organic_pigments_raw` | Material Science | Regression | Raw Raman spectral library of approximately 270 synthetic organic pigments (~300 reference spectra, nearly 400 samples total) compiled at the Royal Institute for Cultural Heritage (KIK/IRPA), Brussels. Acquired on a Renishaw inVia dispersive Raman spectrometer (785 nm, Peltier-cooled CCD 203 K, 1200 |
| `tg_ecoli_fermentation` | Biological & Biotechnological | Regression | Raman spectra collected during E. coli fermentation using Time-Gated Raman Spectroscopy. |
| `tg_ecoli_fermentation_supernatant` | Biological & Biotechnological | Regression | Raman spectra collected during E. coli fermentation, measured on the supernatant using Time-Gated Raman Spectroscopy. |
| `wheat_lines` | Biological & Biotechnological | Classification | Raman spectra from the 7th generation of salt-stress-tolerant wheat mutant lines and their commercial cultivars. Features 785 nm excitation and tracks biochemical shifts in carotenoids and protein-related bands for agricultural phenotyping. |
| `yeast_fermentation` | Biological & Biotechnological | Regression | This dataset contains Raman spectra acquired during the continuous ethanolic fermentation of sucrose using Saccharomyces cerevisiae (Baker's yeast). To facilitate continuous processing and high-quality optical measurements, the yeast cells were immobilized in calcium alginate beads. |

<!-- AUTO-GENERATED: END - datasets table. -->
<!-- DATASETS_TABLE_END -->

</details>

## Contributing a Dataset

> **Just want to suggest a dataset without implementing it?**
> Add it to the community proposals list:
> [PROPOSED_DATASETS.md](https://github.com/ml-lab-htw/RamanBench/blob/main/PROPOSED_DATASETS.md)
> (in the RamanBench repo). It's the running queue we draw from for the next
> benchmark release. The full implementation steps below are only needed if
> you want to add a loader yourself.

We welcome contributions of new Raman datasets, especially from underrepresented domains, novel instrumentation, or larger sample sizes.
A dataset is eligible for inclusion if it:

- Contains **real, experimentally acquired** 1D Raman spectra (not synthetic or simulated)
- Is **publicly available** without access restrictions or upon-request-only policies
- Provides **ground-truth labels** (class labels for classification; continuous values for regression)
- Is accompanied by a **citable reference** (paper, report, or dataset DOI)

### Step-by-step

**1. Host your data publicly.**
Upload your dataset to [HuggingFace Datasets](https://huggingface.co/docs/datasets), [Zenodo](https://zenodo.org), [Figshare](https://figshare.com), or [Kaggle](https://www.kaggle.com/datasets).
HuggingFace is preferred for its versioning and direct streaming support — see [`dataset_to_huggingface.md`](dataset_to_huggingface.md) for a step-by-step guide.

**2. Choose the right loader.**
Add your dataset entry to the corresponding loader file in `raman_data/loaders/`:

| Source | Loader file |
|--------|-------------|
| HuggingFace | `HuggingFaceLoader.py` |
| Zenodo | `ZenodoLoader.py` |
| Figshare | `FigshareLoader.py` |
| Kaggle | `KaggleLoader.py` |
| Other URL (ZIP/file) | `MiscLoader.py` |

**3. Add a dataset entry.**
Each loader contains a `DATASETS` dict mapping a unique string ID to a `DatasetInfo` object. Add an entry following the existing pattern:

```python
from raman_data.types import DatasetInfo, TASK_TYPE, APPLICATION_TYPE

"your_dataset_id": DatasetInfo(
    id="your_dataset_id",               # unique snake_case identifier
    name="Your Dataset Name",           # human-readable name
    short_name="Short Name",            # for tables and figures (≤ 30 chars)
    task_type=TASK_TYPE.Regression,     # or TASK_TYPE.Classification
    application_type=APPLICATION_TYPE.Biological,
    license="CC BY 4.0",
    loader=lambda df: _load_your_dataset(df),
    metadata={
        "source": "https://doi.org/...",    # DOI or URL of the dataset
        "paper": "https://doi.org/...",     # DOI of the associated paper
        "description": "One-sentence description of the dataset and task.",
    },
)
```

**4. Implement the loader function.**
The loader receives the raw data (e.g. a HuggingFace dataset or a file path) and must return a `RamanDataset`:

```python
def _load_your_dataset(raw_data) -> RamanDataset:
    spectra      = ...  # np.ndarray, shape (n_samples, n_wavenumbers)
    raman_shifts = ...  # np.ndarray, shape (n_wavenumbers,), values in cm⁻¹
    targets      = ...  # np.ndarray, shape (n_samples,) or (n_samples, n_targets)
    return RamanDataset(spectra=spectra, raman_shifts=raman_shifts, targets=targets)
```

**5. Add tests and open a pull request.**
Add a test in `tests/` that loads your dataset and checks basic properties (shape, dtype, value ranges).
Then open a pull request — we will review it and, if it meets the inclusion criteria, merge it and include it in the next RamanBench release.

## Releasing a New Version

Releases are automated via GitHub Actions. A `v*.*.*` git tag triggers
tests, build, and PyPI publish.

See **[RELEASING.md](RELEASING.md)** for the full step-by-step checklist
(README regeneration, smoke tests, and common gotchas).

Short version:

```bash
git checkout main && git pull
PYTHONPATH=. python3 scripts/generate_readme_datasets.py  # refresh table
git add -A && git commit -m "Release vX.Y.Z"
git push origin main
git tag vX.Y.Z && git push origin vX.Y.Z
```

The CI workflow runs at [github.com/ml-lab-htw/raman_data/actions](https://github.com/ml-lab-htw/raman_data/actions) and publishes to [PyPI](https://pypi.org/project/raman-data/).

## Citation

If you use `raman-data` in your research, please cite the RamanBench paper:

```bibtex
@misc{koddenbrock2026ramanbench,
  title         = {{RamanBench}: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy},
  author        = {Koddenbrock, Mario and Lange, Christoph and Legner, Robin and Jaeger, Martin
                   and K{\"o}gler, Martin and Cruz Bournazou, Mariano N. and Neubauer, Peter
                   and Bie{\ss}mann, Felix and Rodner, Erik},
  year          = {2026},
  eprint        = {2605.02003},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.02003}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
The individual datasets retain their original licenses as specified in each dataset's metadata.
