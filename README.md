# Raman-Data: A Unified Python Library for Raman Spectroscopy Datasets

[![PyPI version](https://badge.fury.io/py/raman-data.svg)](https://pypi.org/project/raman-data/)
[![GitHub](https://img.shields.io/github/license/ml-lab-htw/raman_data)](https://github.com/ml-lab-htw/raman_data/blob/main/LICENSE)

This project aims to create a unified Python package for accessing various Raman spectroscopy datasets. The goal is to provide a simple and consistent API to load data from different sources like Kaggle, Hugging Face, GitHub, and Zenodo. This will be beneficial for the Raman spectroscopy community, enabling easier evaluation of models, such as foundation models for Raman spectroscopy.

## ✨ Features

- A single, easy-to-use Python package available on [PyPI](https://pypi.org/project/raman-data/).
- Automatic downloading and caching of datasets from their original sources.
- A unified data format for all datasets.
- A simple function to list available datasets, with filtering options.
- Datasets are annotated with an **application domain** (`APPLICATION_TYPE`) for easy filtering:
  - `MaterialScience` -- mineral identification, pigment libraries
  - `Biological` -- bioprocess monitoring, fermentation, agricultural phenotyping
  - `Medical` -- clinical diagnostics, pathogen identification, disease screening
  - `Chemical` -- fuel analysis, chemical quantification, polymer characterisation

## 📦 Installation

Install directly from PyPI:

```bash
pip install raman-data
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/ml-lab-htw/raman_data.git
cd raman_data

# Install the package
pip install -e .
```

**Note:** For Kaggle datasets, you need to configure your Kaggle API credentials. See [Kaggle API documentation](https://www.kaggle.com/docs/api) for details.

## 🚀 Getting Started

The basic interface for the package is defined in `raman_data/__init__.py`. Here's a preview of how it works:

```python
from raman_data import raman_data
# To specify a task type or application domain, import these enums as well
from raman_data import TASK_TYPE, APPLICATION_TYPE

# List all available datasets
print(raman_data())

# List only classification datasets
print(raman_data(task_type=TASK_TYPE.Classification))

# List only medical datasets
print(raman_data(application_type=APPLICATION_TYPE.Medical))

# Combine filters: only medical classification datasets
print(raman_data(task_type=TASK_TYPE.Classification, application_type=APPLICATION_TYPE.Medical))

# Load a dataset by name
dataset = raman_data(dataset_name="codina_diabetes_AGEs")

# Access the spectra (intensity data), raman_shifts (wavenumbers), targets, and metadata
spectra = dataset.spectra           # 2D array: (n_samples, n_wavenumbers)
raman_shifts = dataset.raman_shifts # 1D array: wavenumber values in cm⁻¹
targets = dataset.targets            # Target labels or values
metadata = dataset.metadata         # Dataset metadata (source, paper, description)

print(f"Number of spectra: {dataset.n_spectra}")
print(f"Raman shift range: {dataset.min_shift} - {dataset.max_shift} cm⁻¹")
print(metadata)
```

For more detailed examples see [Demo Notebook](examples/demo.ipynb) or [Demo Script](examples/example_all.py).

<!-- DATASETS_TABLE_START -->
<!-- AUTO-GENERATED: START - datasets table. Do not edit manually. -->

| Dataset Name | Application | Task Type | Description |
|--------------|-------------|-----------|-------------|
| `acid_species_acetic` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `acid_species_citric` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `acid_species_formic` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `acid_species_itaconic` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `acid_species_levulinic` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `acid_species_succinic` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `adenine_colloidal_gold` | Chemical | Regression | Quantitative SERS spectra of adenine measured using colloidal gold substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_colloidal_silver` | Chemical | Regression | Quantitative SERS spectra of adenine measured using colloidal silver substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_solid_gold` | Chemical | Regression | Quantitative SERS spectra of adenine measured using solid gold substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_solid_silver` | Chemical | Regression | Quantitative SERS spectra of adenine measured using solid silver substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `alzheimer` | Medical | Classification | Raman spectra from dried saliva drops targeting Alzheimer's Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment. |
| `amino_acids_glycine` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Glycine elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `amino_acids_leucine` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Leucine elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `amino_acids_phenylalanine` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Phenylalanine elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `amino_acids_tryptophan` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Tryptophan elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `bacteria_identification` | Medical | Classification | 60,000 spectra from 30 clinically relevant bacterial and yeast isolates (including an MRSA/MSSA isogenic pair). Acquired with 633 nm illumination on gold-coated silica substrates with low SNR to simulate rapid clinical acquisition times. |
| `bioprocess_substrates` | Biological | Regression | A benchmark dataset of 6,960 spectra featuring eight key metabolites (glucose, glycerol, acetate, etc.) sampled via a statistically independent uniform distribution. Designed to evaluate regression robustness against common bioprocess correlations, including background effects from mineral salts and |
| `cancer_cell_(cooh)2` | Biological | Classification | SERS spectra of cancer cell metabolites collected on gold nanourchins functionalized with the (COOH)2 moiety. Designed to provide specificity toward specific proteins and lipids for cell line identification. |
| `cancer_cell_cooh` | Biological | Classification | SERS spectra of cancer cell metabolites collected on gold nanourchins functionalized with the COOH moiety. Designed to provide specificity toward specific proteins and lipids for cell line identification. |
| `cancer_cell_nh2` | Biological | Classification | SERS spectra of cancer cell metabolites collected on gold nanourchins functionalized with the NH2 moiety. Designed to provide specificity toward specific proteins and lipids for cell line identification. |
| `covid19_salvia` | Medical | Classification | Curated for non-invasive SARS-CoV-2 screening. Includes ~25 spectral replicates per subject from 101 patients (positive, negative symptomatic, and healthy controls) acquired from dried saliva drops using a 785 nm spectrometer. |
| `covid19_serum` | Medical | Classification | This study proposed the diagnosis of COVID-19 by means of Raman spectroscopy. Samples of blood serum from 10 patients positive and 10 patients negative for COVID-19 by RT-PCR RNA and ELISA tests were analyzed. |
| `diabetes_skin_ages` | Medical | Classification | Part of the Diabetes Skin Raman Dataset. This subset focuses on Advanced Glycation End-products (AGEs) signatures in the skin. Data acquired in vivo using a portable 785 nm Raman spectrometer to discern between diabetic patients and healthy controls. |
| `diabetes_skin_ear_lobe` | Medical | Classification | Part of the Diabetes Skin Raman Dataset. This subset focuses on Advanced Glycation End-products (Ear Lobe) signatures in the skin. Data acquired in vivo using a portable 785 nm Raman spectrometer to discern between diabetic patients and healthy controls. |
| `diabetes_skin_inner_arm` | Medical | Classification | Part of the Diabetes Skin Raman Dataset. This subset focuses on Advanced Glycation End-products (Inner Arm) signatures in the skin. Data acquired in vivo using a portable 785 nm Raman spectrometer to discern between diabetic patients and healthy controls. |
| `diabetes_skin_thumbnail` | Medical | Classification | Part of the Diabetes Skin Raman Dataset. This subset focuses on Advanced Glycation End-products (Thumbnail) signatures in the skin. Data acquired in vivo using a portable 785 nm Raman spectrometer to discern between diabetic patients and healthy controls. |
| `diabetes_skin_vein` | Medical | Classification | Part of the Diabetes Skin Raman Dataset. This subset focuses on Advanced Glycation End-products (Vein) signatures in the skin. Data acquired in vivo using a portable 785 nm Raman spectrometer to discern between diabetic patients and healthy controls. |
| `ecoli_fermentation` | Biological | Regression | Spectra captured during batch and fed-batch fermentation of E. coli. Measurements were performed on the supernatant using a 785 nm spectrometer to track glucose and acetate concentrations in a dynamic, high-throughput bioprocess environment. |
| `flow_microgel_synthesis` | Chemical | Regression | This data set contains in-line Raman spectroscopy measurements and predicted microgel sizes from Dynamic Light Scattering (DLS).The Raman spectroscopy measurements were conducted inside a customized measurement cell for monitoring in a tubular flow reactor.Inside the flow reactor, the microgel synth |
| `fuel_benchtop` | Chemical | Regression | Raman spectra from 179 commercial gasoline samples recorded using a benchtop 1064 nm FT-Raman system. Targets include Research Octane Number (RON), Motor Octane Number (MON), and oxygenated additive concentrations. |
| `fuel_handheld` | Chemical | Regression | Counterpart to the benchtop fuel dataset, acquired from the same 179 samples using a handheld 785 nm spectrometer. Used for benchmarking model transferability across different hardware and wavelengths. |
| `microgel_size_lf_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Linear Fit, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_lf_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Linear Fit, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_lf_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Linear Fit, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_lf_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Linear Fit, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_rb_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Rubber Band, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_mm_rb_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: MinMax + Rubber Band, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_raw_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Raw, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_raw_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Raw, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_rb_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Rubber Band, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_rb_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: Rubber Band, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_lf_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Linear Fit, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_lf_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Linear Fit, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_rb_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Rubber Band, spectral range: FingerPrint. Task: predict particle diameter from Raman spectrum. |
| `microgel_size_snv_rb_global` | Chemical | Regression | Raman spectra of 235 microgel samples with DLS-measured particle diameters (208–483 nm). Pretreatment: SNV + Rubber Band, spectral range: Global. Task: predict particle diameter from Raman spectrum. |
| `microgel_synthesis` | Chemical | Regression | This data set contains in-line Raman spectroscopy measurements inside a customized measurement cell for monitoring in a tubular flow reactor. The setup aims at monitoring the microgel synthesis in a flow reactor while aiming at a high measurement precision. The measurements include a systematic accu |
| `organic_compounds_preprocess` | Chemical | Classification | Preprocess Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data. |
| `organic_compounds_raw` | Chemical | Classification | Raw Raman spectra of organic compounds collected with several different excitation sources. Designed to benchmark transfer learning and domain adaptation for chemical identification with limited data. |
| `parkinson` | Medical | Classification | Raman spectra from dried saliva drops targeting Parkinson's Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment. |
| `pharmaceutical_ingredients` | Medical | Classification | A Raman spectral dataset comprising 3,510 spectra from 32 chemical substances. This dataset includes organic solvents and reagents commonly used in API development, along with information regarding the products in the XLSX, and code to visualise and perform technical validation on the data. |
| `ralstonia_fermentations` | Biological | Regression | Monitoring of P(HB-co-HHx) copolymer synthesis in Ralstonia eutropha batch cultivations. Includes a hybrid mix of experimental and high-fidelity synthetic data to handle high multicollinearity between process variables. |
| `rruff_mineral_preprocess` | MaterialScience | Classification | Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm). |
| `rruff_mineral_raw` | MaterialScience | Classification | Comprehensive resource of raw Raman spectra for over 1,000 mineral species, representing a diverse array of crystallographic structures and chemical compositions measured under varying experimental conditions (e.g., 532 nm and 785 nm). |
| `sop_spectral_library_baseline_corrected` | MaterialScience | Regression | Baseline Corrected Raman spectral library comprising nearly 300 reference spectra of synthetic organic pigments (SOPs). Designed for spectral matching and identification of pigments in modern and contemporary art conservation. |
| `sop_spectral_library_raw` | MaterialScience | Regression | Raw Raman spectral library comprising nearly 300 reference spectra of synthetic organic pigments (SOPs). Designed for spectral matching and identification of pigments in modern and contemporary art conservation. |
| `sugar_mixtures_high_snr` | Chemical | Regression | The high signal-to-noise ratio subset of the Sugar Mixtures benchmark (7,680 measurements at 0.5 s integration). Used for evaluating the noise-robustness of hyperspectral unmixing and quantification algorithms. |
| `sugar_mixtures_low_snr` | Chemical | Regression | The low signal-to-noise ratio subset of the Sugar Mixtures benchmark (7,680 measurements at 0.5 s integration). Used for evaluating the noise-robustness of hyperspectral unmixing and quantification algorithms. |
| `wheat_lines` | Biological | Classification | Raman spectra from the 7th generation of salt-stress-tolerant wheat mutant lines and their commercial cultivars. Features 785 nm excitation and tracks biochemical shifts in carotenoids and protein-related bands for agricultural phenotyping. |
| `yeast_fermentation` | Biological | Regression | This dataset contains Raman spectra acquired during the continuous ethanolic fermentation of sucrose using Saccharomyces cerevisiae (Baker's yeast). To facilitate continuous processing and high-quality optical measurements, the yeast cells were immobilized in calcium alginate beads. |

<!-- AUTO-GENERATED: END - datasets table. -->
<!-- DATASETS_TABLE_END -->

## 📊 RamanDataset Class

Each loaded dataset returns a `RamanDataset` object with the following attributes and properties:

| Attribute/Property | Type | Description |
|-------------------|------|-------------|
| `spectra` | `np.ndarray` | Raman spectra intensity data (2D: samples × wavenumbers, or 2D/object array if variable) |
| `raman_shifts` | `np.ndarray` | Wavenumber/Raman shift values in cm⁻¹ (1D array, or 2D/object array if variable) |
| `targets` | `np.ndarray` | Target labels (classification) or values (regression) |
| `metadata` | `dict` | Dataset metadata including source, paper, and description |
| `name` | `str` | Name of the dataset |
| `task_type` | `TASK_TYPE` | Classification or Regression |
| `application_type` | `APPLICATION_TYPE` | Application domain (MaterialScience, Biological, Medical, Chemical) |
| `n_spectra` | `int` | Number of spectra in the dataset |
| `n_frequencies` | `int` | Number of frequency points per spectrum |
| `n_raman_shifts` | `int` | Number of Raman shift values |
| `n_classes` | `int \| None` | Number of classes (classification only) |
| `class_names` | `list \| None` | Unique class names (classification only) |
| `target_range` | `tuple \| None` | (min, max) targets values (regression only) |
| `min_shift` | `float` | Minimum Raman shift value |
| `max_shift` | `float` | Maximum Raman shift value |

**Support for Datasets with Multiple Raman Shifts:**

- If all spectra share identical raman_shifts, `raman_shifts` is a 1D array and `spectra` is a 2D array (n_samples × n_points).
- If all spectra have the same number of points but different raman_shift values, both `raman_shifts` and `spectra` are 2D arrays (n_samples × n_points).
- If spectra have different numbers of points, both `raman_shifts` and `spectra` are returned as 1D object arrays, where each entry is a 1D array for that sample.
- This allows the library to support real-world datasets with variable or non-uniform spectral grids.

**Note:**
- Downstream code should check the shape and dtype of `raman_shifts` and `spectra` to handle all cases robustly.
- For machine learning, it is recommended to interpolate or pad spectra to a common grid if uniformity is required.

The dataset can also be converted to a pandas DataFrame:

```python
# df = dataset.to_dataframe()
```

## 🎯 Milestones

- [x] View Datasets
- [x] Software architecture with dummy data
- [x] Software tests
- [x] Integration of Kaggle
- [x] Integration of Huggingface
- [x] Integration of Github
- [x] Integration of Zenodo
- [x] Code documentation (docstrings)
- [x] Publish to PyPi
- [ ] Integration of other datasets
- [ ] API documentation website

## 🤝 Contributing

Contributions are welcome! To add a new dataset:

1. Choose the appropriate loader based on the data source:
   - `KaggleLoader` for Kaggle datasets
   - `HuggingFaceLoader` for Hugging Face datasets
   - `ZenodoLoader` for Zenodo datasets
   - `ZipLoader` for other URL-based sources
   - `MiscLoader` for datasets that do not fit into the above categories (e.g., DeepeR)

2. Implement a loader function that returns a tuple of `(spectra, raman_shifts, targets)`:
   - `spectra`: 2D numpy array of intensity values (samples × wavenumbers)
   - `raman_shifts`: 1D numpy array of wavenumber values in cm⁻¹
   - `targets`: numpy array of target labels or values

3. Add the dataset to the loader's `DATASETS` dictionary with appropriate metadata.

4. Add tests for the new dataset.

## 🔮 For Later (Future Datasets)

### Remaining / For Later (still not integrated)
The following items remain to be added (suggested action: add a `DATASETS` entry under the appropriate loader, or add a loader placeholder with the source link):

- High-throughput molecular imaging (DeepeR)
  - URL: https://github.com/conor-horgan/DeepeR?tab=readme-ov-file#dataset
  - Suggested loader: `MiscLoader` or `ZipLoader` depending on availability of packaged data
  - Notes: README points to datasets; may require dataset-specific processing

- spectrai raman spectra
  - URL: https://github.com/conor-horgan/spectrai
  - Suggested loader: `MiscLoader` / `ZipLoader`

- Quantitative volumetric Raman imaging (Zenodo record)
  - URL: https://zenodo.org/records/256329
  - Suggested loader: `ZenodoLoader`

- Spectra of illicit adulterants (Mendeley)
  - URL: https://data.mendeley.com/datasets/y4md8znppn/1
  - Suggested loader: `ZipLoader` / `MiscLoader`

- Raman spectra of chemical compounds (Springer / figshare)
  - URL: https://springernature.figshare.com/articles/dataset/Open-source_Raman_spectra_of_chemical_compounds_for_active_pharmaceutical_ingredient_development/27931131
  - Suggested loader: `ZipLoader`

- Inline Raman Spectroscopy and Indirect Hard Modeling
  - URL: https://publications.rwth-aachen.de/record/978266/files/
  - Suggested loader: `ZipLoader` (file formats may be non-standard)

- The Effect of Sulfate Electrolytes on the Liquid-Liquid Equilibrium
  - URL: https://publications.rwth-aachen.de/record/978265/files/
  - Suggested loader: `ZipLoader`

- In-line Monitoring of Microgel Synthesis (weird format)
  - URL: https://publications.rwth-aachen.de/record/834113/files/
  - Suggested loader: `ZipLoader` (may require manual preprocessing)

- N-isopropylacrylamide Microgel Synthesis
  - URL: https://publications.rwth-aachen.de/record/959050/files/
  - Suggested loader: `ZipLoader`

- Nonlinear Manifold Learning Determines Microgel Size from Raman Spectroscopy
  - URL: https://publications.rwth-aachen.de/record/959137
  - Suggested loader: `ZipLoader`

- NASA AHEAD dataset
  - URL: https://ahed.nasa.gov/datasets/f5b6051bfeb18c5a7eaef6504582
  - Suggested loader: `ZipLoader` / `MiscLoader`
