# Raman-Data: A Unified Python Library for Raman Spectroscopy Datasets

[![PyPI version](https://badge.fury.io/py/raman-data.svg)](https://pypi.org/project/raman-data/)
[![GitHub](https://img.shields.io/github/license/ml-lab-htw/raman_data)](https://github.com/ml-lab-htw/raman_data/blob/main/LICENSE)

This project aims to create a unified Python package for accessing various Raman spectroscopy datasets. The goal is to provide a simple and consistent API to load data from different sources like Kaggle, Hugging Face, GitHub, and Zenodo. This will be beneficial for the Raman spectroscopy community, enabling easier evaluation of models, such as foundation models for Raman spectroscopy.

## ✨ Features

- A single, easy-to-use Python package available on [PyPI](https://pypi.org/project/raman-data/).
- Automatic downloading and caching of datasets from their original sources.
- A unified data format for all datasets.
- A simple function to list available datasets, with filtering options.

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
# To specify a task type import this enum as well
from raman_data import TASK_TYPE

# List all available datasets
print(raman_data())

# List only classification datasets
print(raman_data(task_type=TASK_TYPE.Classification))

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

| Dataset Name | Source | Task Type | Description |
|--------------|--------|-----------|-------------|
| `adenine` | Zenodo Datasets | Regression | This dataset contains all the spectra used in "Surface Enhanced Raman Spectroscopy for quantitative analysis: results of a large-scale European multi-instrument interlaboratory study". Data are available in 2 different formats: - a compressed archive with 1 folder ("Dataset") cointaining all the 351 |
| `andriitrelin_cells-raman-spectra_(COOH)2` | Kaggle Datasets | Classification | SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with (COOH)2 moiety. Contains 12 cell type classes for classification. |
| `andriitrelin_cells-raman-spectra_COOH` | Kaggle Datasets | Classification | SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with COOH moiety. Contains 12 cell type classes for classification. |
| `andriitrelin_cells-raman-spectra_NH2` | Kaggle Datasets | Classification | SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with NH2 moiety. Contains 12 cell type classes for classification. |
| `chlange_FuelRamanSpectraBenchtop` | Hugging Face Datasets | Regression | This dataset contains Raman spectra for the analysis and prediction of key parameters in commercial fuel samples (gasoline). It includes spectra of 179 fuel samples from various refineries. |
| `chlange_RamanSpectraEcoliFermentation` | Hugging Face Datasets | Regression | Dataset Card for Raman Spectra from High-Throughput Bioprocess Fermentations of E. Coli. Raman spectra were obtained during an E. coli fermentation process consisting of a batch and a glucose-limited feeding phase, each lasting about four hours. Samples were automatically collected hourly, centrifug |
| `chlange_SubstrateMixRaman` | Hugging Face Datasets | Regression | This dataset, designed for biotechnological applications, provides a valuable resource for calibrating models used in high-throughput bioprocess development, particularly for bacterial fermentations. It features Raman spectra of samples containing varying, statistically independent concentrations of |
| `codina_diabetes_AGEs` | Kaggle Datasets | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina_diabetes_earLobe` | Kaggle Datasets | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina_diabetes_innerArm` | Kaggle Datasets | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina_diabetes_thumbNail` | Kaggle Datasets | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina_diabetes_vein` | Kaggle Datasets | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `csho33_bacteria` | Miscellaneous Datasets | Classification | Bacterial Raman spectra dataset used in Ho et al. (2019). Add data source or download link to enable loading in the library. |
| `csho33_bacteria_id` | Zip/URL Datasets | Classification |  |
| `dtu_raman-spectrum-matching` | Zip/URL Datasets | Classification |  |
| `HTW-KI-Werkstatt_FuelRamanSpectraHandheld` | Hugging Face Datasets | Regression | Handheld Raman spectra for fuel analysis. Structure similar to FuelRamanSpectraBenchtop. |
| `HTW-KI-Werkstatt_RamanSpectraRalstoniaFermentations` | Hugging Face Datasets | Regression | Raman spectra collected during Ralstonia fermentations. Dataset structure matches HTW-KI-Werkstatt_FuelRamanSpectraHandheld (wavenumber columns + metadata columns). |
| `knowitall_organics_preprocessed` | Miscellaneous Datasets | Classification | Organic (preprocessed) dataset from Transfer-learningbased Raman spectra identification. Organic compounds measured with several excitation sources. |
| `knowitall_organics_raw` | Miscellaneous Datasets | Classification | Organic (raw) dataset from Transfer-learningbased Raman spectra identification. Organic compounds measured with several excitation sources. |
| `mendeley_surface-enhanced-raman` | Zip/URL Datasets | Classification |  |
| `MIND_Lab_covid_and_pd_ad_bundle` | Zip/URL Datasets | Classification |  |
| `mind_covid` | Miscellaneous Datasets | Classification | Per-patient saliva Raman spectra and clinical metadata used for COVID diagnosis study (IRCCS Fondazione Don Carlo Gnocchi, Milano and Centro Spalenza, Rovato). Each patient folder contains spectra.csv, raman_shift.csv and user_information.csv. |
| `mind_pd_ad` | Miscellaneous Datasets | Classification | Per-patient saliva Raman spectra and clinical metadata used for Parkinson's Disease and Alzheimer studies (IRCCS Fondazione Don Carlo Gnocchi and Istituto Auxologico Italiano). Each patient folder contains spectra.csv, raman_shift.csv and user_information.csv. |
| `rruff_mineral_preprocessed` | Miscellaneous Datasets | Classification | Mineral (preprocessed) raman spectra subset from RRUFF database |
| `rruff_mineral_raw` | Miscellaneous Datasets | Classification | Mineral (raw) raman spectra subset from RRUFF database |
| `sergioalejandrod_AminoAcids_glycine` | Kaggle Datasets | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sergioalejandrod_AminoAcids_leucine` | Kaggle Datasets | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sergioalejandrod_AminoAcids_phenylalanine` | Kaggle Datasets | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sergioalejandrod_AminoAcids_tryptophan` | Kaggle Datasets | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sugar_mixtures` | Zenodo Datasets | Regression | Experimental and synthetic Raman data used in Georgiev et al., PNAS (2024) DOI:10.1073/pnas.2407439121. |
| `wheat_lines` | Zenodo Datasets | Classification | Data and codes used in the manuscript titled "DIFFERENTIATION OF ADVANCED GENERATION MUTANT wheat_lines: CONVENTIONAL TECHNIQUES VERSUS RAMAN SPECTROSCOPY". The decision tree model is trained and tested using the Classification Learner app of MATLAB (R2021b, The MathWorks, Inc.). |

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
