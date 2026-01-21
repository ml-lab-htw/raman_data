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
dataset = raman_data(dataset_name="codina/diabetes/AGEs")

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
df = dataset.to_dataframe()
```

<!-- DATASETS_TABLE_START -->
<!-- AUTO-GENERATED: START - datasets table. Do not edit manually. -->

### Hugging Face Datasets

| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `chlange/FuelRamanSpectraBenchtop` | Regression | This dataset contains Raman spectra for the analysis and prediction of key parameters in commercial fuel samples (gasoline). It includes spectra of 179 fuel samples from various refineries. |
| `chlange/RamanSpectraEcoliFermentation` | Regression | Dataset Card for Raman Spectra from High-Throughput Bioprocess Fermentations of E. Coli. Raman spectra were obtained during an E. coli fermentation process consisting of a batch and a glucose-limited feeding phase, each lasting about four hours. Samples were automatically collected hourly, centrifug |
| `chlange/SubstrateMixRaman` | Regression | This dataset, designed for biotechnological applications, provides a valuable resource for calibrating models used in high-throughput bioprocess development, particularly for bacterial fermentations. It features Raman spectra of samples containing varying, statistically independent concentrations of |
| `HTW-KI-Werkstatt/FuelRamanSpectraHandheld` | Regression | Handheld Raman spectra for fuel analysis. Structure similar to FuelRamanSpectraBenchtop. |

### Kaggle Datasets

| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `andriitrelin/cells-raman-spectra/(COOH)2` | Classification | SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with (COOH)2 moiety. Contains 12 cell type classes for classification. |
| `andriitrelin/cells-raman-spectra/COOH` | Classification | SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with COOH moiety. Contains 12 cell type classes for classification. |
| `andriitrelin/cells-raman-spectra/NH2` | Classification | SERS spectra of melanoma cells, melanocytes, fibroblasts, and culture medium collected on gold nanourchins functionalized with NH2 moiety. Contains 12 cell type classes for classification. |
| `codina/diabetes/AGEs` | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina/diabetes/earLobe` | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina/diabetes/innerArm` | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina/diabetes/thumbNail` | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `codina/diabetes/vein` | Classification | This is the dataset of our work where the application of portable Raman spectroscopy coupled with several supervised machine-learning techniques, is used to discern between diabetic patients (DM2) and healthy controls (Ctrl), with a high degree of accuracy. |
| `sergioalejandrod/AminoAcids/glycine` | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sergioalejandrod/AminoAcids/leucine` | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sergioalejandrod/AminoAcids/phenylalanine` | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |
| `sergioalejandrod/AminoAcids/tryptophan` | Classification | This data set was produced by Hirotsugu Hiramatsu as part of his experiment revolving around the enhancement of Raman signal utilizing a vertical flow method. |

### Miscellaneous Datasets

| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `knowitall_organics_preprocessed` | Classification | Organic (preprocessed) dataset from Transfer-learningbased Raman spectra identification. Organic compounds measured with several excitation sources. |
| `knowitall_organics_raw` | Classification | Organic (raw) dataset from Transfer-learningbased Raman spectra identification. Organic compounds measured with several excitation sources. |
| `rruff_mineral_preprocessed` | Classification | Mineral (preprocessed) raman spectra subset from RRUFF database |
| `rruff_mineral_raw` | Classification | Mineral (raw) raman spectra subset from RRUFF database |

### Zenodo Datasets

| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `Adenine` | Regression | This dataset contains all the spectra used in "Surface Enhanced Raman Spectroscopy for quantitative analysis: results of a large-scale European multi-instrument interlaboratory study". Data are available in 2 different formats: - a compressed archive with 1 folder ("Dataset") cointaining all the 351 |
| `sugar mixtures` | Regression | Experimental and synthetic Raman data used in Georgiev et al., PNAS (2024) DOI:10.1073/pnas.2407439121. |
| `Wheat lines` | Classification | Data and codes used in the manuscript titled "DIFFERENTIATION OF ADVANCED GENERATION MUTANT WHEAT LINES: CONVENTIONAL TECHNIQUES VERSUS RAMAN SPECTROSCOPY". The decision tree model is trained and tested using the Classification Learner app of MATLAB (R2021b, The MathWorks, Inc.). |

### Zip/URL Datasets

| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `csho33_bacteria_id` | Classification |  |
| `dtu_raman-spectrum-matching` | Classification |  |
| `mendeley_surface-enhanced-raman` | Classification |  |
| `MIND-Lab_covid+pd_ad_bundle` | Classification |  |

<!-- AUTO-GENERATED: END - datasets table. -->
<!-- DATASETS_TABLE_END -->

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

### Kaggle
- ~~[Cancer Cells SERS Spectra](https://www.kaggle.com/code/mathiascharconnet/cancer-cells-sers-spectra)~~ - Now available as `andriitrelin/cells-raman-spectra/*`

### GitHub
- [Raman Spectra Data](https://github.com/MIND-Lab/Raman-Spectra-Data)
- [Raman spectra of pathogenic bacteria](https://www.dropbox.com/scl/fo/fb29ihfnvishuxlnpgvhg/AJToUtts-vjYdwZGeqK4k-Y?rlkey=r4p070nsuei6qj3pjp13nwf6l&e=2&dl=0) 
(_more info on [this GitHub page](https://github.com/csho33/bacteria-ID)_)
- [High-throughput molecular imaging](https://github.com/conor-horgan/DeepeR?tab=readme-ov-file#dataset)
- [spectrai raman spectra](https://github.com/conor-horgan/spectrai)

### Zenodo
- [Quantitative volumetric Raman imaging](https://zenodo.org/records/256329)

### Other Sources
- [Spectra of illicit adulterants](https://data.mendeley.com/datasets/y4md8znppn/1)
- [Raman Spectrum Matching with Contrastive Representation Learning](https://data.dtu.dk/articles/dataset/Datasets_for_replicating_the_paper_Raman_Spectrum_Matching_with_Contrastive_Representation_Learning_/20222331?file=36144495)
- [Raman spectra of chemical compounds](https://springernature.figshare.com/articles/dataset/Open-source_Raman_spectra_of_chemical_compounds_for_active_pharmaceutical_ingredient_development/27931131)
- [Inline Raman Spectroscopy and Indirect Hard Modeling](https://publications.rwth-aachen.de/record/978266/files/)
- [The Effect of Sulfate Electrolytes on the Liquid-Liquid Equilibrium](https://publications.rwth-aachen.de/record/978265/files/)
- [In-line Monitoring of Microgel Synthesis](https://publications.rwth-aachen.de/record/834113/files/) (_weird format_)
- [N-isopropylacrylamide Microgel Synthesis](https://publications.rwth-aachen.de/record/959050/files/)
- [Nonlinear Manifold Learning Determines Microgel Size from Raman Spectroscopy](https://publications.rwth-aachen.de/record/959137)
- [NASA AHEAD](https://ahed.nasa.gov/datasets/f5b6051bfeb18c5a7eaef6504582)
- [RRUFF](https://rruff.info/)
