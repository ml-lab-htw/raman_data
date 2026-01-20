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
| `spectra` | `np.ndarray` | Raman spectra intensity data (2D: samples × wavenumbers) |
| `raman_shifts` | `np.ndarray` | Wavenumber/Raman shift values in cm⁻¹ (1D array) |
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

The dataset can also be converted to a pandas DataFrame:

```python
df = dataset.to_dataframe()
```

## 📚 Available Datasets

Here is the list of datasets that are currently included in the package:

### Kaggle Datasets
| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `codina/diabetes/AGEs` | Classification | Diabetes detection using AGEs spectroscopy |
| `codina/diabetes/earLobe` | Classification | Diabetes detection from ear lobe |
| `codina/diabetes/innerArm` | Classification | Diabetes detection from inner arm |
| `codina/diabetes/thumbNail` | Classification | Diabetes detection from thumbnail |
| `codina/diabetes/vein` | Classification | Diabetes detection from vein |
| `sergioalejandrod/AminoAcids/glycine` | Classification | Amino acid (glycine) spectroscopy |
| `sergioalejandrod/AminoAcids/leucine` | Classification | Amino acid (leucine) spectroscopy |
| `sergioalejandrod/AminoAcids/phenylalanine` | Classification | Amino acid (phenylalanine) spectroscopy |
| `sergioalejandrod/AminoAcids/tryptophan` | Classification | Amino acid (tryptophan) spectroscopy |
| `andriitrelin/cells-raman-spectra/COOH` | Classification | Cell type classification (COOH functionalized) |
| `andriitrelin/cells-raman-spectra/NH2` | Classification | Cell type classification (NH2 functionalized) |
| `andriitrelin/cells-raman-spectra/(COOH)2` | Classification | Cell type classification ((COOH)2 functionalized) |

**Sources:**
- [Diabetes Spectroscopy](https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes)
- [Amino Acids Spectroscopy](https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy)
- [Cells Raman Spectra](https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra)

### Hugging Face Datasets
| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `chlange/SubstrateMixRaman` | Regression | Substrate mixture concentration prediction |
| `chlange/RamanSpectraEcoliFermentation` | Regression | E. coli fermentation monitoring |
| `chlange/FuelRamanSpectraBenchtop` | Regression | Fuel property prediction |

**Sources:**
- [Substrate Mix Raman](https://huggingface.co/datasets/chlange/SubstrateMixRaman)
- [Ecoli Fermentation](https://huggingface.co/datasets/chlange/RamanSpectraEcoliFermentation)
- [Fuel Spectra Benchtop](https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop)

### Zenodo Datasets
| Dataset Name | Task Type | Description |
|-------------|-----------|-------------|
| `sugar mixtures` | Regression | Hyperspectral unmixing of sugar mixtures |
| `Wheat lines` | Classification | Wheat line differentiation |
| `Adenine` | Classification | SERS quantitative analysis interlaboratory study |

**Sources:**
- [Hyperspectral Unmixing](https://zenodo.org/records/10779223)
- [Mutant Wheat Lines](https://zenodo.org/records/7644521)
- [Surface Enhanced Spectroscopy](https://zenodo.org/records/3572359)

### Miscellaneous Datasets
| Dataset Name                | Task Type        | Description                                                      |
|----------------------------|------------------|------------------------------------------------------------------|
| `deepr_denoising`           | Denoising        | Raman spectral denoising dataset from DeepeR paper               |
| `deepr_super_resolution`    | SuperResolution  | Hyperspectral super-resolution dataset from DeepeR paper         |

**Sources:**
- [DeepeR: High-throughput molecular imaging via deep learning enabled Raman spectroscopy](https://github.com/conor-horgan/DeepeR?tab=readme-ov-file#dataset)
- [Denoising Dataset (OneDrive)](https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EqZaY-_FrGdImybIGuMCvb8Bo_YD1Bc9ATBxbLxdDIv0RA?e=5%3aHhLp91&fromShare=true&at=9)
- [Super-Resolution Dataset (OneDrive)](https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/EuIIZkQGtT5NgQcYO_SOzigB706Q8b0EddSLEDGUN22EbA?e=5%3axGyu4b&fromShare=true&at=9)

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
