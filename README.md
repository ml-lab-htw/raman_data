# Raman-Data: A Unified Python Library for Raman Spectroscopy Datasets

This project aims to create a unified Python package for accessing various Raman spectroscopy datasets. The goal is to provide a simple and consistent API to load data from different sources like Kaggle, Hugging Face, GitHub, and Zenodo. This will be beneficial for the Raman spectroscopy community, enabling easier evaluation of models, such as foundation models for Raman spectroscopy.

## ✨ Features

- A single, easy-to-use Python package (planned for PyPI).
- Automatic downloading and caching of datasets from their original sources.
- A unified data format for all datasets.
- A simple function to list available datasets, with filtering options.

## 🚀 Getting Started

The basic interface for the package is defined in `raman_data/__init__.py`. Here's a preview of how it will work:

```python
from raman_data import raman_data
# To specify a task type import this enum as well
from raman_data import TASK_TYPE

# List all available datasets
print(raman_data())

# List only classification datasets
print(raman_data(task_type=TASK_TYPE.Classification))

# Load a dataset
dataset = raman_data(name="codina/diabetes/AGEs")

# Access the data, targets, and metadata
X = dataset.data
y = dataset.target
metadata = dataset.metadata

print(X.shape)
print(y.shape)
print(metadata)
```

For more detailed examples see [Demo Notebook](./demo.ipynb).

## 📚 Available Datasets

Here is the list of datasets that are currently included in the package:

### Kaggle
- [Diabetes Spectroscopy](https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes)
- [Liquid Chromatography](https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy)

### Hugging Face
- [Substrate Mix Raman](https://huggingface.co/datasets/chlange/SubstrateMixRaman)
- [Ecoli Fermentation](https://huggingface.co/datasets/chlange/RamanSpectraEcoliFermentation)
- [Fuel Spectra Benchtop](https://huggingface.co/datasets/chlange/FuelRamanSpectraBenchtop)

### Zenodo
- [Hyperspectral Unmixing](https://zenodo.org/records/10779223)
- [Mutant Wheat Lines](https://zenodo.org/records/7644521)
- [Surface Enhanced Spectroscopy for quantitative analysis](https://zenodo.org/records/3572359)

## 🎯 Milestones

- [x] View Datasets
- [x] Software architecture with dummy data
- [x] Software tests
- [x] Integration of Kaggle
- [x] Integration of Huggingface
- [x] Integration of Github
- [x] Integration of Zenodo
- [ ] Integration of other datasets
- [ ] Finalize Package
    - [ ] Documentation
    - [ ] Publish to PyPi

## 🔮 For Later (Future Datasets)

### Kaggle
- [Cancer Cells SERS Spectra](https://www.kaggle.com/code/mathiascharconnet/cancer-cells-sers-spectra) (requires authentification)

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
