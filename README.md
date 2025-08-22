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

# List all available datasets
print(raman_data())

# List only classification datasets
print(raman_data(task_type='classification'))

# Load a dataset
dataset = raman_data(name="diabetes_kaggle")

# Access the data, targets, and metadata
X = dataset.data
y = dataset.target
metadata = dataset.metadata

print(X.shape)
print(y.shape)
print(metadata)
```

## 📚 Available Datasets

Here is the list of datasets that will be included in the package:

### Kaggle
- [Raman Spectroscopy of Diabetes](https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes)
- [Raman Spectroscopy for detecting COVID-19](https://www.kaggle.com/datasets/sfran96/raman-spectroscopy-for-detecting-covid19)
- [Raman Spectroscopy](https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy)
- [Cells Raman Spectra](https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra)

### Hugging Face
- [SubstrateMixRaman](https://huggingface.co/datasets/chlange/SubstrateMixRaman)

### GitHub
- [Raman-Spectra-Data](https://github.com/MIND-Lab/Raman-Spectra-Data)

### Zenodo
- [Record 10779223](https://zenodo.org/records/10779223)

### Other Sources
- [DTU Dataset](https://data.dtu.dk/articles/dataset/Datasets_for_replicating_the_paper_Raman_Spectrum_Matching_with_Contrastive_Representation_Learning_/20222331)
- [Mendeley Data](https://data.mendeley.com/datasets/y4md8znppn/1)
- [Nature Article Data](https://www.nature.com/articles/s41467-019-12898-9)


## 🎯 Milestones

- [x] View Datasets
- [ ] Software architecture with dummy data
- [ ] Software tests
- [ ] Integration of Kaggle
- [ ] Integration of Huggingface
- [ ] Integration of Github
- [ ] Integration of Zenodo
- [ ] Integration of other datasets
- [ ] Finalize Package
    - [ ] Documentation
    - [ ] Publish to PyPi

## 🔮 For Later (Future Datasets)

- [NASA AHEAD](https://ahed.nasa.gov/datasets/f5b6051bfeb18c5a7eaef6504582)
- [RRUFF](https://rruff.info/)
