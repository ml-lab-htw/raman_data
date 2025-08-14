# raman_data

A unified Python package for loading and accessing Raman spectroscopy datasets from various sources. This project aims to support research in building Foundation Models for Raman spectroscopy by providing a standardized API to access real-world datasets.

## Project Goals

- Provide a Python package (ideally published on PyPI) for easy access to Raman spectroscopy datasets.
- Download datasets from their original sources (Huggingface, Kaggle, GitHub, Zenodo, etc.) on first use.
- Cache datasets locally in user-defined folders for efficient reuse.
- Offer a unified data interface, regardless of the original dataset format.
- List available datasets, with filtering options (e.g., classification vs. regression tasks).
- Facilitate reproducible research and accelerate progress in the Raman community.

## Supported Datasets

### Kaggle
- [Raman Spectroscopy of Diabetes](https://www.kaggle.com/datasets/codina/raman-spectroscopy-of-diabetes)
- [Raman Spectroscopy for Detecting COVID19](https://www.kaggle.com/datasets/sfran96/raman-spectroscopy-for-detecting-covid19)
- [Raman Spectroscopy](https://www.kaggle.com/datasets/sergioalejandrod/raman-spectroscopy)
- [Cells Raman Spectra](https://www.kaggle.com/datasets/andriitrelin/cells-raman-spectra)

### Huggingface
- [SubstrateMixRaman](https://huggingface.co/datasets/chlange/SubstrateMixRaman)

### GitHub
- [MIND-Lab Raman Spectra Data](https://github.com/MIND-Lab/Raman-Spectra-Data)

### Zenodo
- [Raman Spectra Dataset](https://zenodo.org/records/10779223)

### Other Sources
- [DTU Raman Spectrum Matching Dataset](https://data.dtu.dk/articles/dataset/Datasets_for_replicating_the_paper_Raman_Spectrum_Matching_with_Contrastive_Representation_Learning_/20222331)
- [Mendeley Raman Dataset](https://data.mendeley.com/datasets/y4md8znppn/1)
- [Nature Communications Dataset](https://www.nature.com/articles/s41467-019-12898-9)

#### For Future Integration
- [NASA AHED Raman Dataset](https://ahed.nasa.gov/datasets/f5b6051bfeb18c5a7eaef6504582)
- [RRUFF Raman Database](https://rruff.info/)

## Milestones

Here are the key milestones for the development of the raman_data package:

- ✅ **Review and curate datasets**
- 🛠️ Design software architecture with dummy data
- 🧪 Implement software tests
- 📦 Integrate Kaggle datasets
- 🤗 Integrate Huggingface datasets
- 🗂️ Integrate GitHub datasets
- 🗃️ Integrate Zenodo datasets
- 🌐 Integrate other datasets
- 🚀 Finalize package
- 📚 Write documentation
- 🎉 Publish to PyPI

## Contributing

We welcome contributions from the Raman spectroscopy and machine learning communities. Please open issues or pull requests for suggestions, bug reports, or new dataset integrations.
