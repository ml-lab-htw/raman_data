# raman-data

[![PyPI version](https://badge.fury.io/py/raman-data.svg)](https://pypi.org/project/raman-data/)
[![License](https://img.shields.io/github/license/ml-lab-htw/raman_data)](https://github.com/ml-lab-htw/raman_data/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/raman-data)](https://pypi.org/project/raman-data/)

A unified Python library for accessing public Raman spectroscopy datasets.
`raman-data` is the dataset layer of [RamanBench](https://github.com/ml-lab-htw/RamanBench), a large-scale benchmark for machine learning on Raman spectroscopy data.
It provides a single API to discover, download, and load **89 datasets** — covering classification, regression, denoising, and super-resolution tasks — from diverse sources (Kaggle, HuggingFace, Zenodo, Figshare, GitHub) in a standardized, ML-ready format.
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
| `acetic_acid_species` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. Includes acetic, citric, formic, itaconic, levulinic, oxalic, and succinic acids. Data for concentration monitoring and indirect hard modeling. |
| `adenine_colloidal_gold` | Chemical | Regression | Quantitative SERS spectra of adenine measured using colloidal gold substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_colloidal_silver` | Chemical | Regression | Quantitative SERS spectra of adenine measured using colloidal silver substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_solid_gold` | Chemical | Regression | Quantitative SERS spectra of adenine measured using solid gold substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `adenine_solid_silver` | Chemical | Regression | Quantitative SERS spectra of adenine measured using solid silver substrates across 15 different European laboratories. Benchmarks model reproducibility and inter-instrumental variability. |
| `alzheimer` | Medical & Clinical | Classification | Raman spectra from dried saliva drops targeting Alzheimer's Disease (PD) vs. healthy controls. Reveals hidden trends in proteins, lipids, and saccharides for early detection of cognitive and motor impairment. |
| `amino_acids_glycine` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Glycine elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `amino_acids_leucine` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Leucine elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `amino_acids_phenylalanine` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Phenylalanine elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `amino_acids_tryptophan` | Chemical | Regression | Time-resolved (on-line) Raman spectra for Tryptophan elution using a vertical flow LC-Raman method. Features 785 nm excitation and 0.2s exposure frames to benchmark label-free analyte detection. |
| `bacteria_identification` | Medical & Clinical | Classification | 60,000 spectra from 30 clinically relevant bacterial and yeast isolates (including an MRSA/MSSA isogenic pair). Acquired with 633 nm illumination on gold-coated silica substrates with low SNR to simulate rapid clinical acquisition times. |
| `biomolecules_reference` | Biological & Biotechnological | Classification | Reference Raman spectra (450–1800 cm⁻¹, 1 cm⁻¹ resolution) of ~140 pure biomolecules including amino acids, nucleotides, lipids, and sugars. Each spectrum is labelled by biomolecule name. Useful for spectral assignment and as a reference library for classification benchmarks. |
| `bioprocess_analytes_anton_532` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with an Anton Paar 532 nm spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_anton_785` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with an Anton Paar 785 nm spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_kaiser` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with a Kaiser spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_metrohm` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with a Metrohm spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_mettler_toledo` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with a Mettler Toledo spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_tec5` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with a Tec5 spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_timegate` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with a Timegate spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_analytes_tornado` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with a Tornado spectrometer. Part of an 8-spectrometer cross-instrument series. |
| `bioprocess_substrates` | Biological & Biotechnological | Regression | A benchmark dataset of 6,960 spectra featuring eight key metabolites (glucose, glycerol, acetate, etc.) sampled via a statistically independent uniform distribution. Designed to evaluate regression robustness against common bioprocess correlations. |
| `cancer_cell_(cooh)2` | Biological & Biotechnological | Classification | SERS spectra of cancer cell metabolites collected on gold nanourchins functionalized with the (COOH)2 moiety. Designed to provide specificity toward specific proteins and lipids for cell line identification. |
| `cancer_cell_cooh` | Biological & Biotechnological | Classification | SERS spectra of cancer cell metabolites collected on gold nanourchins functionalized with the COOH moiety. Designed to provide specificity toward specific proteins and lipids for cell line identification. |
| `cancer_cell_nh2` | Biological & Biotechnological | Classification | SERS spectra of cancer cell metabolites collected on gold nanourchins functionalized with the NH2 moiety. Designed to provide specificity toward specific proteins and lipids for cell line identification. |
| `chembl_molecules` | Chemical | Regression | 140k DFT-computed Raman spectra for ChEMBL drug-like molecules. Targets: HOMO-LUMO gap, HOMO/LUMO energies, isotropic polarizability, heat capacity, dipole moment. |
| `citric_acid_species` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. |
| `comfile_stroke` | Medical & Clinical | Classification | SERS serum spectra for binary stroke vs. healthy-control classification. ~4,020 spectra across 723 wavenumber points (202.985–1999.92 cm⁻¹). |
| `covid19_salvia` | Medical & Clinical | Classification | Non-invasive SARS-CoV-2 screening from dried saliva drops. ~25 spectral replicates per subject from 101 patients (positive, negative symptomatic, and healthy controls), 785 nm excitation. |
| `covid19_serum` | Medical & Clinical | Classification | Blood serum Raman spectra from 10 COVID-19 positive and 10 negative patients validated by RT-PCR and ELISA. |
| `deepr_denoising` | — | Denoising | Raman spectral denoising dataset from the DeepeR paper. Noisy input spectra paired with denoised targets. |
| `deepr_super_resolution` | — | SuperResolution | Hyperspectral super-resolution dataset from the DeepeR paper. Low-resolution inputs paired with high-resolution targets. |
| `diabetes_skin_ages` | Medical & Clinical | Classification | AGEs signatures in skin Raman spectra, acquired in vivo with a portable 785 nm spectrometer to distinguish diabetic patients from healthy controls. |
| `diabetes_skin_ear_lobe` | Medical & Clinical | Classification | Ear lobe skin Raman spectra for diabetic vs. healthy classification using a portable 785 nm spectrometer. |
| `diabetes_skin_inner_arm` | Medical & Clinical | Classification | Inner arm skin Raman spectra for diabetic vs. healthy classification using a portable 785 nm spectrometer. |
| `diabetes_skin_thumbnail` | Medical & Clinical | Classification | Thumbnail skin Raman spectra for diabetic vs. healthy classification using a portable 785 nm spectrometer. |
| `diabetes_skin_vein` | Medical & Clinical | Classification | Vein skin Raman spectra for diabetic vs. healthy classification using a portable 785 nm spectrometer. |
| `ecoli_fermentation` | Biological & Biotechnological | Regression | Batch and fed-batch E. coli fermentation spectra. Supernatant measurements with a 785 nm spectrometer tracking glucose and acetate concentrations. |
| `ecoli_metabolites` | Biological & Biotechnological | Regression | Raman spectra of glucose and sodium acetate mixtures measured with an automated liquid handling station for high-throughput E. coli fermentation monitoring. |
| `ecoli_metabolites_dig4bio` | Biological & Biotechnological | Regression | Raman spectra of glucose, sodium acetate, and magnesium sulfate mixtures measured with an automated high-throughput system for E. coli fermentation monitoring. |
| `flow_microgel_synthesis` | Chemical | Regression | In-line Raman spectra from a tubular flow reactor during microgel synthesis, paired with DLS-measured particle sizes. |
| `formic_acid_species` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. |
| `fuel_benchtop` | Chemical | Regression | Raman spectra from 179 commercial gasoline samples recorded with a benchtop 1064 nm FT-Raman system. Targets: RON, MON, and oxygenated additive concentrations. |
| `fuel_handheld` | Chemical | Regression | Same 179 gasoline samples as `fuel_benchtop`, acquired with a handheld 785 nm spectrometer. Benchmarks model transferability across hardware. |
| `hair_dyes_sers` | Chemical | Classification | SERS spectra of commercial hair dye products acquired with a portable Raman spectrometer. Target: brand identity. |
| `head_neck_cancer` | Medical & Clinical | Classification | Raman spectra of blood plasma and saliva from head and neck cancer patients and healthy controls. Target: cancer vs. control (binary). |
| `ht_raman_bio_catalysis_axp` | Biological & Biotechnological | Regression | Raman spectra for real-time monitoring of biocatalytic reactions in Deep Eutectic Solvents (DES). |
| `illicit_adulterants_ft_raman` | Medical & Clinical | Classification | FT-Raman spectra (1064 nm) of 11 pharmaceutically active adulterants in dietary supplements. Target: compound identity. |
| `illicit_adulterants_sers` | Medical & Clinical | Classification | SERS spectra (785 nm) of 11 illicit adulterants in dietary supplements. Target: compound identity. |
| `itaconic_acid_species` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. |
| `kaiser_ecoli_fermentation` | Biological & Biotechnological | Regression | E. coli fermentation Raman spectra collected with a Kaiser spectrometer. |
| `kaiser_ecoli_fermentation_supernatant` | Biological & Biotechnological | Regression | E. coli fermentation supernatant Raman spectra collected with a Kaiser spectrometer. |
| `levulinic_acid_species` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. |
| `microgel_size_lf_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: Linear Fit, range: fingerprint region. |
| `microgel_size_lf_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: Linear Fit, range: global. |
| `microgel_size_mm_lf_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: MinMax + Linear Fit, range: fingerprint region. |
| `microgel_size_mm_lf_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: MinMax + Linear Fit, range: global. |
| `microgel_size_mm_rb_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: MinMax + Rubber Band, range: fingerprint region. |
| `microgel_size_mm_rb_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: MinMax + Rubber Band, range: global. |
| `microgel_size_raw_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: Raw, range: fingerprint region. |
| `microgel_size_raw_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: Raw, range: global. |
| `microgel_size_rb_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: Rubber Band, range: fingerprint region. |
| `microgel_size_rb_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: Rubber Band, range: global. |
| `microgel_size_snv_lf_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: SNV + Linear Fit, range: fingerprint region. |
| `microgel_size_snv_lf_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: SNV + Linear Fit, range: global. |
| `microgel_size_snv_rb_fingerprint` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: SNV + Rubber Band, range: fingerprint region. |
| `microgel_size_snv_rb_global` | Chemical | Regression | Raman spectra of 235 microgel samples (208–483 nm diameter). Pretreatment: SNV + Rubber Band, range: global. |
| `microgel_synthesis` | Chemical | Regression | In-line Raman spectra monitoring microgel synthesis in a tubular flow reactor with a customized measurement cell. |
| `microplastics_weathered` | Material Science | Classification | Raman spectra of 167 virgin and UV-weathered microplastic particles across multiple polymer types (PE, PP, PS, PET, PVC, etc.). Target: polymer type. |
| `mlrod` | Material Science | Classification | 500,000+ Raman spectra of rock-forming silicate, carbonate, and sulfate minerals under Mars-like low-SNR conditions, without spectral preprocessing. |
| `organic_compounds_preprocess` | Chemical | Classification | Preprocessed Raman spectra of organic compounds from multiple excitation sources. Designed for transfer learning and domain adaptation benchmarks. |
| `organic_compounds_raw` | Chemical | Classification | Raw Raman spectra of organic compounds from multiple excitation sources. Designed for transfer learning and domain adaptation benchmarks. |
| `parkinson` | Medical & Clinical | Classification | Raman spectra from dried saliva drops for Parkinson's Disease vs. healthy control classification. |
| `pharmaceutical_ingredients` | Medical & Clinical | Classification | 3,510 Raman spectra from 32 chemical substances including organic solvents and API development reagents. Target: compound identity. |
| `ralstonia_fermentations` | Biological & Biotechnological | Regression | P(HB-co-HHx) copolymer synthesis monitoring in Ralstonia eutropha batch cultivations, combining experimental and high-fidelity synthetic data. |
| `rruff_mineral_preprocess` | Material Science | Classification | Preprocessed Raman spectra of 1,000+ mineral species from the RRUFF database, measured under varying conditions (532 nm and 785 nm). |
| `rruff_mineral_raw` | Material Science | Classification | Raw Raman spectra of 1,000+ mineral species from the RRUFF database, measured under varying conditions (532 nm and 785 nm). |
| `serum_alzheimer_disease` | Medical & Clinical | Classification | Serum SERS spectra for classifying Alzheimer's disease (AD), Mild Cognitive Impairment (MCI), and healthy controls. |
| `serum_prostate_cancer` | Medical & Clinical | Classification | Serum SERS spectra for classifying Prostate Cancer (PCa), Benign Prostatic Hyperplasia (BPH), and healthy controls. |
| `streptococcus_thermophilus_fermentation_kaiser` | Biological & Biotechnological | Regression | Offline Raman spectra of Streptococcus thermophilus batch cultivations using a Kaiser RXN1 spectrometer. Two 24-hour fermentation runs in shake flasks. |
| `streptococcus_thermophilus_fermentation_timegate` | Biological & Biotechnological | Regression | Offline Time-Gated Raman spectra of Streptococcus thermophilus batch cultivations. Two 24-hour fermentation runs in shake flasks. |
| `succinic_acid_species` | Chemical | Regression | Raman spectra and composition data for titration experiments of various acids in aqueous solution. |
| `sugar_mixtures_high_snr` | Chemical | Regression | High-SNR sugar mixture Raman spectra (7,680 measurements at 0.5 s integration) for benchmarking quantification algorithms. |
| `sugar_mixtures_low_snr` | Chemical | Regression | Low-SNR sugar mixture Raman spectra (7,680 measurements at 0.5 s integration) for evaluating noise robustness. |
| `synthetic_organic_pigments_baseline_corrected` | Material Science | Regression | Baseline-corrected spectral library of ~300 synthetic organic pigments for art conservation identification. |
| `synthetic_organic_pigments_raw` | Material Science | Regression | Raw spectral library of ~300 synthetic organic pigments for art conservation identification. |
| `tg_ecoli_fermentation` | Biological & Biotechnological | Regression | E. coli fermentation Raman spectra collected using Time-Gated Raman Spectroscopy. |
| `tg_ecoli_fermentation_supernatant` | Biological & Biotechnological | Regression | E. coli fermentation supernatant Raman spectra collected using Time-Gated Raman Spectroscopy. |
| `wheat_lines` | Biological & Biotechnological | Classification | Raman spectra of salt-stress-tolerant wheat mutant lines and commercial cultivars (785 nm). Target: mutant line vs. cultivar. |
| `yeast_fermentation` | Biological & Biotechnological | Regression | Raman spectra from continuous ethanolic fermentation of sucrose using Saccharomyces cerevisiae immobilized in calcium alginate beads. |

<!-- AUTO-GENERATED: END - datasets table. -->
<!-- DATASETS_TABLE_END -->

</details>

## Contributing a Dataset

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

Releases are automated via GitHub Actions. A version tag triggers the CI pipeline: tests run first, then the package is built and published to PyPI automatically.

```bash
# Ensure all changes are committed and pushed to main
git checkout main && git pull

# Tag and push (uses setuptools-scm for versioning)
git tag v1.2.3
git push origin v1.2.3
```

The tag must match `v*.*.*`. The [CI workflow](https://github.com/ml-lab-htw/raman_data/actions) will run tests across Python 3.10–3.13, build the package, and publish to [PyPI](https://pypi.org/project/raman-data/).

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
