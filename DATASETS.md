# RamanBench Dataset Collection

RamanBench is a curated collection of publicly available Raman spectroscopy datasets, assembled to provide a rigorous and comprehensive benchmark for machine learning models. The collection spans a wide range of spectral resolutions, excitation wavelengths (532 nm to 1064 nm), and experimental substrates, covering both classification and regression tasks across four application domains.

All datasets are provided in a consistent, ML-ready format via the `raman-data` Python library while preserving the unique noise profiles and artifacts characteristic of their respective application domains.

## Collection at a Glance

| | |
|---|---|
| **Total datasets** | 75 |
| **Total benchmark tasks** | 166 |
| **Total spectra** | ~325,000 |
| **Domains** | 4 |
| **Task types** | Classification, Regression |
| **New datasets (released with RamanBench)** | 13 |

## Inclusion Criteria

To be included in RamanBench, a dataset must be:

1. **Publicly accessible** — no access restrictions or upon-request-only policies.
2. **Experimentally acquired** — real measured Raman spectra, not simulated.
3. **Labelled** — class labels for classification tasks, or continuous regression targets.

Additional filters applied:

- **Minimum size:** At least 10 labelled spectra per dataset. For classification, classes with fewer than 9 spectra are removed; if fewer than 2 classes remain, the dataset is excluded.
- **Learnability:** Each regression target must achieve R² > 0 with at least one model; each classification dataset must exceed the majority-class baseline by ΔF1 > 0.05.

Tiny datasets (fewer than 50 spectra) are intentionally included, as small sample sizes are common in experimental Raman spectroscopy.

## Domains

### Material Science

| Dataset | Task | Datasets | Targets | Samples | Features | Range (cm⁻¹) |
|---|---|---|---|---|---|---|
| ML Raman Open Dataset (MLROD) | Classification | 1 | 1 | 130,061 | 1,836 | 141–1100 |
| RRUFF Minerals (Raw) | Classification | 1 | 1 | 1,162 | 1,142 | 303–853 |
| Synthetic Organic Pigments (Raw) | Regression | 1 | 1 | 325 | 561 | 1189–1651 |
| Weathered Microplastics | Classification | 1 | 1 | 77 | 1,144 | 202–3498 |

Material Science accounts for 48% of all spectra in the collection, driven by the large RRUFF mineral library and MLROD. Classification difficulty ranges from binary polymer identification (Microplastics) to fine-grained mineral identification with 79 classes (RRUFF Raw).

### Biological & Biotechnological

| Dataset | Task | Datasets | Targets | Samples | Features | Range (cm⁻¹) |
|---|---|---|---|---|---|---|
| Bio-Catalysis Monitoring of AXP ★ | Regression | 1 | 4 | 344 | 2,048 | −32–3385 |
| Bioprocess Analytes | Regression | 8 | 24 | 2,261 | 1,601 | 300–3500 |
| Bioprocess Monitoring | Regression | 1 | 8 | 6,960 | 1,870 | 391–3385 |
| Cancer Cell | Classification | 3 | 3 | 1,892 | 2,090 | 100–4278 |
| E. coli Fermentation | Regression | 1 | 2 | 379 | 1,870 | 391–3385 |
| Ecoli Metabolites ★ | Regression | 2 | 5 | 2,304 | 594 | 402–1599 |
| Kaiser Ecoli ★ | Regression | 2 | 8 | 28 | 1,699 | 301–1999 |
| Mutant Wheat | Classification | 1 | 1 | 53,134 | 1,748 | 296–2043 |
| R. eutropha Copolymer Fermentations ★ | Regression | 1 | 6 | 82 | 2,776 | 405–3180 |
| Streptococcus thermophilus Fermentations Kaiser ★ | Regression | 1 | 4 | 14 | 1,501 | 300–1800 |
| Tg Ecoli ★ | Regression | 2 | 8 | 25 | 114 | 604–1508 |
| Yeast Fermentation ★ | Regression | 1 | 4 | 58 | 1,900 | 401–2300 |

The Biological & Biotechnological domain contributes 20% of total spectra (67,070). A highlight is the Bioprocess Analytes collection, which uniquely provides the same analytes measured across eight different spectrometer models — enabling cross-instrument generalization studies.

### Medical & Clinical

| Dataset | Task | Datasets | Targets | Samples | Features | Range (cm⁻¹) |
|---|---|---|---|---|---|---|
| Alzheimer's SERS Serum | Classification | 1 | 1 | 3,417 | 724 | 0–723 |
| Diabetes Skin | Classification | 4 | 4 | 80 | 3,160 | 0–3159 |
| Head & Neck Cancer | Classification | 1 | 1 | 111 | 1,004 | 789–910 |
| Pathogenic Bacteria | Classification | 1 | 1 | 78,500 | 1,000 | 382–1792 |
| Pharmaceutical Ingredients | Classification | 1 | 1 | 3,510 | 3,276 | 150–3425 |
| Prostate Cancer SERS Serum | Classification | 1 | 1 | 12,601 | 725 | 0–724 |
| Saliva Alzheimer | Classification | 1 | 1 | 1,151 | 885 | 401–1598 |
| Saliva COVID-19 | Classification | 1 | 1 | 2,501 | 885 | 401–1598 |
| Saliva Parkinson | Classification | 1 | 1 | 1,476 | 885 | 401–1598 |
| Stroke SERS Serum | Classification | 1 | 1 | 4,020 | 724 | 200–2000 |

The Medical & Clinical domain contributes 26% of total spectra (87,284). All datasets are classification tasks, including SERS serum diagnostics for neurodegenerative diseases (Alzheimer's, Parkinson's), cancer screening (head & neck, prostate), and non-invasive diagnostics from saliva and skin spectra.

### Chemical & Industrial

| Dataset | Task | Datasets | Targets | Samples | Features | Range (cm⁻¹) |
|---|---|---|---|---|---|---|
| Acetic Concentration | Regression | 1 | 2 | 42 | 11,084 | 100–3425 |
| Adenine Colloidal ★ | Regression | 2 | 2 | 855 | 534 | 400–1999 |
| Adenine Solid ★ | Regression | 2 | 2 | 2,661 | 534 | 400–1999 |
| Amino Acids | Regression | 3 | 3 | 270 | 1,024 | 326–2035 |
| Citric Concentration | Regression | 1 | 2 | 45 | 11,084 | 100–3425 |
| Formic Concentration | Regression | 1 | 3 | 24 | 11,084 | 100–3425 |
| Gasoline Properties (Benchtop) ★ | Regression | 1 | 12 | 179 | 961 | 98–3801 |
| Gasoline Properties (Handheld) ★ | Regression | 1 | 12 | 179 | 1,901 | 400–2300 |
| Hair Dyes SERS | Classification | 1 | 1 | 1,713 | 1,340 | 309–1952 |
| Itaconic Concentration | Regression | 1 | 3 | 21 | 11,689 | −37–3470 |
| Levulinic Concentration | Regression | 1 | 2 | 36 | 11,084 | 100–3425 |
| Microgel Size | Regression | 14 | 14 | 3,290 | 3,500 | 800–1850 |
| Microgel Synthesis Flow vs. Batch | Regression | 1 | 1 | 14 | 11,084 | 100–3425 |
| Microgel Synthesis in Flow | Regression | 1 | 1 | 86 | 11,084 | 100–3425 |
| Succinic Concentration | Regression | 1 | 2 | 70 | 11,567 | −20–3450 |
| Sugar Mixtures | Regression | 2 | 10 | 9,800 | 2,000 | 142–3685 |

The Chemical & Industrial domain has the most datasets (35 of 75) but the fewest spectra (7%, ~23,317). It is dominated by regression tasks and includes several ultra-high-dimensional datasets (up to 11,689 wavenumber points) from acid concentration monitoring experiments.

★ = dataset released for the first time with this paper.

## Diversity Highlights

### Scale Diversity

Dataset sizes span more than four orders of magnitude — from 14 spectra (Microgel Synthesis Flow vs. Batch) to 130,061 spectra (MLROD) — with a median of ~235 spectra. This reflects the typical data scarcity of labeled experimental spectroscopy data.

- Classification datasets account for 91% of total spectra despite being fewer in number (21 of 75).
- Regression datasets are more numerous (54 of 75) but small: 81% have fewer than 500 samples.

### Spectral Diversity

- **Raman shift coverage:** 462 to 4,178 cm⁻¹ across the collection.
- **Feature dimensionality:** 114 to 11,689 wavenumber points (median 2,090).
- Features routinely outnumber training samples — most extremely in the Microgel Size datasets (11,689 points, ~235 samples).

### Task Complexity

- Classification ranges from binary screening (Diabetes Skin, COVID-19) to 79-class mineral identification (RRUFF Raw).
- Regression includes 32 multi-target datasets with up to 12 simultaneous prediction targets (Gasoline Properties), yielding 166 distinct benchmark tasks in total.

### Instrument Diversity

- Excitation wavelengths: 532 nm, 785 nm, and 1064 nm across benchtop, portable, and in-line process instruments.
- The Bioprocess Analytes collection provides the same analytes measured across nine different instruments — a unique resource for cross-instrument transfer learning.
- 13 datasets are released for the first time with RamanBench; the remaining 62 originate from HuggingFace, Kaggle, Zenodo, Figshare, and GitHub.
