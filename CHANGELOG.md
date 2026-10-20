# Changelog

All notable changes to raman_data are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- `raman_mpi`: RaMPI — Raman Maps for Plastics Identification (same paper and
  Borealis release as `raman_spi`, doi:10.1038/s41597-026-07103-8 /
  doi:10.5683/SP3/8UQQQN, CC-BY 4.0). Per-pixel spectra from 34 Raman maps
  (30 pristine two-material blends + 4 Great Lakes Seabin environmental maps),
  633 nm / 1,800 l/mm. The 16,598 of 33,119 pixels the authors manually
  assigned to one of 14 plastic types (the 16,521 `blank` pixels are
  excluded), resampled from each map's native 1,017-point non-uniform axis
  onto a common 710–1,827 cm⁻¹ / 1 cm⁻¹ grid. Raw spectra as exported from
  Renishaw WiRE (no preprocessing). Group ids are the 34 maps — pixels within
  a map are spatially correlated, but note the groups are *not* label-pure
  (mixed-material maps contain several classes by design).
- `raman_spi`: RaSPI — Raman Spectra for Plastics Identification (Hogan et al. 2026,
  *Scientific Data* 13:765, doi:10.1038/s41597-026-07103-8; data release
  doi:10.5683/SP3/8UQQQN, Borealis, CC-BY 4.0). 396 fully-labeled spectra of 14
  plastic types (of the 402 rows in the source file — 6 metadata rows lack a
  spectrum, 4 spectra lack metadata, and PE1a/PE1b are stored twice), on a
  uniform 225–3,949 cm⁻¹ / 1 cm⁻¹ grid, 226 spectra at 633 nm + 176 at 523 nm
  (Renishaw inVia). Pristine commercial plastics plus environmental microplastics
  from Great Lakes Seabin collections. The authors' pipeline (polynomial baseline
  subtraction, truncation, 1 cm⁻¹ interpolation, 0–1 normalization) is documented
  in `embedded_preprocessing`. 272 physical samples — the trailing letter in each
  spectrum id (PE1a, PE1b, PET4c …) marks repeated measurements of the same
  sample — exposed as `group_ids` (`is_grouped=True`).

### Fixed

- `mlrod` is CC BY-NC (the Open Data Repository states "BY-NC"), not CC BY 4.0; source link now
  https://odr.io/MLROD.
- Croissant licences: `generate_croissant.py` turned every licence string it did not recognise into
  CC BY 4.0, and every CC BY-NC 3.0 into 4.0. It now reads the Creative Commons variant and version,
  maps GPL-3.0, and keeps other stated terms ("research use only", "Free access", ...) as
  `conditionsOfAccess`; only datasets with no licence at all still get the assumed CC BY 4.0. Fixed in
  21 Croissant files: `mlrod`, the 14 `microgel_size_*` (CC BY-NC 3.0), `biomolecules_reference`
  (GPL-3.0), `rruff_mineral_*`, `synthetic_organic_pigments_*` and `chlorinated_samples`.
- Stale "CC BY 4.0 has been assumed" notes removed from 24 datasets whose licence is stated (CC BY 4.0,
  MIT). `alzheimer`, `parkinson` and `covid19_salvia` now keep their licence-clarification note on
  regeneration. The note stays only where no licence is known (`covid19_serum`, `organic_compounds_*`).

### Changed

---

## [1.6.6] — 2026-10-15

### Added

- `glioma_methylation` (6 classes, LGm1-6) and `glioma_idh` (IDH-mutant vs.
  IDH-wildtype): Raman maps of 59 glioma tissue slides from 46 patients
  (Lita et al. 2024, *Neuro-Oncology*, doi:10.1093/neuonc/noae101), from the
  authors' public Google Drive folder. 257,904 tumour spectra (the authors'
  DBSCAN tumour maps applied), 1,738 bands from 50 to 3,400 cm⁻¹, grouped by
  patient. Licence: none stated for the data; the paper is CC BY-NC.

### Fixed

- `GoogleDriveLoader.load_dataset` now passes `group_ids` through when a loader
  returns them.
- README dataset table: the two `synthetic_organic_pigments_*` datasets show as
  Classification (the 1.6.5 fix had not regenerated it).

---

## [1.6.5] — 2026-09-30

### Fixed

- `synthetic_organic_pigments_raw`/`_baseline_corrected` were registered as
  `Regression`, but the loader already produced classification-shaped output
  (`encode_labels()` label-encoding the per-sample pigment filename, e.g.
  `PB12_A_785`, essentially unique per sample). Under `Regression`, those
  integer codes were used as literal targets (just the row index in
  practice) and the class-name list was routed into a `target_names` slot
  meant for per-target-column names, not per-sample labels -- any benchmark
  run against either dataset was regressing against an arbitrary sample
  ordering, not a real quantity. Now registered as `Classification`,
  encoding the color-family grouping (the pigment code's leading letters,
  e.g. `PB`/`PBR`/`PY`) rather than the full per-sample code: PR=red (143),
  PY=yellow (86), PO=orange (37), PV=violet (26), PB=blue (19), PBR=brown
  (6), PG=green (6), PBK=black (2).

---

## [1.6.4] — 2026-09-22

### Added

- New dataset `marine_pathogens_binary`: 261 Raman spectra, binary
  classification between *Acinetobacter baumannii* (150 spectra) and
  *Pseudomonas nitritireducens* (111 spectra). Rehosted alongside
  `marine_pathogens` at github.com/zshicode/Raman-Spectra-Deep-Learning,
  which attributes it to the same Yu et al. (2021) paper as
  `marine_pathogens`. That attribution is recorded as given by the source
  repo but was **not** independently confirmed (the paper's own abstract
  only describes the 8-strain experiment covered by `marine_pathogens`) --
  see the citation-confidence note on the dataset's `DatasetInfo` entry.
  Licensed CC BY-NC-SA (non-commercial) via the same paper's Green-OA
  status.

### Changed

- Refactored the shared flat-CSV download/parse logic between
  `marine_pathogens` and `marine_pathogens_binary` into
  `MiscLoader._load_zshicode_raman_csv()`.

---

## [1.6.3] — 2026-09-22

### Added

- New dataset `marine_pathogens`: 1,138 Raman spectra of 8 bacterial/yeast
  strains isolated from the marine organism *Urechis unicinctus*
  (classification), from Yu et al. (2021), *Analytical Chemistry*
  93(32), 11089-11098 (doi:10.1021/acs.analchem.1c00431). Licensed
  CC BY-NC-SA (non-commercial) via the paper's own Green-OA status,
  since the rehosting GitHub repo itself carries no license.

### Fixed

- `scripts/generate_croissant.py`: license-string normalisation used a
  naive `.startswith("cc by")` prefix check that silently mapped
  restrictive `CC BY-NC-SA`/`CC BY-NC`/`CC BY-NC-ND` license strings to
  the permissive `CC BY 4.0` URL in generated Croissant metadata. Added
  explicit higher-priority mappings for these variants; regenerated the
  affected files (`chembl_molecules`, the `microgel_size_*` family) so
  they now carry their correct, more restrictive license URLs.

---

## [1.6.2] — 2026-09-01

### Fixed

- Skip live-network dataset downloads in CI tests to prevent flaky failures from
  external service timeouts (Zenodo, Kaggle, HuggingFace). Tests that bypass the
  HF mirror (e.g., `test_load_dataset`, `test_sugar_mixtures_low_snr`) now carry
  appropriate `pytest.mark.skip` or conditional `pytest.mark.skipif` decorators
  following the repo's existing convention.

### Changed

- README: rewritten introduction and contributing guidelines to improve
  accessibility for new contributors. Integrated docs-agent guidance for
  upstream-first documentation updates.

---

## [1.6.1] — 2026-08-15

Initial public release aligned with RamanBench v0.1.