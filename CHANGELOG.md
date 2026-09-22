# Changelog

All notable changes to raman_data are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

### Fixed

### Changed

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