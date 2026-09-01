# CLAUDE.md — raman_data

Guidance for Claude Code when working with this repository.

## Current State (as of v1.6.2 release)

**Status**: v1.6.2 released (Sept 1, 2026)
- Stable dataset loading package
- 74+ datasets across Chemical, Medical, Biological, Material Science domains
- Uses setuptools-scm for dynamic versioning (version is derived from git tags, not hardcoded)
- All CI tests passing (network-dependent tests skipped appropriately)

## What This Repo Is

**raman_data** is the unified Python dataset API for Raman spectroscopy data across Kaggle, HuggingFace, Zenodo, Figshare, and GitHub.

- **Public**: Published on GitHub (`github.com/ml-lab-htw/raman_data`) and PyPI (`pip install raman-data`)
- **Core function**: `raman_data(dataset_key)` returns a `RamanDataset` object with `.spectra`, `.raman_shifts`, `.targets`, `.metadata`
- **Coverage**: 74+ datasets, ~156 regression targets, curated metadata (task type, application domain, license, etc.)

## Architecture: Loaders → Unified API

Every dataset is loaded through a **loader** class:

| Loader | Datasets | Source |
|--------|----------|--------|
| `KaggleLoader` | 20+ chemical/biomedical datasets | Kaggle (public datasets) |
| `ZenodoLoader` | ~20 datasets | Zenodo (academic deposits) |
| `FigshareLoader` | 5+ datasets (ComFilE, ChEMBL, etc.) | Figshare (research data) |
| `GitHubLoader` | 5+ datasets (COVID-19 saliva, AIT glucose, etc.) | GitHub repositories |
| `MiscLoader` | Catch-all for smaller/legacy datasets | Various (SciKitLearn, local files, etc.) |

**Key flow**:
1. User calls `raman_data("dataset_key")`
2. Registry finds the dataset and its loader class
3. Loader's `_load(cache_path)` method downloads and parses the dataset
4. Returns `(spectra, raman_shifts, targets, class_names, group_ids)` tuple (group_ids is optional)
5. `RamanDataset` constructor wraps it; user accesses `.spectra`, `.raman_shifts`, etc.

## Key Files

| File | Purpose |
|------|---------|
| `raman_data/__init__.py` | Entry point; `raman_data()` function dispatches to loaders |
| `raman_data/loaders/` | Loader implementations (KaggleLoader, ZenodoLoader, etc.) |
| `raman_data/types.py` | `DatasetInfo` (metadata), `RamanDataset` (return type) |
| `raman_data/registry.py` | Dataset registry; maps `dataset_key` → loader + DatasetInfo |
| `raman_data/loaders/LoaderTools.py` | Shared utilities (download, cache, CSV parsing, etc.) |
| `pyproject.toml` | Dependencies, build config; version is dynamic (setuptools-scm) |
| `CHANGELOG.md` | Release notes (NEW as of v1.6.2) |

## Group IDs (Replicate Structure)

Some datasets have physical replicate measurements — e.g., multiple spectra from the same plant, animal, or lab sample.

**How to handle**:
1. If a dataset has explicit replicate structure, the loader should construct a `group_ids` array (numpy array of group labels)
2. This is passed to `RamanDataset(group_ids=group_ids)` 
3. When the dataset is used in RamanBench, group-aware splitting (StratifiedGroupKFold) prevents data leakage from replicates into train/test
4. When mirrored to HuggingFace, `group_ids` is serialized as a `_group_id` column in the Parquet file

**Current loaders that emit group_ids** (9 datasets as of v1.6.2):
- `wheat_lines` (ZenodoLoader)
- `locust_phase_hemolymph` (ZenodoLoader)
- `serum_prostate_cancer` (FigshareLoader)
- `serum_alzheimer_disease` (FigshareLoader)
- `comfile_stroke` (FigshareLoader)
- `covid19_salvia` (GitHubLoader)
- `parkinson` (GitHubLoader)
- `alzheimer` (GitHubLoader)
- `ait_glucose_blood_sers` (GitHubLoader)

If a dataset has `is_grouped=True` in its metadata but no actual `group_ids` array, that's a known gap (documented in the paper repo's memory).

## Versions & Releases

**Version numbering**: Semantic versioning. Uses setuptools-scm for dynamic versioning.

**Release process** (different from RamanBench because version is dynamic):
1. Create/update CHANGELOG.md (move "Unreleased" → dated release section)
2. Commit: `git commit -m "Release vX.Y.Z: <summary>"`
3. Tag: `git tag -a vX.Y.Z -m "<release notes>"`
4. Push: `git push origin main vX.Y.Z` (or create PR if main is protected)
5. Do NOT update pyproject.toml version field (setuptools-scm reads from git tag automatically)

**Current version**: 1.6.2 (released Sept 1, 2026)

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/ -v --cov=raman_data   # With coverage (CI form)
```

Tests that download from raw sources (Zenodo, Kaggle, etc., bypassing the HF mirror) carry `@pytest.mark.skip` or conditional `@pytest.mark.skipif` decorators to prevent flakiness from network timeouts. Examples:
- `tests/test_raman_data.py::test_load_dataset` — skipped (hits live Kaggle/HuggingFace/Zenodo)
- `tests/test_loaders.py::test_sugar_mixtures_low_snr` — skipped (hits live Zenodo)

## Common Tasks

### Adding a new dataset
1. Identify which loader it belongs to (Kaggle → `KaggleLoader`, Zenodo → `ZenodoLoader`, etc.)
2. In that loader's file, add a `_load_<name>` method that:
   - Downloads/parses the dataset
   - Returns `(spectra, raman_shifts, targets, class_names, group_ids)` tuple
   - `group_ids` can be None (ungrouped) or a numpy array of group labels
3. Add a `DatasetInfo(...)` entry in the loader's `DATASETS` dict, referencing your `_load_<name>` method
4. Include metadata: task_type, application_type, id, name, source, paper, description, etc.
5. If the dataset has replicate structure, set `is_grouped=True` and ensure the loader returns actual `group_ids`
6. Add test in `tests/test_loaders.py` (can be skipped if it requires download)
7. Update CHANGELOG.md under "Added"

### Debugging a dataset load
```python
from raman_data import raman_data
ds = raman_data("dataset_key", use_mirror=False)  # Raw source
print(ds.spectra.shape, ds.targets.shape, ds.group_ids is not None)
```

If it fails with a network error, the test should have `@pytest.mark.skip`. If it doesn't, add the decorator.

### Testing a new loader locally
```bash
python -m pytest tests/test_loaders.py::test_my_new_dataset -v -s
```

Use `-s` to see print statements and download progress.

## Important: Network Dependencies

**Zenodo, Kaggle, HuggingFace can be flaky**. CI tests that hit raw sources are **skipped by default**. Do not remove skip decorators or try to "fix" timeouts by retrying — that's a network issue, not a code bug.

If you add a test that downloads from raw sources:
```python
@pytest.mark.skip(reason="Hits live Zenodo/Kaggle/HuggingFace; run manually if needed")
def test_my_dataset(temp_cache):
    ...
```

Or conditionally skip in CI:
```python
@pytest.mark.skipif(os.environ.get("CI"), reason="Live downloads too flaky for CI")
def test_my_dataset(temp_cache):
    ...
```

## Datasets Structure (Example)

A typical dataset info entry:
```python
"dataset_key": DatasetInfo(
    task_type=TASK_TYPE.Classification,
    application_type=APPLICATION_TYPE.Medical,
    id="zenodo_12345",
    name="My Dataset",
    short_name="My Dataset",
    file_typ="*.csv",
    license="CC BY 4.0",
    loader=lambda cache_path: MyLoader._load_my_dataset(cache_path),
    metadata={
        "full_name": "Full descriptive name",
        "source": "https://zenodo.org/record/12345",
        "paper": "https://doi.org/10.1234/xyz",
        "description": "What this dataset contains and why it's useful for Raman spectroscopy",
    },
    is_grouped=True,  # If there are replicates (optional)
    has_missing_labels=False,  # If any target value is NaN (optional)
),
```

## Leaderboard & Mirror

**HuggingFace Mirror** (`HTW-KI-Werkstatt/RamanBench`):
- Datasets cached as Parquet files for reliability
- Built by `scripts/mirror_to_huggingface.py` (runs in raman_bench_paper, not here)
- If a dataset has `group_ids`, they're serialized as `_group_id` column in Parquet
- RamanBench's `_load_from_mirror()` reads them back

**You don't need to update the mirror manually** — that's the paper repo's job. Just ensure your dataset's loader correctly constructs and returns `group_ids` if it exists.

## Troubleshooting

**"Import fails with 'no module named raman_data'"**
→ Install: `pip install -e .` in this repo's root

**"A dataset download times out in CI"**
→ This is expected for raw-source tests. Ensure the test has `@pytest.mark.skip` or `@pytest.mark.skipif(os.environ.get("CI"))`.

**"How do I know if a dataset should have group_ids?"**
→ Look at the paper/documentation. If measurements are grouped (multiple spectra per sample/plant/animal), construct `group_ids`. Examples:
- `wheat_lines`: one plant line per group → factorize plant-line column
- `covid19_salvia`: one subject per group → extract from filename/directory structure
- If there's no inherent grouping (one spectrum = one independent sample), `group_ids = None`

**"The mirror is out of sync with my new dataset"**
→ Not your problem right now. The paper repo's CI will rebuild the mirror. Just ensure your loader works locally with raw sources.

## Links

- Public repo: https://github.com/ml-lab-htw/raman_data
- PyPI: https://pypi.org/project/raman-data/
- Paper: https://arxiv.org/abs/2605.02003
- Leaderboard: https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench
- RamanBench: https://github.com/ml-lab-htw/RamanBench
- Paper repo (mirror sync): https://github.com/ml-lab-htw/raman-bench-paper
