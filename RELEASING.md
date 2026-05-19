# Releasing a new `raman-data` version

Releases are fully automated. Pushing a git tag matching `v*.*.*` triggers
`.github/workflows/ci.yml`: it runs tests on Python 3.10–3.13, builds the
package (version comes from `setuptools-scm`), and publishes to PyPI via
trusted publishing.

This file is the human checklist around that automation.

## 1. Check the current version

```bash
git fetch --tags
git tag --sort=-creatordate | head -5
```

Pick the next version using semver: patch for new datasets or doc fixes,
minor for new loader features, major for breaking API changes.

## 2. Update the README datasets table

The table and the dataset count line in `README.md` are auto-generated from
the loader registrations. **The script imports `raman_data`, so you must
point Python at the local source** — otherwise it silently regenerates from
the *installed* PyPI version and your new datasets won't show up.

```bash
# from the repo root
rm -rf raman_data/__pycache__ raman_data/loaders/__pycache__
PYTHONPATH=. python3 scripts/generate_readme_datasets.py
```

Expected output:

```
Updated /path/to/raman_data/README.md with <N> datasets.
```

If `<N>` doesn't include your new dataset, the import is resolving against
the installed package. Re-run with `PYTHONPATH=.` set.

## 3. Smoke-test new loaders

For every new dataset added since the last release, verify a clean download:

```bash
PYTHONPATH=. python3 -c "
from raman_data.datasets import load_dataset
ds = load_dataset('<your_dataset_id>', cache_dir='/tmp/raman_data_release_check')
print(ds.spectra.shape, ds.target_names)
"
```

(Use a throwaway `cache_dir` so you confirm the download path works, not a
warm cache.)

## 4. Run the test suite

```bash
PYTHONPATH=. pytest -x
```

CI runs the same on tag push, but failing locally is much faster to debug.

## 5. Commit and push to main

```bash
git status                 # review changed loaders, README, etc.
git add -A                 # or specific files
git commit -m "Add <dataset_id> dataset"
git push origin main
```

## 6. Tag and push

```bash
git tag v1.2.7             # match the version you picked in step 1
git push origin v1.2.7
```

The tag **must** match `v*.*.*` to trigger publish.

## 7. Watch CI and verify on PyPI

- Actions: https://github.com/ml-lab-htw/raman_data/actions
- PyPI:    https://pypi.org/project/raman-data/

Once green, smoke-test the published wheel in a fresh venv:

```bash
python3 -m venv /tmp/raman_data_pypi_check
/tmp/raman_data_pypi_check/bin/pip install -U raman-data==1.2.7
/tmp/raman_data_pypi_check/bin/python -c "
from raman_data.datasets import list_datasets
print(len(list_datasets()), 'datasets available')
"
```

## 8. Downstream nudges

Once the release is on PyPI:

- Update the version pin in `RamanBench`'s `pyproject.toml` if applicable.
- Cluster sbatch scripts do `git pull` on `raman_data` per task, so they
  already pick up the new loader from `main` — they don't depend on PyPI.
  Only relevant if a benchmark consumer pins by version.

---

## Common gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| Generator prints old dataset count | Importing installed `raman-data`, not local source | `PYTHONPATH=. python3 scripts/...` |
| CI publish step skipped | Tag doesn't match `v*.*.*` | Delete tag, re-tag with correct format |
| PyPI publish fails on "version exists" | Already released this version | Bump and retag — never delete a PyPI release |
| `setuptools-scm` reports `0.0.0` | Run from outside a git checkout, or shallow clone with no tags | `git fetch --tags --unshallow` |