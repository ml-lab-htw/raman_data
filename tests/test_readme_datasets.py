import re
import pkgutil
import importlib
from pathlib import Path

import pytest

from raman_data.types import TASK_TYPE

README_PATH = Path(__file__).resolve().parents[1] / "README.md"

DATASET_TYPE_PATTERN = re.compile(r'^\|\s*`([^`]+)`\s*\|\s*(Classification|Regression|Denoising|SuperResolution)\s*\|')


def parse_readme_datasets(readme_path: Path):
    text = readme_path.read_text(encoding='utf-8')
    mapping = {}
    for line in text.splitlines():
        m = DATASET_TYPE_PATTERN.match(line)
        if m:
            name = m.group(1).strip()
            t = m.group(2).strip()
            mapping[name] = t
    return mapping


def gather_loaders_datasets(package_name="raman_data.loaders"):
    loaders_map = {}
    # iterate through submodules in raman_data.loaders
    for finder, name, ispkg in pkgutil.iter_modules(importlib.import_module(package_name).__path__):
        # import module
        full_name = f"{package_name}.{name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            # skip modules that fail to import (tests shouldn't break on optional deps)
            continue

        # Prefer module-level DATASETS if present and non-empty
        datasets = None
        try:
            candidate = getattr(mod, 'DATASETS', None)
            if isinstance(candidate, dict) and candidate:
                datasets = candidate
        except Exception:
            datasets = None

        # Otherwise inspect module attributes for classes that define DATASETS
        if datasets is None:
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)
                except Exception:
                    continue
                if isinstance(attr, type):
                    cand = getattr(attr, 'DATASETS', None)
                    if isinstance(cand, dict) and cand:
                        datasets = cand
                        break

        if not datasets:
            continue

        for ds_name, ds_info in datasets.items():
            try:
                tt = ds_info.task_type
                if isinstance(tt, TASK_TYPE):
                    loaders_map[ds_name] = tt.name
                else:
                    loaders_map[ds_name] = str(tt)
            except Exception:
                loaders_map[ds_name] = str(getattr(ds_info, 'task_type', 'Unknown'))

    return loaders_map


def test_readme_task_types_match_loaders():
    """Ensure datasets listed in README have the same TASK_TYPE as declared in the loaders."""
    readme_map = parse_readme_datasets(README_PATH)
    loaders_map = gather_loaders_datasets()

    missing = []
    mismatches = []

    for name, readme_type in readme_map.items():
        if name not in loaders_map:
            missing.append(name)
        else:
            loader_type = loaders_map[name]
            if loader_type.lower() != readme_type.lower():
                mismatches.append((name, readme_type, loader_type))

    msgs = []
    if missing:
        msgs.append("Datasets present in README but not implemented in loaders: " + ", ".join(missing))
    if mismatches:
        msgs.append("Task type mismatches (README -> loaders):")
        for name, rd, ld in mismatches:
            msgs.append(f"  - {name}: README={rd}  loaders={ld}")

    if msgs:
        pytest.fail("\n".join(msgs))
