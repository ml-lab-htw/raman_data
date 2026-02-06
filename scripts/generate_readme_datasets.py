#!/usr/bin/env python3
"""
Generate the datasets table in README.md from loader registrations.

Usage:
    python scripts/generate_readme_datasets.py

This script imports loader modules under `raman_data.loaders` and reads their
`DATASETS` dicts. It generates a single markdown table with a `Source` column
(describing the loader) and replaces the README section between
<!-- DATASETS_TABLE_START --> and <!-- DATASETS_TABLE_END -->.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from raman_data.types import TASK_TYPE, APPLICATION_TYPE
except Exception as e:
    print("Failed to import raman_data.types:", e)
    raise

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
LOADERS_PKG = "raman_data.loaders"


def gather_datasets() -> List[Tuple[str, str, str, str]]:
    """Return a flat list of datasets as (name, application, task, description).
    """
    rows_all: List[Tuple[str, str, str, str]] = []
    try:
        pkg = importlib.import_module(LOADERS_PKG)
    except Exception as e:
        print(f"Failed to import {LOADERS_PKG}: {e}")
        raise

    for finder, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        full_name = f"{LOADERS_PKG}.{modname}"
        try:
            mod = importlib.import_module(full_name)
        except Exception as e:
            print(f"Warning: could not import loader module {full_name}: {e}")
            continue

        # Try module-level DATASETS first
        datasets = None
        try:
            candidate = getattr(mod, "DATASETS", None)
            if isinstance(candidate, dict) and candidate:
                datasets = candidate
        except Exception:
            datasets = None

        # Otherwise, scan for classes defined in the module that expose DATASETS
        if datasets is None:
            for obj in list(vars(mod).values()):
                try:
                    if isinstance(obj, type):
                        cand = getattr(obj, "DATASETS", None)
                        if isinstance(cand, dict) and cand:
                            datasets = cand
                            break
                except Exception:
                    continue

        if datasets is None:
            continue

        for ds_name, ds_info in datasets.items():
            # application type
            try:
                at = ds_info.application_type
                application = at.name if isinstance(at, APPLICATION_TYPE) else str(at)
            except Exception:
                application = "Unknown"
            # task type
            try:
                tt = ds_info.task_type
                task = tt.name if isinstance(tt, TASK_TYPE) else str(tt)
            except Exception:
                task = "Unknown"
            # description
            desc = ""
            try:
                meta = getattr(ds_info, "metadata", {}) or {}
                desc = meta.get("description", "") if isinstance(meta, dict) else str(meta)
            except Exception:
                desc = ""
            rows_all.append((ds_name, application, task, desc))

    return rows_all


def render_markdown(rows: List[Tuple[str, str, str, str]]) -> str:
    """Render a single markdown table with columns: Dataset Name | Application | Task Type | Description
    """
    parts: List[str] = []
    parts.append("<!-- AUTO-GENERATED: START - datasets table. Do not edit manually. -->")
    parts.append("")
    parts.append("| Dataset Name | Application | Task Type | Description |")
    parts.append("|--------------|-------------|-----------|-------------|")
    for name, application, task, desc in sorted(rows, key=lambda x: x[0].lower()):
        short = (desc.replace("\n", " ")[:300]).strip()
        short = short.replace("|", "\\|")
        parts.append(f"| `{name}` | {application} | {task} | {short} |")
    parts.append("")
    parts.append("<!-- AUTO-GENERATED: END - datasets table. -->")
    return "\n".join(parts)


def replace_readme_table(new_table: str, dataset_count: int | None = None) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    start_tag = "<!-- DATASETS_TABLE_START -->"
    end_tag = "<!-- DATASETS_TABLE_END -->"
    if start_tag not in text or end_tag not in text:
        print("README does not contain DATASETS_TABLE_START/END markers")
        raise SystemExit(2)
    new_text = re.sub(
        re.compile(re.escape(start_tag) + r".*?" + re.escape(end_tag), flags=re.S),
        start_tag + "\n" + new_table + "\n" + end_tag,
        text,
    )
    README_PATH.write_text(new_text, encoding="utf-8")
    count_msg = f" with {dataset_count} datasets" if dataset_count is not None else ""
    print(f"Updated {README_PATH}{count_msg}.")


def main():
    rows = gather_datasets()
    if not rows:
        print("No datasets discovered. Aborting.")
        raise SystemExit(1)
    md = render_markdown(rows)
    replace_readme_table(md, dataset_count=len(rows))


if __name__ == "__main__":
    main()
