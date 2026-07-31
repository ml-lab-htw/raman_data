#!/usr/bin/env python
"""
Mirror raman_data datasets to HuggingFace Hub.

Downloads all raman_data datasets, converts them to wide-format Parquet
(one float column per wavenumber, target columns unspilt), then uploads
to a HuggingFace dataset repo.

Two-phase design:
1. Phase "download" — fetch datasets, serialise as Parquet locally
2. Phase "upload" — push Parquets to HuggingFace as separate configs
3. Phase "all" (default) — download then upload

Multi-target datasets stay together (target columns are analyte names,
not split into separate datasets). Single-target datasets get a "target"
column. Datasets with known physical-replicate structure (`group_ids` set
on the RamanDataset) get an explicit `_group_id` column so downstream
consumers (RamanBench's grouped train/test splitting) can identify
replicates without inferring them from target values.

Usage:
    python scripts/mirror_to_huggingface.py                      # all datasets, all phases
    python scripts/mirror_to_huggingface.py --dataset bioprocess_substrates
    python scripts/mirror_to_huggingface.py --phase download --output-dir ./cache
    python scripts/mirror_to_huggingface.py --phase upload --hf-repo my-org/my-data
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetInfo

from raman_data import raman_data, TASK_TYPE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _format_shift_col_name(shift: float) -> str:
    """Format raman shift value as column name.

    Removes trailing zeros and decimal point for clean names.
    E.g. 200.5000 -> "200.5", 200.0000 -> "200"
    """
    s = f"{shift:.4f}".rstrip("0").rstrip(".")
    return s


def _to_wide_format_dataframe(ds) -> tuple[pd.DataFrame, dict]:
    """Convert RamanDataset to wide-format DataFrame.

    Returns:
        (DataFrame, metadata_dict)
        - DataFrame: float32 wavenumber columns + optional `_group_id` column +
          target/analyte columns
        - metadata: {"target_names": [...], "task_type": int, "has_group_id": bool}
    """
    raman_shifts = ds.raman_shifts
    shift_col_names = [_format_shift_col_name(w) for w in raman_shifts]

    # Create DataFrame from spectra (float32 for efficiency)
    df = pd.DataFrame(ds.spectra.astype("float32"), columns=shift_col_names)

    has_group_id = ds.group_ids is not None
    if has_group_id:
        df["_group_id"] = ds.group_ids

    # Handle targets
    if ds.targets.ndim == 1:
        # Single-target: add as "target" column
        if ds.task_type == TASK_TYPE.Classification:
            df["target"] = ds.targets.astype(str)
        else:
            df["target"] = ds.targets.astype("float64")
    else:
        # Multi-target: add one column per target (named after analytes)
        for i, name in enumerate(ds.target_names):
            df[name] = ds.targets[:, i].astype("float64")

    # Ensure all metadata values are properly typed strings/ints
    target_names = [str(name) for name in ds.target_names] if ds.target_names else []
    task_type_value = int(ds.task_type.value)

    metadata = {
        "target_names": target_names,
        "task_type": task_type_value,
        "has_group_id": has_group_id,
    }
    return df, metadata


def _validate_metadata(dataset_id: str, metadata: dict) -> None:
    """Validate metadata format and types.

    Args:
        dataset_id: dataset identifier (for logging)
        metadata: metadata dict to validate

    Raises:
        ValueError: if metadata format is invalid
    """
    # Check required keys
    if "target_names" not in metadata:
        raise ValueError(f"{dataset_id}: missing 'target_names' in metadata")
    if "task_type" not in metadata:
        raise ValueError(f"{dataset_id}: missing 'task_type' in metadata")

    # Validate types
    if not isinstance(metadata["target_names"], list):
        raise ValueError(f"{dataset_id}: 'target_names' must be a list, got {type(metadata['target_names'])}")

    if not isinstance(metadata["task_type"], int):
        raise ValueError(f"{dataset_id}: 'task_type' must be an int, got {type(metadata['task_type'])}")

    # has_group_id is optional for backward compatibility with mirrors written
    # before group-id support existed; default to False when absent.
    if "has_group_id" in metadata and not isinstance(metadata["has_group_id"], bool):
        raise ValueError(f"{dataset_id}: 'has_group_id' must be a bool, got {type(metadata['has_group_id'])}")

    # Validate all target_names are strings
    for i, name in enumerate(metadata["target_names"]):
        if not isinstance(name, str):
            raise ValueError(f"{dataset_id}: target_names[{i}] must be str, got {type(name)}: {name}")

    # Validate task_type is valid enum value (0-4: Unknown, Classification, Regression, Denoising, SuperResolution)
    valid_task_types = {0, 1, 2, 3, 4}
    if metadata["task_type"] not in valid_task_types:
        raise ValueError(f"{dataset_id}: task_type must be in {valid_task_types}, got {metadata['task_type']}")

    logger.debug(f"  ✓ Metadata valid: target_names={metadata['target_names']}, task_type={metadata['task_type']}")


def _to_hf_dataset(ds) -> Dataset:
    """Convert RamanDataset to HuggingFace Dataset."""
    df, metadata = _to_wide_format_dataframe(ds)

    # Store metadata in dataset info description
    info = DatasetInfo(description=json.dumps(metadata))
    hf_ds = Dataset.from_pandas(df, info=info, preserve_index=False)
    return hf_ds


def download_phase(output_dir: str, cache_dir: str | None = None,
                   dataset_filter: str | None = None, retries: int = 2,
                   dry_run: bool = False):
    """Phase 1: Download all datasets and save as Parquet files.

    Args:
        output_dir: directory to save parquet files
        cache_dir: cache directory for raman_data loaders (None = defaults)
        dataset_filter: only mirror this one dataset (for testing)
        retries: number of retry attempts for failed downloads
        dry_run: if True, only show what would be done without saving
    """
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get all available datasets
    all_datasets = raman_data(cache_dir=cache_dir)
    datasets = [dataset_filter] if dataset_filter else all_datasets

    mode_str = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{mode_str}Mirroring {len(datasets)} dataset(s) to {output_dir}")

    failed = []
    for i, name in enumerate(datasets, start=1):
        logger.info(f"[{i}/{len(datasets)}] Downloading {name}...")

        # Skip if already downloaded
        parquet_path = output_dir / name / "train.parquet"
        if parquet_path.exists() and not dry_run:
            logger.info(f"  → Already cached, skipping")
            continue

        # Try downloading with retries
        attempt = 0
        success = False
        while attempt < retries and not success:
            attempt += 1
            try:
                ds = raman_data(name, cache_dir=cache_dir)
                if ds is None:
                    logger.warning(f"  → Failed to load {name}")
                    break

                # Convert to wide format
                df, metadata = _to_wide_format_dataframe(ds)
                logger.info(f"  → {df.shape[0]} spectra × {len(df.columns)} columns"
                            + (" (grouped)" if metadata["has_group_id"] else ""))

                # Validate metadata format
                _validate_metadata(name, metadata)

                if dry_run:
                    logger.info(f"  [DRY-RUN] Would save {len(df)} rows to {parquet_path}")
                    logger.info(f"  [DRY-RUN] Metadata: {json.dumps(metadata)}")
                else:
                    # Save parquet
                    dataset_dir = output_dir / name
                    dataset_dir.mkdir(exist_ok=True)
                    parquet_path = dataset_dir / "train.parquet"
                    df.to_parquet(parquet_path, index=False, engine="pyarrow")

                    # Save metadata separately for reference
                    meta_path = dataset_dir / "metadata.json"
                    with open(meta_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"  ✓ Saved to {parquet_path}")
                success = True
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"  ⚠ Attempt {attempt}/{retries} failed: {e} (retrying...)")
                else:
                    logger.error(f"  ✗ Error after {retries} attempt(s): {e}")
                    failed.append(name)

    logger.info(f"\nDownload phase complete: {len(datasets) - len(failed)}/{len(datasets)} succeeded")
    if failed:
        logger.warning(f"Failed datasets: {', '.join(failed)}")

    return output_dir


def upload_phase(output_dir: str, hf_repo: str = "HTW-KI-Werkstatt/RamanBench",
                 skip_existing: bool = False, dry_run: bool = False):
    """Phase 2: Upload Parquet files to HuggingFace as dataset configs.

    Args:
        output_dir: directory containing downloaded parquet files
        hf_repo: target HuggingFace repo ID (e.g. "user/repo")
        skip_existing: skip datasets that already exist in the repo (unused)
        dry_run: if True, only show what would be done without uploading
    """
    if not dry_run:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
    else:
        api = None

    output_dir = Path(output_dir)

    # Get all dataset subdirs
    dataset_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    mode_str = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{mode_str}Uploading {len(dataset_dirs)} dataset(s) to {hf_repo}")

    failed = []
    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        dataset_name = dataset_dir.name
        parquet_path = dataset_dir / "train.parquet"
        metadata_path = dataset_dir / "metadata.json"

        if not parquet_path.exists():
            logger.warning(f"[{i}/{len(dataset_dirs)}] {dataset_name}: no train.parquet, skipping")
            continue

        logger.info(f"[{i}/{len(dataset_dirs)}] Uploading {dataset_name}...")

        try:
            # Validate metadata before upload
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    _validate_metadata(dataset_name, metadata)

            if dry_run:
                logger.info(f"  [DRY-RUN] Would upload:")
                logger.info(f"    - {dataset_name}/data/train-00000-of-00001.parquet")
                if metadata_path.exists():
                    logger.info(f"    - {dataset_name}/metadata.json")
                logger.info(f"  [DRY-RUN] Commit message: Add {dataset_name} dataset")
            else:
                # Direct file upload - bypasses YAML validation that causes 413 errors
                operations = [
                    CommitOperationAdd(
                        path_in_repo=f"{dataset_name}/data/train-00000-of-00001.parquet",
                        path_or_fileobj=parquet_path
                    )
                ]

                # Add metadata.json if it exists
                if metadata_path.exists():
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=f"{dataset_name}/metadata.json",
                            path_or_fileobj=metadata_path
                        )
                    )

                # Upload both files in a single commit
                api.create_commit(
                    repo_id=hf_repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Add {dataset_name} dataset",
                )
                logger.info(f"  ✓ Uploaded {dataset_name}")

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            failed.append(dataset_name)

    # Upload comprehensive README.md to repo root
    logger.info("\nUploading README.md...")
    readme_path = Path(__file__).parent.parent / "README_HF_DATASET.md"
    if readme_path.exists():
        try:
            if dry_run:
                logger.info(f"  [DRY-RUN] Would upload README.md to {hf_repo}")
            else:
                operations = [
                    CommitOperationAdd(
                        path_in_repo="README.md",
                        path_or_fileobj=readme_path
                    )
                ]
                api.create_commit(
                    repo_id=hf_repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message="Update README with proper citation and mirror notice",
                )
                logger.info("✓ README.md uploaded")
        except Exception as e:
            logger.warning(f"Could not upload README: {e}")
    else:
        logger.warning(f"README_HF_DATASET.md not found at {readme_path}")

    logger.info(f"\nUpload phase complete: {len(dataset_dirs) - len(failed)}/{len(dataset_dirs)} datasets succeeded")
    if failed:
        logger.warning(f"Failed datasets: {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser(
        description="Mirror raman_data datasets to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-repo",
        default="HTW-KI-Werkstatt/RamanBench",
        help="Target HuggingFace dataset repo ID (default: HTW-KI-Werkstatt/RamanBench)",
    )
    parser.add_argument(
        "--output-dir",
        default="./mirror_output",
        help="Local directory for Parquet files (default: ./mirror_output)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for raman_data loaders (None = loader defaults)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Mirror only this one dataset (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip upload if config already exists in HF repo",
    )
    parser.add_argument(
        "--phase",
        choices=["download", "upload", "all"],
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retry attempts for failed downloads (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making actual changes",
    )

    args = parser.parse_args()

    if args.phase in ["download", "all"]:
        download_phase(args.output_dir, args.cache_dir, args.dataset,
                      retries=args.retries, dry_run=args.dry_run)

    if args.phase in ["upload", "all"]:
        upload_phase(args.output_dir, args.hf_repo, args.skip_existing,
                    dry_run=args.dry_run)

    logger.info("Done!")


if __name__ == "__main__":
    main()
