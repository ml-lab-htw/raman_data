"""
General functions and enums meant to be used while loading certain dataset.
"""
from typing import Optional, List, Dict, Tuple

from tqdm import tqdm
import requests, zipfile

from scipy import io
import os, h5py
import numpy as np
import logging
import hashlib
import random
import time

from filelock import FileLock

from raman_data.exceptions import ChecksumError, CorruptedZipFileError
from raman_data.types import CACHE_DIR, HASH_TYPE, DatasetInfo


class LoaderTools:
    """
    A static class contains general methods that
    can be used while loading datasets.
    """
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_cache_root(
        env_var: CACHE_DIR
    ) -> str | None:
        """
        Retrieves the cache path of a certain loader.

        Args:
            env_var (CACHE_DIR): The name of loader's environment variable.

        Returns:
            str|None: The saved cache path or
                      None, if the path wasn't specified earlier.
        """
        try:
            return os.environ[env_var.value]
        except KeyError:
            return None


    @staticmethod
    def set_cache_root(
        path: str,
        loader_key: Optional[CACHE_DIR] = None
    ) -> None:
        """
        Sets the given path as the cache directory either for a specific
        or for all loaders.

        Args:
            path (str): The path to save datasets to or
                        "default" to reset previously saved path.
            loader_key (CACHE_DIR, optional): The name of loader's
                                              environment variable that stores
                                              the cache path. If None, sets
                                              the given path for all loaders.
        """
        path = None if path == "default" else path

        if loader_key is not None:
            os.environ[loader_key.value] = path
            LoaderTools.logger.debug(
                f"Cache root folder for {loader_key.name}'s loader is set to: {path}"
            )
            return

        for env_var in CACHE_DIR:
            os.environ[env_var.value] = path
        LoaderTools.logger.debug(f"Cache root folder is set to: {path}")


    @staticmethod
    def is_dataset_available(
        dataset_name: str,
        datasets: Dict[str, DatasetInfo]
    ) -> bool:
        """
        Checks whether given dataset's name is in the given list.

        Args:
            dataset_name (str): The name of a dataset to look for.
            datasets (List[str]): The list of datasets to look among
                                  (typically the list of a loader itself).

        Returns:
            bool: True, if the dataset is on the list. False otherwise.
        """
        check = dataset_name in datasets
        if not check:
            LoaderTools.logger.warning(
                f"Dataset {dataset_name} is not on the loader's list."
            )

        return check

    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    _MAX_RETRIES = 5
    _MAX_BACKOFF_SECONDS = 30

    @staticmethod
    def _is_file_ready(path: str) -> bool:
        """A destination file counts as "already there" if it exists and,
        for a `.zip`, actually looks like one (see ``is_valid_zip``)."""
        if not os.path.exists(path):
            return False
        return (not path.lower().endswith(".zip")) or LoaderTools.is_valid_zip(path)

    @staticmethod
    def _get_with_retry(url: str, headers: dict, timeout: int) -> requests.Response:
        """``requests.get`` with retry/backoff for transient failures.

        Handles rate-limiting (HTTP 429, respecting a server ``Retry-After``
        header when present) and other transient 5xx/connection errors with
        exponential backoff + jitter. Raises the underlying
        ``requests.exceptions.RequestException`` (e.g. via
        ``response.raise_for_status()``) once retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(LoaderTools._MAX_RETRIES):
            try:
                response = requests.get(
                    url=url, headers=headers, stream=True, allow_redirects=True, timeout=timeout,
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt == LoaderTools._MAX_RETRIES - 1:
                    raise
                delay = min(2 ** attempt, LoaderTools._MAX_BACKOFF_SECONDS) + random.uniform(0, 1)
                LoaderTools.logger.warning(
                    f"Request to {url} failed ({exc}); retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{LoaderTools._MAX_RETRIES})."
                )
                time.sleep(delay)
                continue

            if response.status_code in LoaderTools._RETRYABLE_STATUS_CODES \
                    and attempt < LoaderTools._MAX_RETRIES - 1:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else None
                except ValueError:
                    delay = None
                if delay is None:
                    delay = min(2 ** attempt, LoaderTools._MAX_BACKOFF_SECONDS)
                delay += random.uniform(0, 1)
                LoaderTools.logger.warning(
                    f"{url} returned HTTP {response.status_code}; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{LoaderTools._MAX_RETRIES})."
                )
                response.close()
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response

        # Unreachable in practice (the loop always returns or raises), but
        # keeps type-checkers happy and gives a clear error if it ever isn't.
        raise last_exc or RuntimeError(f"Failed to fetch {url} after {LoaderTools._MAX_RETRIES} attempts")

    @staticmethod
    def download(
            url: str,
            out_dir_path: str,
            out_file_name: Optional[str] = None,
            hash_target: Optional[str] = None,
            hash_type: Optional[HASH_TYPE] = None,
            referer: Optional[str] = None
    ) -> str | None:
        """
        Download files from a URL with optional hash verification
        and stores them as a `.zip` file.

        Args:
            url (str): The URL to download the files from.
            out_dir_path (str): The full path of the directory where
                                the downloaded files will be saved.
            out_file_name (str, optional): The name of the file to create.
                                           If None, it will be inferred from the Content-Disposition header.
            hash_target (str, optional): Expected hash value of the file for
                                         integrity verification.
            hash_type (HASH_TYPE, optional): The type of provided hash.

        Raises:
            requests.HTTPError: If connection / HTTP request fails.
            ChecksumError: If provided hash value doesn't match with
                           the one of downloaded files.

        Returns:
            str|None: The output file path if download is successful and
                      hash verification (if hash's provided) passes.
                      None if either download or hash verification fails.
        Note:
            - Downloads in chunks of 1MB (1048576 bytes) for memory efficiency
            - Concurrent calls for the same ``(out_dir_path, url)`` are
              serialized via a file lock (keyed on the request, acquired
              *before* it is made) and the destination is only ever updated
              via an atomic same-directory temp-file-then-rename, so
              concurrent/cold-cache callers never observe a partial file nor
              hammer the remote host with simultaneous requests for the same
              resource (this matters most when several dataset names share
              one upstream archive -- see e.g. ``RWTHLoader``'s
              ``acid_species``/``microgel_size`` families).
        """

        # size of a download package is set to 1MB
        # so that not the entire date gets loaded in to ram an once
        CHUNK_SIZE = 1048576

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "*/*",
        }

        if referer:
            headers["Referer"] = referer

        os.makedirs(out_dir_path, exist_ok=True)

        lock_key = hashlib.sha256(f"{out_dir_path}::{url}".encode()).hexdigest()[:16]
        lock_path = os.path.join(out_dir_path, f".raman_data_{lock_key}.lock")

        with FileLock(lock_path, timeout=1800):
            # A sibling process may have completed this exact download while
            # we were waiting for the lock -- skip re-downloading if so.
            if out_file_name is not None:
                candidate_path = os.path.join(out_dir_path, out_file_name)
                if LoaderTools._is_file_ready(candidate_path):
                    return candidate_path

            checksum = hash_type.value() if hash_type else HASH_TYPE.md5.value()
            response = LoaderTools._get_with_retry(url, headers, timeout=60)
            with response:
                if out_file_name is None:
                    if "Content-Disposition" in response.headers:
                        content_disposition = response.headers['Content-Disposition']
                        parts = content_disposition.split(';')
                        for part in parts:
                            part = part.strip()
                            if part.lower().startswith('filename='):
                                out_file_name = part[len('filename='):].strip('"')
                                break
                        else:
                            out_file_name = url.split('/')[-1]
                    else:
                        out_file_name = url.split('/')[-1]

                out_file_path = os.path.join(out_dir_path, out_file_name)

                # Filename was only just resolved (from headers) -- re-check
                # now that we know the real destination path.
                if LoaderTools._is_file_ready(out_file_path):
                    return out_file_path

                total_size = (
                        int(response.headers.get("Content-Length", 0)) or None
                )

                # Write to a same-directory temp file, then atomically rename
                # onto the real destination -- a reader (or a crash mid-write)
                # never observes a partial file at out_file_path.
                tmp_path = f"{out_file_path}.tmp{os.getpid()}"
                try:
                    with open(tmp_path, "wb") as file:
                        with tqdm(
                                total=total_size,
                                unit="B",
                                unit_scale=True,
                                desc=f"Downloading file {out_file_name}",
                        ) as pbar:

                            for chunk in response.iter_content(CHUNK_SIZE):
                                if chunk:
                                    file.write(chunk)
                                    checksum.update(chunk)
                                    pbar.update(len(chunk))

                    # ZIP magic-byte validation
                    if out_file_name.lower().endswith(".zip"):
                        with open(tmp_path, "rb") as f:
                            if f.read(4) != b"PK\x03\x04":
                                raise CorruptedZipFileError(
                                    f"{out_file_path} is not a ZIP (likely HTML/JSON response)"
                                )

                    if hash_target and checksum.hexdigest() != hash_target:
                        raise ChecksumError(
                            expected_checksum=hash_target,
                            actual_checksum=checksum.hexdigest()
                        )

                    os.replace(tmp_path, out_file_path)
                except BaseException:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise

            return out_file_path

    @staticmethod
    def extract_zip_file_content(
        zip_file_path: str,
        unzip_target_subdir: Optional[str] = '',
        force_overwrite: Optional[bool] = False
    ) -> str | None:
        """
        Extracts all files and subfiles from a `.zip` file.
        The extracted files are saved in the same directory
        as the `.zip` file by default or in a subdirectory of files' location
        if specified.

        Args:
            zip_file_path (str): Path to the `.zip` file to extract content of.
            unzip_target_subdir (str, optional): The name of the subdirectory
                                                 unzipped files should be stored in.
            force_overwrite (bool, optional): A flag to determine whether
                                              to overwrite previously unzipped files
                                              or not. This doesn't affect any files
                                              other than of specified `.zip` file.

        Returns:
            str|None: If successful the path of the output directory else None.
        """
        if os.path.isfile(zip_file_path):
            if not zipfile.is_zipfile(zip_file_path):
                raise CorruptedZipFileError(zip_file_path)
        else:
            LoaderTools.logger.error(f"There's no .zip file stored at {zip_file_path}")
            return None

        # create dir with the same name as the zip file for uncompressed file data
        out_dir = os.path.join(os.path.dirname(zip_file_path), unzip_target_subdir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # extract files
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            file_list = zf.namelist()
            with tqdm(
                total=len(file_list),
                unit="files",
                unit_scale=True,
                desc=unzip_target_subdir,
            ) as pbar:
                for file in file_list:
                    if force_overwrite or not os.path.isfile(f"{out_dir}/{file}"):
                        zf.extract(file, out_dir)

                    pbar.update(1)

        return out_dir


    @staticmethod
    def read_mat_file(mat_file_path: str) -> dict[str, np.ndarray]|None:
        """
        Extracts the content of a MATLAB .mat file as a python dictonary.

        Args:
            mat_file_path (str): Complet path to the MAT file

            Returns:
                dict|None: A dictonary whre the keys are the variabel names definded in the file
                and data/header information as values. The data is converted to numpy arrays
                with a uniform type. If possible the type of the data is used, if not python strings
                used as default the data type.

                If the file couldn't be loaded None is returned.
        """

        try:
            #check the file format, matlab version 7.3 or above use hdf5
            #everything below can be opend using scipys loadmat
            if h5py.is_hdf5(mat_file_path):
                LoaderTools.logger.debug("Reading HDF5 .mat file")
                data_dict = {}
                with h5py.File(mat_file_path, "r") as file:
                    for key in file.keys():
                        try:
                            data_dict[key] = np.array(file[key])
                        except TypeError:
                            data_dict[key] = np.array(file[key], dtype=str)
            else:
                data_dict = io.loadmat(mat_file_path)
        except OSError as e:
            LoaderTools.logger.error(f"Failed to read .mat file: {e}")
            return None

        return data_dict

    @staticmethod
    def is_valid_zip(path):
        try:
            with zipfile.ZipFile(path, "r"):
                return True
        except zipfile.BadZipFile:
            return False

    @staticmethod
    def align_raman_shifts(raman_shifts_list: list[np.ndarray], spectra_list: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        min_shift = np.max([rs[0] for rs in raman_shifts_list])
        max_shift = np.min([rs[-1] for rs in raman_shifts_list])
        frequency_steps = [rs[1] - rs[0] for rs in raman_shifts_list]
        min_step = min(frequency_steps)
        raman_shifts = np.arange(min_shift, max_shift, min_step)
        new_spectra_list = [np.interp(raman_shifts, rs, spec) for rs, spec in zip(raman_shifts_list, spectra_list)]
        spectra = np.stack(new_spectra_list)
        return raman_shifts, spectra
