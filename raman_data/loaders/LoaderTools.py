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

    @staticmethod
    def download(
            url: str,
            out_dir_path: str,
            out_file_name: str,
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
            out_file_name (str): The name of the file to create.
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
        """

        # size of a download package is set to 1MB
        # so that not the entire date gets loaded in to ram an once
        CHUNK_SIZE = 1048576
        checksum = hash_type.value() if hash_type else HASH_TYPE.md5.value()

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "*/*",
        }

        if referer:
            headers["Referer"] = referer

        os.makedirs(out_dir_path, exist_ok=True)
        out_file_path = os.path.join(out_dir_path, out_file_name)

        # DO NOT trust existing files blindly
        if os.path.exists(out_file_path):
            os.remove(out_file_path)

        with requests.get(
                url=url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=60,
        ) as response:

            response.raise_for_status()

            total_size = (
                    int(response.headers.get("Content-Length", 0)) or None
            )

            with open(out_file_path, "wb") as file:
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
            with open(out_file_path, "rb") as f:
                if f.read(4) != b"PK\x03\x04":
                    os.remove(out_file_path)
                    raise CorruptedZipFileError(
                        f"{out_file_path} is not a ZIP (likely HTML/JSON response)"
                    )

        if hash_target and checksum.hexdigest() != hash_target:
            os.remove(out_file_path)
            raise ChecksumError(
                expected_checksum=hash_target,
                actual_checksum=checksum.hexdigest()
            )

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
