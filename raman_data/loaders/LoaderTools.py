"""
General functions and enums meant to be used while loading certain dataset.
"""

from enum import Enum
from scipy import io

import os, h5py, hashlib
import numpy as np


class CACHE_DIR(Enum):
    """
    An enum contains names of environment variables used
    by certain loaders for saving their cache directories.
    """
    Kaggle = "KAGGLEHUB_CACHE"
    HuggingFace = "HF_HOME"
    Zenodo = "ZEN_CACHE"
    Zip = "ZIP_CACHE"


class TASK_TYPE(Enum):
    """
    An enum contains possible task types of a
    certain dataset.
    """
    Classification = 0
    Regression = 1


class HASH_TYPE(Enum):
    """    
    An enum contains possible hash types of a
    certain dataset's checksum. Each enums' value
    is a respective `hashlib`'s generating function
    which outputs the related hash upon execution.  
    """
    md5 = hashlib.md5
    sha256 = hashlib.sha256


from raman_data.loaders.ILoader import ILoader
from raman_data.exceptions import ChecksumError

from typing import Optional, List
from tqdm import tqdm
from pathlib import Path
import requests, zipfile


class LoaderTools:
    """
    A static class contains general methods that
    can be used while loading datasets.
    """
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
        except (KeyError):
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
            print(f"[!] Cache root folder for {loader_key.name}'s loader is set to: {path}")
            
            return
        
        for env_var in CACHE_DIR:
            os.environ[env_var.value] = path
        print(f"[!] Cache root folder is set to: {path}")


    @staticmethod
    def is_dataset_available(
        dataset_name: str,
        datasets: List[str]
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
            print(f"[!] Dataset {dataset_name} is not on the loader's list.")
        
        return check


    @staticmethod
    def list_datasets(
        loader: ILoader,
    ) -> None:
        """
        Prints a formatted list of datasets of a certain loader.

        Args:
            loader (ILoader): The loader to list datasets of.
        """
        print(f"[*] Datasets available with {loader.__qualname__}:")
        for dataset_name, task_type in loader.DATASETS.items():
            print(f" |-> Name: {dataset_name} | Task type: {task_type.name}")
            
    
    @staticmethod
    def download(
        url: str,
        out_dir_path: str,
        out_file_name: str,
        hash_target: Optional[str] = None,           
        hash_type: Optional[HASH_TYPE] = None
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

        
        # http get request 
        with requests.get(url=url, stream=True) as response: 

            #if its failed raise error 
            if not response.ok:
                raise requests.HTTPError(response=response)
            #total size of the to download data 
            total_size = int(response.headers["Content-Length"]) if "Content-Length" in response.headers else None
            os.makedirs(out_dir_path, exist_ok=True)
            
            out_file_path = os.path.join(out_dir_path, out_file_name)
            if os.path.exists(out_file_path):
                return out_file_path
            
            #open/create file to write the data to
            with open(out_file_path, 'xb+') as file:
                #displays a loadingbar in the cli 
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading file {out_file_name}") as pbar:
                    #writes chunks with predefined size into the file 
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            file.write(chunk)
                            #calculate checksum
                            checksum.update(chunk)
                            pbar.update(len(chunk))
                                
        
        #check the calculated checksum with the given one, 
        # if its a mismatch rasie an Error


        if hash_target and (checksum.hexdigest() != hash_target):

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
        if not zipfile.is_zipfile(zip_file_path):
            print(f"There's no .zip file stored at {zip_file_path}")
            return None
        
        # create dir with the same name as the zip file for uncompressed file data
        out_dir = os.path.join(os.path.dirname(zip_file_path), unzip_target_subdir)
        os.makedirs(out_dir, exist_ok=True)
        
        # extract files
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            file_list = zf.namelist()
            with tqdm(
                total=len(file_list),
                unit="files",
                unit_scale=True,
                desc=unzip_target_subdir
            ) as pbar:
                for file in file_list:
                    if (force_overwrite or not os.path.isfile(f"{out_dir}/{file}")):
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
                print("hdf5")
                with h5py.File(mat_file_path, "r") as file:
                    for key in file.keys():
                        try:
                            data_dict[key] = np.array(file[key])
                        except TypeError:
                            data_dict[key] = np.array(file[key], dtype=str)
            else:
                data_dict = io.loadmat(mat_file_path)
        except OSError as e:
            return None

        return data_dict
