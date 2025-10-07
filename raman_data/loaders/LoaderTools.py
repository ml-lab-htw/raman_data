"""
General functions and enums meant to be used while loading certain dataset.
"""

from enum import Enum
import os

class CACHE_DIR(Enum):
    """
    An enum contains names of environment variables used
    by certain loaders for saving their cache directories.
    """
    Kaggle = "KAGGLEHUB_CACHE"
    HuggingFace = "HF_HOME"
    Zenodo = "ZEN_CACHE"


class TASK_TYPE(Enum):
    """
    An enum contains possible task types of a
    certain dataset.
    """
    Classification = 0
    Regression = 1


from raman_data.loaders.ILoader import ILoader
from raman_data.exceptions import ChecksumError

from typing import Optional, List
from tqdm import tqdm
from pathlib import Path
import requests, zipfile, hashlib


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
            environment variable that stores the cache path. If None,
            sets the given path for all loaders.
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
        md5_hash: Optional[str] = None
    ) -> str | None:
        """
        Download a file from a URL with optional MD5 verification.
        
        Args:
            url (str): The URL to download the file from.
            out_dir_path (str): The full path of the directory where the downloaded file will be saved.
            out_file_name (str): The name of the file being downloaded.
            md5_hash (Optional[str], optional): Expected MD5 hash of the file for integrity 
                verification. If provided, the download is considered failed if the hash 
                doesn't match. Defaults to None.
        Returns:
                    str | None: The output file path if download is successful and MD5 verification 
                            (if provided) passes, None if the download fails or MD5 verification fails.
        Note:
            - Downloads in chunks of 1MB (1048576 bytes) for memory efficiency
        """
        
        # size of a download package is set to 1MB 
        # so that not the entire date gets loaded in to ram an once
        CHUNK_SIZE = 1048576
        
        checksum = hashlib.md5()
        
        # http get request 
        with requests.get(url=url, stream=True) as response: 
            print(response.status_code)
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
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading file: {out_file_name}") as pbar:
                    #writes chunks with predefined size into the file 
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            file.write(chunk)
                            #calculate checksum
                            checksum.update(chunk)
                            pbar.update(len(chunk))
                                
        
        #check the calculated checksum with the given one, 
        # if its a mismatch rasie an Error
        
        if md5_hash is not None and checksum.hexdigest().strip() != md5_hash.strip():
            os.remove(out_file_path)
            raise ChecksumError(expected_checksum=md5_hash, actual_checksum=checksum.hexdigest())
        
        return out_file_path
    
    
    @staticmethod
    def extract_zip_file_comtent(zip_file_path: str, zip_file_name: str) -> str | None:
        """
        Extracts all files and subfiles from a zip file into a directory with the same name as the zip file.
        The extracted files are saved in the same directory as the zip file.

        Args:
            zip_file_path (str): Path to the zip file.
            zip_file_name (str): The name of the zip file.

        Returns:
            str|None: If successfull the path of the output directory else None.
        """
        if zipfile.is_zipfile(zip_file_path):
            #create dir with the same name as the zip file for uncompressed file data
            out_dir = f"{os.path.dirname(zip_file_path)}/{zip_file_name.split('.')[0]}"
            os.makedirs(out_dir, exist_ok=True)

            #extract files 
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                file_list = zf.namelist()
                with tqdm(total=len(file_list), unit="files", unit_scale=True, desc=f"Extracting file:  {zip_file_name}") as pbar:
                    for file in file_list:
                        if not os.path.isfile(f"{out_dir}/{file}"):
                            zf.extract(file, out_dir)
                            
                        pbar.update(1)

            # TODO check if we really want to delete the zip file after extraction
            # os.remove(zip_file_path)
            
            return out_dir