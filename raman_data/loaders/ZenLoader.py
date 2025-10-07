from pathlib import Path
from typing import Optional

import os
import pandas as pd
import requests
from numpy import ndarray

from raman_data import types
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools


class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.
    """
    BASE_URL = "https://zenodo.org/api/records/"
    BASE_CACHE_DIR = os.path.join(Path.home(), ".cache", "zenodo")
    
    #TODO Needs to be checked if Classification or Regression
    DATASETS = {
        "10779223": TASK_TYPE.Regression,
        "256329": TASK_TYPE.Classification,
        "3572359": TASK_TYPE.Classification,
        "7644521": TASK_TYPE.Classification
    }
    
    
    @staticmethod
    def download_dataset(
        dataset_name: Optional[str|None] = None,
        dataset_id: Optional[str|None] = None,
        cache_path: Optional[str|None] = None
    ) -> str | None:
        """
        Downloads certain dataset into a predefined cache folder.

        Args:
            dataset_name (str, optional): The name of a dataset to download.
            dataset_id (str, optional): The Zenodo id of the dataset.
             cache_path (str, optional): The path to look for the file at.
                                       If None, uses the lastly saved path.
                                       If "default", sets the default path ('~/.cache').
        
        Notes: 
            If neither dataset_name nor dataset_id is given, all available dataset are downloaded. 
        
        Returns:
            str: The path the dataset is downloaded to.
            If the dataset isn't on the list of the loader, returns None.
        """
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with Zenodo loader")
            return None
        
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        cache_path = cache_path if cache_path else ZenLoader.BASE_CACHE_DIR
        
        file_list = []
        
        
        if dataset_id and dataset_id.isdigit():
            
            try:
                response = requests.get(ZenLoader.BASE_URL + dataset_id.strip())
            except requests.HTTPError as e:
                print(f"Could not get further information about the requested dataset:{dataset_id}")
                print(f"The following Error occured: {e}")
                raise e
            
            file_list = ZenLoader.__get_downloadeble_files(retrieve_response=response)
        elif dataset_name:
            try:
                response = requests.get(ZenLoader.BASE_URL,
                                        params={"q": dataset_name.strip(),
                                                 "status": "published"},
                                        headers={"response": "application/json"})
            except requests.HTTPError as e:
                print(f"Could not get further information about the requested dataset:{dataset_name}")
                print(f"The following Error occured: {e}")
                raise e
            
            file_list = ZenLoader.__get_downloadeble_files(search_response=response)
        else:
            for id in ZenLoader.DATASETS.keys():
                try:
                    response = requests.get(ZenLoader.BASE_URL + id)
                except requests.HTTPError as e:
                    print(f"Could not get further information about the requested dataset:{dataset_id}")
                    print(f"The following Error occured: {e}")
                    raise e
                
                file_list.extend(ZenLoader.__get_downloadeble_files(retrieve_response=response))
            
            
            
            
            

        if not file_list:
            return None
        
        try:
            for file in file_list: 
                file_path = LoaderTools.download(
                    url=file.download_link,
                    out_dir_path=cache_path,
                    out_file_name=file.key,
                    md5_hash=file.checksum
                )
                
                if file_path:
                    file_dir = LoaderTools.extract_zip_file_comtent(file_path, file.key)
                else: 
                    file_dir = None
        except requests.HTTPError as e:
            print(f"Could not download requested dataset: {dataset_name} ID: {dataset_id}")
            return None
        except OSError as e:
            return None
                
        return file_dir
    
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_id: Optional[str|None] = None,
        cache_path: str | None = None
    ) -> ndarray | None:
        """
        Loads certain dataset's file from cache folder as a numpy array.
        If requested file isn't in the cache folder, downloads it into that folder.

        Args:
            dataset_name (str): The name of a dataset.
            dataset_id (str, optional): The Zenodo id of the dataset.
            cache_path (str, optional): The path to look for the file at.
                                       If None, uses the lastly saved path.
                                       If "default", sets the default path ('~/.cache').

        Returns:
            ndarray: A numpy array representing the loaded file.
            If the dataset isn't on the list of a loader, returns None.
        """
        
        
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with Zenodo loader")
            return
        
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        cache_path = cache_path if cache_path else os.path.join(os.path.expanduser('~'), ".cache", "zenodo")

        data_folder_parent = os.path.join(cache_path, "Raw data", "Raw data", "Experimental data from sugar mixtures", "Raw datasets for analyses")

        snr = "Low SNR"
        data_folder = os.path.join(data_folder_parent, snr)
        data_path = os.path.join(data_folder, "data.pkl")

        if not os.path.isfile(data_path):
            zip_filename = "Raw data.zip"

            # check if the zip file exists
            if not os.path.isfile(os.path.join(cache_path, zip_filename)):
                # download the dataset if the zip file doesn't exist
                download_path = ZenLoader.download_dataset(dataset_name, dataset_id, None, cache_path)

            # extract the zip file
            LoaderTools.extract_zip_file_comtent(cache_path, zip_filename)

        # load the data file
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find {zip_filename} in {data_path}")

        # read spectra with pandas
        spectra = pd.read_pickle(data_path)

        # read shifts with pandas
        shifts_path = os.path.join(data_folder, "spectral_axis.pkl")
        if not os.path.isfile(shifts_path):
            raise FileNotFoundError(f"Could not find spectral_axis.pkl in {shifts_path}")

        raman_shifts = pd.read_pickle(shifts_path)

        # read gt with pandas
        gt_path = os.path.join(data_folder, "gt_endmembers.pkl")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Could not find gt_endmembers.pkl in {gt_path}")

        concentrations = pd.read_pickle(gt_path)

        return raman_shifts, spectra, concentrations
        
        
    @staticmethod
    def __get_downloadeble_files(
        retrieve_response: Optional[requests.Response] = None,
        search_response: Optional[requests.Response] = None
    ) -> list[types.ZenodoFileInfo]:
        """
        Extracts a list of downloadeble files from either the "Zenodo search" or the "Zenodo retrieve" response object.

        Args:
            retrieve_response (Optional[requests.Response], optional): Response Object from the Zenodo retrieve API endpoint.. Defaults to None.
            search_response (Optional[requests.Response], optional): Response Object from the Zenodo search API endpoint. Defaults to None.

        Returns:
            list | None: _description_
        """
        
        file_list = []
        
        if retrieve_response: 
            results = retrieve_response.json()["files"]
        elif search_response:
            results = search_response.json()["hits"]["hits"][0]["files"]
            
        for result in results:
            zenodo_file_info = types.ZenodoFileInfo(id=result["id"],
                                       key=result["key"],
                                       size=result["size"],
                                       checksum=result["checksum"],
                                       download_link=result["links"]["self"],
                                       links=result["links"])
            file_list.append(zenodo_file_info)
            
        return file_list