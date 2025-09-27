from typing import Optional
from numpy import ndarray, genfromtxt, load
from pathlib import Path
from pandas import read_excel
import requests, os

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools
from raman_data import types

class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.
    """
    BASE_URL = "https://zenodo.org/api/records/"
    BASE_CACHE_DIR = os.path.join(Path.home(), ".cache", "zenodo")
    
    #TODO Needs to be checked if Classification or Regression
    DATASETS = {
        "10779223": TASK_TYPE.Classification,
        "256329": TASK_TYPE.Classification,
        "3572359": TASK_TYPE.Classification,
        "7644521": TASK_TYPE.Classification
    }
    
    
    @staticmethod
    def download_dataset(
        dataset_name: Optional[str|None] = None,
        dataset_id: Optional[str|None] = None,
        file_name: Optional[str|None] = None,
        cache_path: Optional[str|None] = None
    ) -> str | None:
        """
        Downloads certain dataset into a predefined cache folder.

        Args:
            dataset_name (str, optional): The name of a dataset to download.
            dataset_id (str, optional): The Zenodo id of the dataset.
            file_name (str, optional): This loader dosen't support this feature, 
                                       it will always download the entire dataset.
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
            print(f"Could not download requested dataset: {dataset_name} ID: {dataset_id if dataset_id else "None"}")
            return None
        except OSError as e:
            return None
                
        return file_dir
    
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        file_name: str, 
        dataset_id: Optional[str|None] = None,
        cache_path: str | None = None
    ) -> ndarray | None:
        """
        Loads certain dataset's file from cache folder as a numpy array.
        If requested file isn't in the cache folder, downloads it into that folder.

        Args:
            dataset_name (str): The name of a dataset.
            file_name (str): The name of a specific dataset's file to load.
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
        
        file_path = os.path.join(cache_path, dataset_name, file_name)
        
        if not os.path.exists(file_path):
            print(f"[!] Dataset's file {file_name} not found at: {cache_path}")
            ZenLoader.download_dataset(
                dataset_name=dataset_name,
                cache_path=cache_path,
                dataset_id= dataset_id if dataset_id else None
            )
            
         # Converting Excel files with pandas
        if file_name[-4:] in ["xlsx", ".xls"]:
            return read_excel(io=file_path).to_numpy()

        # Converting / reading numpy's native files
        if file_name[-4:] == ".npy":
            return load(file=file_path)

        # Converting CSV files with numpy
        return genfromtxt(fname=file_path, delimiter=",")
        
        
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