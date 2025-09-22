from typing import Optional
from numpy import ndarray
from pathlib import Path
import requests, os

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools
from raman_data import types

class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.
    """
    BASE_URL = "https://zenodo.org/api/records/"
    BASE_CACHE_DIR = os.path.join(Path.home(), ".cache", "raman_data")
    
    #TODO Needs to be checked if Classification or Regression
    DATASETS = {
        "10779223": TASK_TYPE.Classification,
        "256329": TASK_TYPE.Classification,
        "3572359": TASK_TYPE.Classification,
        "7644521": TASK_TYPE.Classification
    }
    
    
    def download_dataset(
        self,
        dataset_name: Optional[str|None] = None,
        file_name: Optional[str|None] = None,
        cache_dir: Optional[str|None] = None,
        dataset_id: Optional[str|None] = None
    ) -> str | None:
        """
        Downloads certain dataset into a predefined cache folder.

        Args:
            dataset_name (str, optional): The name of a dataset to download.
            file_name (str, optional): The name of a specific dataset's file to download.
                                       If None, downloads whole dataset.
                                       This loader dosn't suport this feature, 
                                       it will always download the entire dataset
            cache_dir (str, optional): The path to save the dataset to.
                                       If None, uses the lastly saved path.

        Returns:
            str: The path the dataset is downloaded to.
            If the dataset isn't on the list of a loader, returns None.
        """
        
        
        if cache_dir is not None:
            pass
            #LoaderTools.set_cache_root(cache_dir, CACHE_DIR.Zenodo)
        
        if dataset_id and dataset_id.isdigit():
            
            try:
                response = requests.get(self.BASE_URL + dataset_id.strip())
            except requests.HTTPError as e:
                print(f"Could not get further information about the requested dataset:{dataset_id}")
                print(f"The following Error occured: {e}")
                raise e
            
            file_list = self.__get_downloadeble_files(retrieve_response=response)
        elif dataset_name:
            try:
                response = requests.get(self.BASE_URL,
                                        params={"q": dataset_name.strip(),
                                                 "status": "published"},
                                        headers={"response": "application/json"})
            except requests.HTTPError as e:
                print(f"Could not get further information about the requested dataset:{dataset_name}")
                print(f"The following Error occured: {e}")
                raise e
            
            file_list = self.__get_downloadeble_files(search_response=response)
        else:
            return

        if not file_list:
            return
        
        try:
            for file in file_list: 
                file_path = LoaderTools.download(
                    url=file.download_link,
                    out_dir_path=self.BASE_CACHE_DIR,
                    out_file_name=file.key,
                    md5_hash=file.checksum
                )
                
                if file_path:
                    file_dir = LoaderTools.extract_zip_file_comtent(file_path, file.key)
        except requests.HTTPError as e:
            pass
        except OSError as e:
            pass
        finally:
            pass
                
        return file_dir
    
    
    def load_dataset(
        self,
        dataset_name: str, 
        file_name: str, 
        cache_dir: str | None = None
    ) -> ndarray | None:
        """
        Loads certain dataset's file from cache folder as a numpy array.
        If requested file isn't in the cache folder, downloads it into that folder.

        Args:
            dataset_name (str): The name of a dataset.
            file_name (str): The name of a specific dataset's file to load.
            cache_dir (str, optional): The path to look for the file at.
                                       If None, uses the lastly saved path.
                                       If "default", sets the default path ('~/.cache').

        Raises:
            NotImplementedError: If not implemented raises the error by default.

        Returns:
            ndarray: A numpy array representing the loaded file.
            If the dataset isn't on the list of a loader, returns None.
        """
        
        
        raise NotImplementedError
        
        
    @staticmethod
    def __get_downloadeble_files(
        retrieve_response: Optional[requests.Response] = None,
        search_response: Optional[requests.Response] = None
    ) -> list[types.ZenodoFileInfo] | None:
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