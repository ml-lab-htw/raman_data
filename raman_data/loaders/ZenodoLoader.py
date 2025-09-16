from typing import Optional
from numpy import ndarray
from tqdm import tqdm
import requests, zipfile, os, hashlib

from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools
from raman_data import types

class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.

    """
    BASE_URL = "https://zenodo.org/api/records/"
    
    #TODO Needs to be checked if Classification or Regression
    DATASETS = {
        "10779223": TASK_TYPE.Classification,
        "256329": TASK_TYPE.Classification,
        "3572359": TASK_TYPE.Classification,
        "7644521": TASK_TYPE.Classification
    }
    
    
    @staticmethod
    def download_dataset(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        file_name: Optional[str] = None,
        cache_path: Optional[str] = None
    ) -> str | None:
        """_summary_

        Args:
            dataset_name (Optional[str], optional): _description_. Defaults to None.
            dataset_id (Optional[str], optional): _description_. Defaults to None.
            file_name (Optional[str], optional): _description_. Defaults to None.
            cache_path (Optional[str], optional): _description_. Defaults to None.
        Returns:
            str|None: _description_
        """
        
        if dataset_id and dataset_id.isdigit():
            response = requests.get(ZenLoader.BASE_URL + dataset_id.strip())
            file_list = ZenLoader.__get_downloadeble_files(retrieve_response=response)
        elif dataset_name:
            response = requests.get(ZenLoader.BASE_URL,
                                    params={"q": dataset_name.strip(),
                                             "status": "published"},
                                    headers={"response": "application/json"})
            file_list = ZenLoader.__get_downloadeble_files(search_response=response)
        else:
            return None

        if not file_list:
            return None
        
        for file in file_list: 
            file_path = LoaderTools.download(url=file.download_link,
                                 out_dir_path="",
                                 out_file_name=file.key,
                                 md5_hash=file.checksum)
            if file_path:
                file_dir = LoaderTools.extract_zip_file_comtent(file_path, file.key)
                
        return file_dir
        
        
    @staticmethod
    def __get_downloadeble_files(
        retrieve_response: Optional[requests.Response] = None,
        search_response: Optional[requests.Response] = None
    ) -> list[types.ZenodoFileInfo] | None:
        """
        Extracts a list of downloadeble files from either the "zenodo search" or the "zenodo retrieve" response object.

        Args:
            retrieve_response (Optional[requests.Response], optional): Response Object from the zenode retrieve API endpoint.. Defaults to None.
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
            zfi = types.ZenodoFileInfo(id=result["id"],
                                       key=result["key"],
                                       size=result["size"],
                                       checksum=result["checksum"],
                                       download_link=result["links"]["self"],
                                       links=result["links"])
            file_list.append(zfi)
            
        return file_list