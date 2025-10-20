from pathlib import Path
from typing import Optional

import os, requests, csv
import pandas as pd
import numpy as np

from raman_data import types
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools


class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.
    """
    
    @staticmethod
    def __load_10779223(cache_path: str) -> np.ndarray|None:
        zip_filename = "Raw data.zip"
        
        data_dir = LoaderTools.extract_zip_file_content(os.path.join(cache_path, "10779223", zip_filename), zip_filename)
        
        if data_dir is None:
            return None
        
        data_folder_parent = os.path.join(data_dir, "Raw data", "Raw data", "Experimental data from sugar mixtures", "Raw datasets for analyses")

        snr = "Low SNR"
        data_folder = os.path.join(data_folder_parent, snr)
        data_path = os.path.join(data_folder, "data.pkl")
        
        
        # load the data file
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find data.pkl in {data_path}")

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
    def __load_256329(cache_path: str) -> np.ndarray|None:
        raise NotImplementedError
    
    
    @staticmethod
    def __load_7644521(cache_path: str) -> np.ndarray|None:
        raise NotImplementedError
    
    
    
    @staticmethod
    def load_3572359(cache_path: str) -> np.ndarray|None:
        
        data_path = os.path.join(cache_path, "3572359", "ILSdata.csv")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find ILSdata.csv in {data_path}")
        
        with open(data_path, newline='') as csv_file:
            #reader = csv.DictReader(csv_file)
            reader = csv.reader(csv_file)

            raman_shifts = np.zeros((3518, 534))
            concentrations = []

            i = 0

            for row in reader: 
                data_row = row[9:]

                if i == 0:
                    spectra = np.array(data_row, dtype=float)
                else:
                    #TODO:What the hell should I do with NA values
                    raman_shifts[i] = np.array(data_row, dtype=float)

                    concentrations.append(row[6])

                i += 1

            return raman_shifts, spectra, np.array(concentrations, dtype=float)


                    
                


                






    
    BASE_URL = "https://zenodo.org/api/records/ID/files-archive"
    BASE_CACHE_DIR = os.path.join(os.path.expanduser('~'), ".cache", "zenodo")
    
    DATASETS = {
        "sugar mixtures" : types.datasetInfo(
                                        task_type=TASK_TYPE.Regression, 
                                        id="10779223", 
                                        loader=__load_10779223),
        "three dimensional cell cultures" : types.datasetInfo(
                                        task_type=TASK_TYPE.Classification, 
                                        id="256329", 
                                        loader=__load_256329),
        "mutant wheat" : types.datasetInfo(
                                        task_type=TASK_TYPE.Classification, 
                                        id="7644521", 
                                        loader=__load_7644521),
        "Surface Enhanced Raman Spectroscopy" : types.datasetInfo(
                                        task_type=TASK_TYPE.Classification, 
                                        id="3572359", 
                                        loader=load_3572359)

    }
    
    @staticmethod
    def download_dataset(
        *dataset_names: str,
        cache_path: Optional[str|None] = None
    ) -> str | None:
        
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        cache_path = cache_path if cache_path else ZenLoader.BASE_CACHE_DIR
        
        dataset_to_download = []
        
        if dataset_names:
            dataset_to_download = dataset_names
        else:
            dataset_to_download = ZenLoader.DATASETS.keys()
        
        
        for dataset_name in dataset_to_download:
            if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
               print(f"[!] Cannot download {dataset_name} dataset with Zenodo loader")
               return None

            try:
                dataset_id = ZenLoader.DATASETS[dataset_name].id
                file_name = dataset_id + ".zip"
                url = ZenLoader.BASE_URL.replace("ID", dataset_id)
                
                LoaderTools.download(url, cache_path, file_name)
            except requests.HTTPError as e:
                print(f"Could not download requested dataset")
                return None
            except OSError as e:
                return None
            
        return cache_path
    
    
    @staticmethod
    def load_dataset(
        *dataset_names: str,
        cache_path: str | None = None
    ) -> np.ndarray | None:
        
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        cache_path = cache_path if cache_path else ZenLoader.BASE_CACHE_DIR
        
        
        datasets_to_load = []
        
        loaded_data = {}
        
        if dataset_names:
            datasets_to_load = dataset_names
        else: 
            datasets_to_load = ZenLoader.DATASETS.keys()
            
            
        for dataset_name in datasets_to_load:
            if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
                return
            
            dataset_id = ZenLoader.DATASETS[dataset_name].id
            
            zip_file_name = dataset_id + ".zip"
            zip_file_path = os.path.join(cache_path, zip_file_name)
            
            if not os.path.isfile(zip_file_path):
                ZenLoader.download_dataset(dataset_name, cache_path)
            
            if not os.path.isdir(os.path.join(cache_path, dataset_id)):
                LoaderTools.extract_zip_file_content(zip_file_path, zip_file_name)
                
            loaded_data[dataset_name] = ZenLoader.DATASETS[dataset_name].loader(cache_path)
            
        
        #TODO: convert data to RamanDatasets type