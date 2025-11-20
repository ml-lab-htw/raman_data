from typing import Optional

import os, requests
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
    def load_10779223(cache_path: str) -> np.ndarray|None:
        zip_filename = "Raw data.zip"
        
        data_dir = LoaderTools.extract_zip_file_content(os.path.join(cache_path, "10779223", zip_filename), zip_filename)
        
        if data_dir is None:
            return None

        data_folder_parent = os.path.join(data_dir, "Raw data", "Experimental data from sugar mixtures", "Raw datasets for analyses")

        snr = "Low SNR"
        data_folder = os.path.join(data_folder_parent, snr)
        data_path = os.path.join(data_folder, "data.pkl")
        
        
        # load the data file
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find data.pkl in {data_path}")

        # read raman_shifts with pandas
        raman_shifts = pd.read_pickle(data_path).T

        # read shifts with pandas
        spectra_path = os.path.join(data_folder, "spectral_axis.pkl")

        if not os.path.isfile(spectra_path):
            raise FileNotFoundError(f"Could not find spectral_axis.pkl in {spectra_path}")

        spectra = pd.read_pickle(spectra_path)

        # read gt with pandas
        gt_path = os.path.join(data_folder, "gt_endmembers.pkl")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Could not find gt_endmembers.pkl in {gt_path}")

        concentrations = pd.read_pickle(gt_path).T

        return raman_shifts, spectra, concentrations
    
    
    @staticmethod
    def __load_256329(cache_path: str) -> np.ndarray|None:
        raise NotImplementedError
    
    
    @staticmethod
    def load_7644521(cache_path: str) -> np.ndarray|None:

        #data field names in the mat file
        data_keys = ["COM", "COM_125mM", "ML1_125mM", "ML2_125mM"]

        data_path = os.path.join(cache_path, "7644521", "Data.mat")

        #load data file
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find Data.mat in {data_path}")

        #read content
        file_content = LoaderTools.read_mat_file(data_path)

        if file_content == None:
            return None
        
        #spectra scale
        spectra = file_content["Calx"].squeeze()
        raman_shifts = []
        concentrations = np.array(np.empty)

        #raman shift data
        for key in data_keys:
            data_row = file_content[key]
            raman_shifts.append(data_row)

        #TODO Get the concentrations: 
        #tihs is waht ramanspy dose:
        #   y = []
        #   for i, dataset in enumerate(labels):
        #       #apperently they just add the index of the labe to each label as concentration
                #COM would 0, COM_125mM would be 1, and so on 
        #       y.append(np.repeat(i, data[dataset].shape[0]))

        raman_shifts = np.concatenate(raman_shifts).T
        
        return raman_shifts, spectra, concentrations
    
    
    @staticmethod
    def load_3572359(cache_path: str) -> np.ndarray|None:
        
        data_path = os.path.join(cache_path, "3572359", "ILSdata.csv")

        #load data file
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find ILSdata.csv in {data_path}")
        

        df = pd.read_csv(data_path)
        concentrations = df.pop("conc").to_numpy()
        spectra = np.array(df.columns.values[8:], dtype=int)
        raman_shifts = df.loc[:, "400":].to_numpy().T
        
        return raman_shifts, spectra, concentrations

    
    BASE_URL = "https://zenodo.org/api/records/ID/files-archive"
    BASE_CACHE_DIR = os.path.join(os.path.expanduser('~'), ".cache", "zenodo")
    
    DATASETS = {
        "sugar mixtures" : types.datasetInfo(
                                        task_type=TASK_TYPE.Regression, 
                                        id="10779223", 
                                        loader=load_10779223),
        #"Volumetric cells" : types.datasetInfo(
        #                                task_type=TASK_TYPE.Classification, 
        #                                id="256329", 
        #                                loader=load_256329),
        "Wheat lines" : types.datasetInfo(
                                        task_type=TASK_TYPE.Classification, 
                                        id="7644521", 
                                        loader=load_7644521),
        "Adenine" : types.datasetInfo(
                                        task_type=TASK_TYPE.Classification, 
                                        id="3572359", 
                                        loader=load_3572359)
    }
    
    @staticmethod
    def download_dataset(
        dataset_name: str,
        file_name: str | None = None,
        cache_path: Optional[str|None] = None
    ) -> str | None:
        
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        cache_path = cache_path if cache_path else ZenLoader.BASE_CACHE_DIR
        
        #dataset_to_download = []
        
        #if dataset_names:
        #    dataset_to_download = dataset_names
        #else:
        #    dataset_to_download = ZenLoader.DATASETS.keys()
        
        
        #for dataset_name in dataset_to_download:
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
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
        dataset_name: str,
        file_name: str | None = None,
        cache_path: str | None = None
    ) -> np.ndarray | None:
        
        if cache_path is not None:
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        cache_path = cache_path if cache_path else ZenLoader.BASE_CACHE_DIR
        
        
        #datasets_to_load = []
        
        loaded_datasets = []
        
        #if dataset_name:
        #    datasets_to_load.append(dataset_name)
        #else: 
        #    datasets_to_load = ZenLoader.DATASETS.keys()
            
            
        #for dataset_name in datasets_to_load:
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
            return None
        
        dataset_id = ZenLoader.DATASETS[dataset_name].id
        
        zip_file_name = dataset_id + ".zip"
        zip_file_path = os.path.join(cache_path, zip_file_name)
        
        if not os.path.isfile(zip_file_path):
            ZenLoader.download_dataset(dataset_name, cache_path)
        
        if not os.path.isdir(os.path.join(cache_path, dataset_id)):
            LoaderTools.extract_zip_file_content(zip_file_path, zip_file_name)
        
        raman_shifts, spectra, concentrations = ZenLoader.DATASETS[dataset_name].loader(cache_path)

        dataset = types.RamanDataset(data=raman_shifts, 
                                     target=concentrations, 
                                     metadata={"spectra":spectra})
        
        return dataset
        
        