from typing import Optional

import os, requests
import pandas as pd
import numpy as np

from raman_data.types import DatasetInfo
from raman_data.loaders.ILoader import ILoader
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.LoaderTools import CACHE_DIR, TASK_TYPE, LoaderTools


class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.
    """
    
    @staticmethod
    def load_10779223(cache_path: str) -> np.ndarray|None:
        zip_filename = "Raw data.zip"
        
        try:
            data_dir = LoaderTools.extract_zip_file_content(os.path.join(cache_path, "10779223", zip_filename), zip_filename.split(".")[0])
        except CorruptedZipFileError as e:
            print(f"There seems to be an issue with dataset 10779223/sugar mixtures. \n The following file could not be extracted: {zip_filename}")
            return None
        
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
    def load_256329(cache_path: str) -> np.ndarray|None:

        raise NotImplementedError

        zip_filename = "Kallepitis-et-al-Raw-data.zip"

        print(os.path.join(cache_path, "256329", zip_filename))

        data_dir = LoaderTools.extract_zip_file_content(os.path.join(cache_path, "256329", zip_filename), zip_filename)

        print(data_dir)
        
        if data_dir is None:
            return None
        
        data_folder_parent = os.path.join(data_dir, "Kallepitis-et-al-Raw-data", "Figure 3", "THP-1")

        file_1 = os.path.join(data_folder_parent, "3D THP1 001_15 06 24.wip")

        # this what ramanspy does, it doenst work for me, why? I dont know
        #data = loadmat(file_name=file_1, squeeze_me=True)

    
    
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
    LoaderTools.set_cache_root(BASE_CACHE_DIR, CACHE_DIR.Zenodo)
    
    DATASETS = {
        "sugar mixtures": DatasetInfo(
            task_type=TASK_TYPE.Regression, 
            id="10779223", 
            loader=load_10779223),
        #"Volumetric cells": DatasetInfo(
        #   task_type=TASK_TYPE.Classification, 
        #   id="256329", 
        #   load=load_256329),
        "Wheat lines": DatasetInfo(
            task_type=TASK_TYPE.Classification, 
            id="7644521", 
            loader=load_7644521),
        "Adenine": DatasetInfo(
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
        
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
           return None

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)

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
        
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
            return None

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)
        
        dataset_id = ZenLoader.DATASETS[dataset_name].id
        
        zip_file_path = os.path.join(cache_path, dataset_id + ".zip")
        
        if not os.path.isfile(zip_file_path):
            ZenLoader.download_dataset(dataset_name, cache_path)
        
        try:

            if not os.path.isdir(os.path.join(cache_path, dataset_id)):
                LoaderTools.extract_zip_file_content(zip_file_path, dataset_id)
        except CorruptedZipFileError as e:
            print(f"{e.zip_file_path} got removed, because it was damaged.")
            os.remove(e.zip_file_path)

            ZenLoader.download_dataset(dataset_name, cache_path)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                if not os.path.isdir(os.path.join(cache_path, dataset_id)):
                    LoaderTools.extract_zip_file_content(zip_file_path, dataset_id)
                break 

            except CorruptedZipFileError as e:
                print(f"{e.zip_file_path} is corrupted. Attempt {retry_count + 1}/{max_retries}")
                os.remove(e.zip_file_path)
                retry_count += 1

                if retry_count < max_retries:
                    ZenLoader.download_dataset(dataset_name, cache_path)
                else:
                    raise Exception(f"Failed to download valid file after {max_retries} attempts")

        data = ZenLoader.DATASETS[dataset_name].loader(cache_path)
        if data is None:
            return None, None, None
        
        return data
        
        