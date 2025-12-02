from typing import Optional, Tuple

import os, requests
import pandas as pd
import numpy as np

from raman_data.types import DatasetInfo, RamanDataset, CACHE_DIR, TASK_TYPE
from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.ILoader import ILoader
from raman_data.loaders.LoaderTools import  LoaderTools


class ZenLoader(ILoader):
    """
    A static class providing download functionality for datasets hosted on Zenodo.
    """
    @staticmethod
    def __load_10779223(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        zip_filename = "Raw data.zip"

        try:
            data_dir = LoaderTools.extract_zip_file_content(
                os.path.join(cache_path, "10779223", zip_filename),
                zip_filename.split(".")[0]
            )
        except CorruptedZipFileError as e:
            print(
                f"There seems to be an issue with dataset '10779223/sugar mixtures'. \n" \
                f"The following file could not be extracted: {zip_filename}"
            )
            return None

        if data_dir is None:
            print(
                f"There seems to be no file of dataset '10779223/sugar mixtures'.\n " \
                f"The following file could not be extracted: {zip_filename}"
            )
            return None

        data_folder_parent = os.path.join(
            data_dir,
            "Raw data",
            "Experimental data from sugar mixtures",
            "Raw datasets for analyses"
        )

        # load the data file
        snr = "Low SNR"
        data_folder = os.path.join(data_folder_parent, snr)
        
        # read raman_shifts with pandas
        data_path = os.path.join(data_folder, "data.pkl")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find data.pkl in {data_path}")

        raman_shifts = pd.read_pickle(data_path).T

        # read shifts with pandas
        spectra_path = os.path.join(data_folder, "spectral_axis.pkl")
        if not os.path.isfile(spectra_path):
            raise FileNotFoundError(
                f"Could not find spectral_axis.pkl in {spectra_path}"
            )

        spectra = pd.read_pickle(spectra_path)

        # read gt with pandas
        gt_path = os.path.join(data_folder, "gt_endmembers.pkl")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Could not find gt_endmembers.pkl in {gt_path}")

        concentrations = pd.read_pickle(gt_path).T

        return raman_shifts, spectra, concentrations


    @staticmethod
    def __load_256329(cache_path: str) -> np.ndarray | None:

        raise NotImplementedError

        zip_filename = "Kallepitis-et-al-Raw-data.zip"

        print(os.path.join(cache_path, "256329", zip_filename))

        data_dir = LoaderTools.extract_zip_file_content(
            os.path.join(cache_path, "256329", zip_filename),
            zip_filename
        )

        print(data_dir)

        if data_dir is None:
            return None

        data_folder_parent = os.path.join(
            data_dir,
            "Kallepitis-et-al-Raw-data",
            "Figure 3",
            "THP-1"
        )

        file_1 = os.path.join(data_folder_parent, "3D THP1 001_15 06 24.wip")

        # this what ramanspy does, it doenst work for me, why? I dont know
        #data = loadmat(file_name=file_1, squeeze_me=True)


    @staticmethod
    def __load_7644521(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        # data field names in the mat file
        data_keys = ["COM", "COM_125mM", "ML1_125mM", "ML2_125mM"]

        # load data file
        data_path = os.path.join(cache_path, "7644521", "Data.mat")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find Data.mat in {data_path}")

        # read content
        file_content = LoaderTools.read_mat_file(data_path)
        if file_content == None:
            print(
                f"There was an error while reading the dataset '7644521/Wheat lines'.\n " \
                f"The following file could not be read: {data_path}"
            )
            return None

        # spectra scale
        spectra = file_content["Calx"].squeeze()
        raman_shifts = []
        concentrations = np.array(np.empty)

        # raman shift data
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
    def __load_3572359(
        cache_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        # load data file
        data_path = os.path.join(cache_path, "3572359", "ILSdata.csv")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Could not find ILSdata.csv in {data_path}")

        df = pd.read_csv(data_path)
        concentrations = df.pop("conc").to_numpy()
        spectra = np.array(df.columns.values[8:], dtype=int)
        raman_shifts = df.loc[:, "400":].to_numpy().T

        return raman_shifts, spectra, concentrations


    __BASE_URL = "https://zenodo.org/api/records/ID/files-archive"
    __BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "zenodo")
    LoaderTools.set_cache_root(__BASE_CACHE_DIR, CACHE_DIR.Zenodo)

    DATASETS = {
        "sugar mixtures": DatasetInfo(
            task_type=TASK_TYPE.Regression,
            id="10779223",
            loader=__load_10779223,
            metadata={
                "full_name" : "Research data supporting \"Hyperspectral unmixing for Raman spectroscopy via physics-constrained autoencoders\"",
                "source" : "https://doi.org/10.5281/zenodo.10779223",
                "paper" : "https://doi.org/10.1073/pnas.2407439121",
                "description" : "Experimental and synthetic Raman data used in Georgiev et al., PNAS (2024) DOI:10.1073/pnas.2407439121."
            }
        ),
        # "Volumetric cells": DatasetInfo(
        #     task_type=TASK_TYPE.Classification,
        #     id="256329",
        #     load=__load_256329
        # ),
        "Wheat lines": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="7644521",
            loader=__load_7644521,
             metadata={
                "full_name" : "DIFFERENTIATION OF ADVANCED GENERATION MUTANT WHEAT LINES: CONVENTIONAL TECHNIQUES VERSUS RAMAN SPECTROSCOPY",
                "source" : "https://doi.org/10.5281/zenodo.7644521",
                "paper" : "https://doi.org/10.3389/fpls.2023.1116876",
                "description" : "Data and codes used in the manuscript titled \"DIFFERENTIATION OF ADVANCED GENERATION MUTANT WHEAT LINES: CONVENTIONAL TECHNIQUES VERSUS RAMAN SPECTROSCOPY\". The decision tree model is trained and tested using the Classification Learner app of MATLAB (R2021b, The MathWorks, Inc.)."
            }
        ),
        "Adenine": DatasetInfo(
            task_type=TASK_TYPE.Classification,
            id="3572359",
            loader=__load_3572359,
            metadata={
                "full_name" : "Dataset for Surface Enhanced Raman Spectroscopy for quantitative analysis: results of a large-scale European multi-instrument interlaboratory study",
                "source" : "https://doi.org/10.5281/zenodo.3572359",
                "paper" : "https://doi.org/10.1021/acs.analchem.9b05658",
                "description" : "This dataset contains all the spectra used in \"Surface Enhanced Raman Spectroscopy for quantitative analysis: results of a large-scale European multi-instrument interlaboratory study\". Data are available in 2 different formats: - a compressed archive with 1 folder (\"Dataset\") cointaining all the 3516 TXT files (1 file = 1 spectrum) uploaded by all participants (all spectra of the Interlaboratory study); - 1 single CSV file (“ILSspectra.csv”) with all the 3516 spectra uploaded by all participants in the form of a table. The data are structured as follow, with each row being 1 spectrum, preceded by metadata: \"labcode\", \"substrate\", \"laser\", \"method\", \"sample\", \"type\", \"conc\", \"batch\", \"replica\". Note that for those spectra starting after 400 cm-1 and/or ending before 2000 cm-1 missing values were expressed as NAs."
            }
        )
    }


    @staticmethod
    def download_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> str | None:
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
            print(f"[!] Cannot download {dataset_name} dataset with ZenLoader")
            return None

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)

        try:
            dataset_id = ZenLoader.DATASETS[dataset_name].id
            file_name = dataset_id + ".zip"
            url = ZenLoader.__BASE_URL.replace("ID", dataset_id)

            LoaderTools.download(url, cache_path, file_name)
        except requests.HTTPError as e:
            print(f"Could not download requested dataset")
            return None
        except OSError as e:
            print(f"A very bad error occurred :(")
            return None

        return cache_path


    @staticmethod
    def load_dataset(
        dataset_name: str,
        cache_path: Optional[str] = None
    ) -> RamanDataset | None:
        
        if not LoaderTools.is_dataset_available(dataset_name, ZenLoader.DATASETS):
            print(f"[!] Cannot load {dataset_name} dataset with ZenLoader")
            return None

        if not (cache_path is None):
            LoaderTools.set_cache_root(cache_path, CACHE_DIR.Zenodo)
        cache_path = LoaderTools.get_cache_root(CACHE_DIR.Zenodo)

        dataset_id = ZenLoader.DATASETS[dataset_name].id

        zip_file_path = os.path.join(cache_path, dataset_id + ".zip")

        if not os.path.isfile(zip_file_path):
            ZenLoader.download_dataset(dataset_name, cache_path)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                if not os.path.isdir(os.path.join(cache_path, dataset_id)):
                    LoaderTools.extract_zip_file_content(zip_file_path, dataset_id)
                break

            except CorruptedZipFileError as e:
                print(
                    f"{e.zip_file_path} is corrupted. " \
                    f"Attempt {retry_count + 1}/{max_retries}"
                )
                os.remove(e.zip_file_path)
                retry_count += 1

                if retry_count < max_retries:
                    ZenLoader.download_dataset(dataset_name, cache_path)
                else:
                    raise Exception(
                        f"Failed to download valid file after {max_retries} attempts"
                    )

        data = ZenLoader.DATASETS[dataset_name].loader(cache_path)

        if data is not None:
            raman_shifts, spectra, concentrations = data
            return RamanDataset(
                data=raman_shifts,
                target=concentrations,
                spectra=spectra,
                metadata=ZenLoader.DATASETS[dataset_name].metadata
            )
        
        return data
