from enum import Enum

class CACHE_DIRS(Enum):
    Kaggle = "KAGGLEHUB_CACHE"
    HuggingFace = "HF_HOME"


from typing import List, Optional
import os

class LoaderTools:
    def get_cache_root(
        env_var: CACHE_DIRS
    ) -> str:
        try:
            return os.environ[env_var.value]            
        except (KeyError):
            return "Default path at \'~/.cache/kagglehub/\'"
    
    
    def set_cache_root(
        path: str,
        loader_key: Optional[CACHE_DIRS] = None
    ) -> None:
        if loader_key != None:
            os.environ[loader_key.value] = path
            print(f"[!] Cache root folder for {loader_key.name}'s loader is set to: {path}")
            
            return
        
        for env_var in CACHE_DIRS:
            os.environ[env_var.value] = path
        print(f"[!] Cache root folder is set to: {path}")


    def is_dataset_available(
        name: str,
        datasets: List[str]
    ) -> bool:
        check = name in datasets
        if not check:
            print(f"[!] Dataset {name} is not on the loader's list.")
        
        return check
    

    def list_datasets(
        loader_key: CACHE_DIRS,
        datasets: List[str]
    ) -> None:
        print(f"[*] Datasets available with {loader_key.name}'s loader:")
        for dataset_name in datasets:
            print(f" |-> Name: {dataset_name}")

