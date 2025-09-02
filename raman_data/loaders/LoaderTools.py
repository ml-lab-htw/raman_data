from enum import Enum

class CACHE_DIR(Enum):
    Kaggle = "KAGGLEHUB_CACHE"
    HuggingFace = "HF_HOME"


class TASK_TYPE(Enum):
    Classification = 0
    Regression = 1


from typing import Dict, Optional, List
import os

class LoaderTools:
    @staticmethod
    def get_cache_root(
        env_var: CACHE_DIR
    ) -> str:
        try:
            return os.environ[env_var.value]
        except (KeyError):
            return "Default path at \'~/.cache/kagglehub/\'"

    
    @staticmethod
    def set_cache_root(
        path: str,
        loader_key: Optional[CACHE_DIR] = None
    ) -> None:
        if loader_key is None:
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
        check = dataset_name in datasets
        if not check:
            print(f"[!] Dataset {dataset_name} is not on the loader's list.")
        
        return check


    @staticmethod
    def list_datasets(
        loader_key: CACHE_DIR,
        datasets: Dict[str, TASK_TYPE]
    ) -> None:
        print(f"[*] Datasets available with {loader_key.name}'s loader:")
        for dataset_name, task_type in datasets.items():
            print(f" |-> Name: {dataset_name} | Task type: {task_type.name}")
