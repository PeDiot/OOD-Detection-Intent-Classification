"""Description. Rename labels in CLINC150 for texts from banking domain."""

import json
import pandas as pd 
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from typing import Dict, List, Tuple 
from pandas.core.frame import DataFrame

DATA_DIR = "./../datasets/"

with open(DATA_DIR + "CLINC150_domains.json") as f: 
    domains = json.load(f) 

banking_intents = domains["banking"] + domains["credit_cards"]

def select_banking_entries(entries: List) -> DataFrame: 
    """Description. Select banking-related entries."""

    banking_entries = {
        text: label
        for text, label in entries
        if label in banking_intents
    }

    banking_entries = pd.DataFrame\
        .from_dict(banking_entries, orient="index")\
        .reset_index()\
        .rename(columns={"index": "text", 0: "label"})
    
    return banking_entries

def df_list_to_ds(dfs: List) -> DatasetDict: 
    df = pd.concat(dfs).reset_index(drop=True)
    ds = DatasetDict({"test": Dataset.from_pandas(df)}) 
    
    return ds

def preprocess_clinc150(ds: Dict) -> DatasetDict: 
    """Description. 
    Apply banking intents selection methods to train/dev/test sets in CLINC150."""

    dfs = [
        select_banking_entries(entries)
        for ds, entries in ds.items()
        if ds in ("train", "val", "test")
    ] 

    ds = df_list_to_ds(dfs) 

    return ds


def save_dataset(ds: DatasetDict, file_name: str): 
    file_path = f"{DATA_DIR}{file_name}"
    ds.save_to_disk(file_path)
    print(f"Dataset successfully saved at {file_path}")

if __name__ == "__main__": 

    with open(DATA_DIR + "CLINC150.json") as f: 
        clinc150 = json.load(f)

    clinc150_bank = preprocess_clinc150(clinc150)
    print(clinc150_bank)
    save_dataset(clinc150_bank, "clinc150_bank")    
