"""Description. Rename labels in CLINC150 for texts from banking domain."""

import json
import pandas as pd 
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from typing import Dict, List, Tuple 
from pandas.core.frame import DataFrame

DATA_DIR = "./../datasets/"

BANKING = {"pin_change": "change_pin"}

CREDIT_CARDS = {
    "replacement_card_duration": "card_delivery_estimate", 
    "expiration_date": "card_about_to_expire", 
    "report_lost_card": "lost_or_stolen_card", 
    "card_declined": "declined_card_payment"
}

NEW_LABELS = list(BANKING.values()) + list(CREDIT_CARDS.values())

with open(DATA_DIR+"/b77_label_mapping.json") as f: 
    b77_LABELID = json.load(f)
    b77_LABELID = {
        label: id 
        for label, id in zip(b77_LABELID["label"], b77_LABELID["id"]) 
    }

def update_label(label: str) -> str: 

    if label in BANKING.keys(): 
        label = BANKING[label]
        
    elif label in CREDIT_CARDS.keys(): 
        label = CREDIT_CARDS[label]
    
    return label 

def process_entries(entries: List, in_labels: bool) -> DataFrame: 
    """Description. Update banking-related labels in CLINC150."""

    data = {
        text: update_label(label) 
        for text, label in entries
    }
    df = pd.DataFrame\
        .from_dict(data, orient="index")\
        .reset_index()\
        .rename(columns={"index": "text", 0: "label"})
    
    if in_labels: 
        df = df.loc[df.label.isin(NEW_LABELS), :]
        df["label"] = df["label"].apply(lambda label: b77_LABELID[label])
    else: 
        df = df.loc[~df.label.isin(NEW_LABELS), :]

    df = df.reset_index(drop=True)
    return df

def df_list_to_dict(dfs: List) -> DatasetDict: 
    df = pd.concat(dfs).reset_index(drop=True)
    ds = DatasetDict({"test": Dataset.from_pandas(df)}) 
    
    return ds

def preprocess_clinc150(ds: Dict) -> Tuple: 
    """Description. 
    Split banking77-related entries accross train/dev/test datasets."""

    in_dfs, out_dfs = [], []

    loop = tqdm(ds.items())

    for key, entries in loop:
        loop.set_description(f"Processing {key}...")

        in_dfs.append(process_entries(entries, in_labels=True))
        out_dfs.append(process_entries(entries, in_labels=False))

    in_ds = df_list_to_dict(in_dfs)
    out_ds = df_list_to_dict(out_dfs)

    return in_ds, out_ds

def save_dataset(ds: DatasetDict, file_name: str): 
    ds.save_to_disk(f"{DATA_DIR}{file_name}")

if __name__ == "__main__": 

    with open(f"{DATA_DIR}CLINC150.json") as f: 
        clinc150 = json.load(f)

    in_ds, out_ds = preprocess_clinc150(clinc150)

    save_dataset(in_ds, "clinc150_in")
    save_dataset(out_ds, "clinc150_out")
    
