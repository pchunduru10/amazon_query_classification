import os
import io
import json
import requests
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# set seed to reproduce results
random.seed(42)


def create_dir(config):
    """
    Create training/test dir.
    :param config: json config file
    """
     # create train directory if not exists
    Path(os.path.join(config["data_dir"],"training_data","Pre")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(config["data_dir"],"training_data","Post")).mkdir(parents=True, exist_ok=True)

    # create test directory if not exists
    Path(os.path.join(config["data_dir"],"test_data","Pre")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(config["data_dir"],"test_data","Post")).mkdir(parents=True, exist_ok=True)


def generate_data(config):
    
    """
    Process the csv file to create train/test dir.
    :param config: json config file
    """
    response = requests.get(config["dataset_url"], timeout= 10)
    print('status_code:', response.status_code)

    if response.ok:
        amazon_df = pd.concat([chunk for chunk in tqdm(pd.read_csv(io.StringIO(response.text), chunksize=10000), desc='Loading data')])
    else:
        amazon_df = None
    
    # format label column to string
    amazon_df["label"] = amazon_df["label"].astype(str)

    train_df, test_df = stratified_subsample(amazon_df)
    print(f"Train Shape {train_df.shape}")
    print(f"Test Shape {test_df.shape}")

    #save data
    save_file(train_df,os.path.join(config["data_dir"],"training_data"))
    save_file(test_df, os.path.join(config["data_dir"],"test_data"))


def save_file(input_df: pd.DataFrame, filepath: str = ""):
    """
    Save every row of df as csv.

    :param input_df: train_df/test_df
    :type input_df:  dataframe
    :param filepath: Training/Test dir, defaults to None
    :type filepath: string, optional
    """

    for name, df in input_df.groupby(config["target_col"]):
        print(f"Saving group {name} in {filepath}")

        for idx, row in df.reset_index().iterrows():
            # row.to_csv(os.path.join(filepath,name,f"{name}_text_{idx+1}.csv"),index =False)
            with open(os.path.join(filepath,name,f"{name}_text_{idx+1}.txt"), 'w') as f:
                f.write(row[config["feature_col"]])
            
            

def stratified_subsample(df):
    """
    Stratified sampling of full dataset to reduce computations

    :param df: amazon product df
    :type df: pandas dataframe
    """
     # random sample for faster computations
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(int(np.rint(config["total_sample_size"]*len(x)/len(df))))).sample(frac=1).reset_index(drop=True)
    
    #split data into train and test
    train_df, test_df, _, _ = train_test_split(df[[config["feature_col"], config["target_col"]]],
                                            df[config["target_col"]],
                                            stratify=df[config["target_col"]], 
                                            test_size=0.20)

    return train_df, test_df
  


if __name__ == "__main__":
    with open("config.json", "r") as c:
        config = json.load(c)
        print("Config read successful")

    create_dir(config)
    generate_data(config)