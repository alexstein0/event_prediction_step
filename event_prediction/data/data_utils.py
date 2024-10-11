import io
import json
import logging
import os
import tarfile
import gzip
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils import data
from tqdm import tqdm
from datasets import Dataset, load_dataset
from event_prediction import utils
import zipfile

log = logging.getLogger(__name__)


def get_huggingface_dataset(cfg):
    data_files = {"train": cfg.url}
    dataset = load_dataset("csv", data_files=data_files)
    return dataset['train']


def bytes_to_df(data_bytes, cfg, raw_data_dir_name="data_raw", save_tar_to_disk=False, save_csv_to_disk=False) -> pd.DataFrame:
    data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
    _, ext = os.path.splitext(urlparse(cfg.url).path)
    filepath = os.path.join(data_dir, f"{cfg.name}{ext}")

    if ext == ".tgz":
        if save_tar_to_disk:
            os.makedirs(data_dir, exist_ok=True)
            write_bytes(data_bytes, filepath)
        try:
            data_bytes = extract_tar(data_bytes)
        except tarfile.ReadError as e:
            log.error(f"Error when trying to extract file. Double-check that the URL actually exists: {cfg.url}")
            raise
        if cfg.raw_type == "csv":
            df = pd.read_csv(data_bytes)
        else:
            raise ValueError

    elif ext == ".gz":
        if save_tar_to_disk:
            os.makedirs(data_dir, exist_ok=True)
            write_bytes(data_bytes, filepath)
        try:
            output = []
            with gzip.open(data_bytes, mode='r') as lines:
                for line in lines:
                    row = line.decode('utf-8')
                    output.append(json.loads(row))
        except tarfile.ReadError as e:
            log.error(f"Error when trying to extract file. Double-check that the URL actually exists: {cfg.url}")
            raise

        if cfg.raw_type == "json":
            df = pd.DataFrame.from_records(output)
        else:
            raise ValueError
    elif ext == ".zip":
        with zipfile.ZipFile(data_bytes, 'r') as z:
            with z.open('trans.asc') as csvfile:
                df = pd.read_csv(csvfile, delimiter=";")  # might not be this delimited
    else:
        df = pd.read_csv(data_bytes)
    return df


def download_and_save_data(cfg, raw_data_dir_name="data_raw", save_tar_to_disk=False, save_csv_to_disk=False) -> pd.DataFrame:
    if cfg.url is not None and not cfg.process_only:
        data_bytes = download_data_from_url(cfg.url)
        df = bytes_to_df(data_bytes, cfg, raw_data_dir_name, save_tar_to_disk, save_csv_to_disk)
    else:
        log.info(f"NO URL")
        df = get_data_from_raw(cfg)  # already locally in raw folder just needs processing
    if save_csv_to_disk:
        data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
        if len(list(cfg.raw_index_columns)) > 0:
            df.sort_values(list(cfg.raw_index_columns), inplace=True)
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, f"{cfg.name}.csv")
        df.to_csv(filepath, index=False)
        log.info(f"Saved csv to {filepath}")
    return df


def get_data_from_raw(cfg, raw_data_dir_name="data_raw") -> pd.DataFrame:
    """
    Return a dataframe of the dataset specified in the config. For a given dataset
    first we will look for it on disk, and if it is not there we will download it.
    Handles either .csv file or .tgz file with single .csv file inside.
    """
    data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
    csv_file = os.path.join(data_dir, f"{cfg.name}.csv")
    if cfg.huggingface is not None:
        ds_name = cfg.huggingface.name
        splits = cfg.huggingface.splits
        return pd.read_csv(f"hf://datasets/{ds_name}/{splits[0]}.csv.gz")
    elif os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        log.info(f"Read CSV from {csv_file}")
    elif os.path.isdir(os.path.join(data_dir, cfg.name)):  # folder of files
        dir = os.path.join(data_dir, cfg.name)
        df_list = []
        for csv_file in os.listdir(dir):
            df_list.append(pd.read_csv(os.path.join(dir, csv_file)))
        df = pd.concat(df_list, ignore_index=True)
    else:
        _, ext = os.path.splitext(urlparse(cfg.url).path)
        filepath = os.path.join(data_dir, f"{cfg.name}{ext}")
        if not os.path.exists(filepath):
            log.info(f"Data not found at {filepath}.  First download dataset from {cfg.url} using download_and_save_data.py")
            raise IOError
        data_bytes = read_bytes(filepath)
        log.info(f"Loaded data_bytes from to {filepath}")
        df = bytes_to_df(data_bytes, cfg)  # dont save the data, just convert to data_bytes
    return df


def save_small(data: pd.DataFrame, path: str, name: str, rows: int = 1000):
    small_df = data.iloc[:rows]
    path = os.path.join(path, f"{name}_small.csv")
    small_df.to_csv(path, index=False)
    log.info(f"Saved mini csv to {path}")
    log.info(f"Small dataset has {small_df.shape[0]} samples")
    return path


def download_data_from_url(url: str) -> io.BytesIO:
    """Download a file to memory without writing it to disk"""
    response = requests.get(url, stream=True, verify=False)
    file_size = int(response.headers.get("Content-Length", 0))
    # Initialize a downloader with a progress bar
    downloader = TqdmToLogger(
        log,
        iterable=response.iter_content(1024),
        desc=f"Downloading {url}",
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    data = io.BytesIO()
    for chunk in downloader.iterable:
        data.write(chunk)
        # Update the progress bar manually
        downloader.update(len(chunk))
    # Reset the file object position to the start of the stream
    data.seek(0)
    return data


def write_bytes(data: io.BytesIO, filepath: str) -> None:
    log.info(f"Saving to {filepath}")
    with open(filepath, "wb") as f:
        for byte in data:
            f.write(byte)
    # Reset the file object position to the start of the stream
    data.seek(0)


def read_bytes(filepath: str) -> io.BytesIO:
    with open(filepath, "rb") as file:
        content = file.read()
    return io.BytesIO(content)


def extract_tar(data: io.BytesIO) -> io.BytesIO:
    """Extract a .tgz file to memory"""
    log.info(f"Extracting .tgz file...")
    with tarfile.open(fileobj=data, mode='r:gz') as tar:
        num_files = len(tar.getmembers())
        assert num_files == 1, f"Expected single csv file in tarball but got {num_files} files."
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                data = f.read()
    return io.BytesIO(data)

def extract_gzip(data: io.BytesIO) -> io.BytesIO:
    """Extract a .gz file to memory"""
    log.info(f"Extracting .gz file...")
    with gzip.open(data, mode='r') as lines:
        data = lines.read()
    return io.BytesIO(data)


def get_timestamps_from_str(X: pd.DataFrame,
                            year_col: str = "Year",
                            month_col: str = "Month",
                            day_col: str = "Day",
                            time_col: str = "Time") -> pd.Series:
    """Return a pd.Series of datetime objects created from a dataframe with columns 'Year', 'Month', 'Day', 'Time'"""
    try:
        X_hm = X[time_col].str.split(
            ":", expand=True
        )  # Expect "Time" to be in the format "HH:MM"
        hour = X_hm[0]
        minute = X_hm[1]
    except:
        try:
            hour = X["hour"]  # todo case sensitive
        except:
            hour = "00"
        try:
            minute = X["minute"]  # todo case sensitive
        except:
            minute = "00"
    d = pd.to_datetime(
        dict(
            year=X[year_col], month=X[month_col], day=X[day_col], hour=hour, minute=minute
        )
    )
    return d


def add_hours_total_minutes(X: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """Return a dataframe with new columns 'Hour' and 'total_minutes'"""
    X["Hour"] = timestamps.dt.hour
    # Add a column for total minutes from timestamp=0 to our dataframe
    zero_time = pd.to_datetime(np.zeros(len(X)))
    total_seconds = (timestamps - zero_time).dt.total_seconds().astype(int)
    total_minutes = total_seconds // 60
    X["total_minutes"] = total_minutes
    return X


def add_is_online(X: pd.Series, flag: str="ONLINE") -> pd.Series:
     return X == flag


def add_minutes_from_last(X: pd.DataFrame, minutes_col: str, by_columns: List[str] = None) -> pd.DataFrame:
    if by_columns is not None:
        col = X.groupby(by_columns)[minutes_col]
    else:
        col = X[minutes_col].copy()
    col = col.diff().fillna(0).astype("int64")
    X["total_minutes_from_last"] = col
    return X


def convert_to_str(X: pd.Series) -> pd.Series:
    X = X.convert_dtypes(convert_integer=True)
    null_spots = X.isna()
    X = X.astype(str)
    X[null_spots] = "NAN"
    return X


def convert_to_bool(X: pd.Series) -> pd.Series:
    if X.dtype == bool:
        return X
    rep = {'yes': True,
           'no': False,
           'true': True,
           'false': False}
    X = X.str.lower()
    X = X.replace(rep)
    return X.astype('str')


def convert_dollars_to_floats(X: pd.Series, log_scale: bool = True) -> pd.Series:
    X = X.str.replace("$", "").astype(float)
    if log_scale:
        X = np.log(X)
    return X


def bucket_numeric(X: pd.Series, bin_type: str, num_bins: Union[int, List[float]]) -> (pd.Series, pd.array):
    """
    Convert all numeric values to integers based on a specified number of bins.
    "uniform" bins will be of equal size, "quantile" bins will have an equal number of
    values in each bin.
    """
    assert bin_type in ["uniform", "quantile"], f"bin_type must be 'uniform' or 'quantile', not {bin_type}"

    if bin_type == "uniform":
        out, bins = pd.cut(X, bins=num_bins, retbins=True, labels=False, duplicates='drop')
    elif bin_type == "quantile":
        out, bins = pd.qcut(X, q=num_bins, retbins=True, labels=False, duplicates='drop')
    else:
        out, bins = None, None  # todo
    return out, bins


def convert_to_binary_string(X: pd.Series, digits_remaining: int = -1) -> (pd.Series, pd.array):
    if len(X) == 0:
        return [], pd.Series()  # shouldnt get here
    if len(X.unique()) == 1:
        return [''], X  # need to account for duplicates
    if digits_remaining == 0:
        return [''], pd.Series()  # all tokens in this bucket will be the same
    med = X.median()
    if med == X.max():
        return [''], pd.Series()  # because we split <= sometimes everything is in the first bucket (this might solve uniqueness problem too)
    left = X <= med  # gets a 0
    right = X > med  # gets a 1
    digits_remaining = max(digits_remaining - 1, -1)
    left_strings, left_buckets = convert_to_binary_string(X[left], digits_remaining)
    right_strings, right_buckets = convert_to_binary_string(X[right], digits_remaining)
    left_tokens = ['0' + x for x in left_strings]
    right_tokens = ['1' + x for x in right_strings]
    output_series = pd.Series(index=range(len(X))).astype(str)
    output_series[left.reset_index(drop=True)] = left_tokens
    output_series[right.reset_index(drop=True)] = right_tokens
    concat = (left_buckets.copy() if right_buckets.empty else right_buckets.copy() if left_buckets.empty else pd.concat([left_buckets, right_buckets], ignore_index=True))  # if both DataFrames non empty
    return output_series, concat



def normalize_numeric(df: pd.DataFrame, normalize_type: str) -> pd.DataFrame:
    # todo add other types of normalization
    if normalize_type == "normal":
        df = (df - df.mean(0)) / df.std(0)
    else:
        log.info("No normalization applied")
    return df

def concat_dataframe_cols(df: pd.DataFrame, separator: str= "_") -> pd.Series:
    return df.astype(str).apply(separator.join, axis=1)

def add_special_tabular_tokens(df: pd.DataFrame, add_col_sep: str='COL', add_row_sep: str='ROW') -> pd.DataFrame:
    output = pd.DataFrame()
    if add_col_sep is not None:
        for col in df.columns:
            output[col] = df[col]
            output[f'sep_{col}'] = add_col_sep
        output.drop(output.columns[-1], axis=1)
    if add_row_sep is not None:
        output['row_sep'] = add_row_sep

    return output

def cols_to_words(df: pd.DataFrame, second_table: pd.DataFrame=None) -> pd.Series:
    df["index_col"] = df.index
    all_tokens = df.values.tolist()
    if second_table is not None:
        second_table["index_col"] = second_table.index
        second_tokens = second_table.values.tolist()
        all_tokens.extend(second_tokens)
    all_tokens.sort(key=lambda x: x[-1])
    all_tokens = [x for row in all_tokens for x in row[:-1]]
    return pd.Series(all_tokens)

def remove_spaces(X: pd.Series) -> pd.Series:
    return X.str.replace(' ', '')


def get_prepended_tokens(labels: pd.DataFrame) -> (pd.DataFrame, List[str]):
    special_tokens_added = []
    index_tokens = []
    for token in labels.columns:
        tok_locs = labels[token][labels[token] != labels[token].shift()].copy().astype(str)
        tok_locs[:] = token
        tok_locs = tok_locs.to_frame()
        tok_locs.columns = ["prepended_tokens"]
        index_tokens.append(tok_locs)
        special_tokens_added.append(token)
    combined = pd.concat(index_tokens, axis=0)
    combined = combined.groupby(combined.index)["prepended_tokens"].apply(list)
    return combined, special_tokens_added


def interweave_series(datasets: List[pd.Series]) -> (pd.DataFrame):
    for ds in datasets:
        ds.reset_index(drop=True, inplace=True)
        ds.name = ["tokens"]

    return pd.concat([datasets], axis=0).sort_index().reset_index(drop=True)


# def get_train_test_split(X: pd.DataFrame, split_year: int = 2018) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Return a train-test split of the data based on a single year cutoff"""
#     train = X.loc[X["Year"] < split_year]
#     test = X.loc[X["Year"] >= split_year]
#     return train, test


def get_users(trainset: pd.DataFrame, testset: pd.DataFrame, user_col: str="User") -> Tuple[set, set]:
    """Return a list of users in both train and test sets, and users in both."""
    train_users = set(trainset[user_col].unique())
    test_users = set(testset[user_col].unique())
    train_test_users = train_users.intersection(test_users)
    test_only_users = test_users.difference(train_users)
    return train_test_users, test_only_users


def concatenated_col(df: pd.DataFrame, cols_to_concat: List[str]) -> pd.Series:
    """Create a Series (single column) that is a concatenation of selected columns in a df."""
    return df[cols_to_concat].astype(str).apply('_'.join, axis=1)


def add_static_fields(df: pd.DataFrame, reference_df: pd.DataFrame=None, groupby_columns=["User", "Card"]) -> pd.DataFrame:
    # reference_df is historic data that can be accessed at inference time. At train time, we can we can simply
    # use reuse the trainset as the reference dataset so that we add static values for all users in the trainset.
    user_static_values = get_user_level_static_values(reference_df, groupby_columns)
    
    # Add the static values for the users that appeared in the reference dataset. Since we are using a left join, this will
    # add NaNs for users that did not appear in the reference dataset. (Users that appeared in the reference dataset but not
    # in the current dataset are ignored by left join.)
    df = df.merge(user_static_values, on=groupby_columns, how="left")
    
    # Fill in missing user-level values with dataset-level values
    dataset_static_values = get_dataset_level_static_values(reference_df)
    df.fillna(value=dataset_static_values, inplace=True)

    return df
    

def get_user_level_static_values(df: pd.DataFrame, groupby_columns=["User", "Card"]) -> pd.DataFrame:
    """
    Computes static values from a DataFrame aggregated by the groupby columns (e.g. ["User", "Card"]).
    The columns to add are:
    1. Average dollar amount of the user
    2. Standard deviation dollar amount of the user
    3. Most frequent MCC
    4. Most frequent Use Chip
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        groupby_columns (List[str]): The columns to group by.
    Returns:
        pd.DataFrame: A DataFrame containing the static values, with one row per groupby combination. (e.g. one row per user)
    """

    assert pd.api.types.is_numeric_dtype(df['Amount']), f"Expected 'Amount' col to have numeric dtype but got: {df['Amount'].dtype}"
    get_most_frequent_item = lambda x: x.mode().iloc[0]
    grouped_static_df = df.groupby(groupby_columns).agg(
        avg_dollar_amt=("Amount", "mean"),
        std_dollar_amt=("Amount", "std"),
        top_mcc=("MCC", get_most_frequent_item),
        top_chip=("Use Chip", get_most_frequent_item),
    )
    return grouped_static_df


def get_dataset_level_static_values(df: pd.DataFrame) -> Dict:
    """Gather dataset-level values"""
    dataset_amt_avg = df["Amount"].mean()
    dataset_amt_std = df["Amount"].std()
    dataset_top_mcc = df["MCC"].mode().iloc[0]
    dataset_top_chip = df["Use Chip"].mode().iloc[0]
    dataset_static_values = {
        "avg_dollar_amt": dataset_amt_avg,
        "std_dollar_amt": dataset_amt_std,
        "top_mcc": dataset_top_mcc,
        "top_chip": dataset_top_chip,
    }
    return dataset_static_values


def create_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df)
    return dataset

def save_processed_dataset(dataset: Dataset, processed_data_dir_name: str, processed_data_file_name: str) -> str:
    """
    Save list of strings to a text file. They are saved with a newline seperator between
    each string in the list by default.
    the file is saved as a list of jsons where the keys are labels (such as 'text' and 'label')
    """
    data_dir = os.path.join(get_original_cwd(), processed_data_dir_name)
    filepath = os.path.join(data_dir, f"{processed_data_file_name}")
    os.makedirs(data_dir, exist_ok=True)
    # with open(filepath, "w") as outfile:
    #     json.dump(list_of_dicts, outfile)
    dataset.save_to_disk(filepath)

    # keys = dataset.keys()
    # values_list = zip(*(dataset[key] for key in keys))
    #
    # list_of_dicts = [dict(zip(keys, values)) for values in values_list]

    # return save_json(dataset, processed_data_dir_name, f"{processed_data_file_name}.json")
    return filepath


def load_processed_dataset(processed_data_dir_name: str, processed_data_file_name: str) -> pd.DataFrame:
    """
    Load the contents of a text file to a single string, with no newlines or 
    whitespace removed.
    """
    data_dir = os.path.join(get_original_cwd(), processed_data_dir_name)
    filepath = os.path.join(data_dir, f"{processed_data_file_name}")

    dataset = Dataset.load_from_disk(filepath)

    # list_of_dicts = read_json(processed_data_dir_name, f"{processed_data_file_name}.json")
    # # todo make this more generic
    # texts = []
    # labels = []
    # for i in list_of_dicts:
    #     texts.append(i['text'])
    #     labels.append(i['label'])
    # return {'text': texts, 'label': labels}
    return dataset.to_pandas()


def save_json(data: [Dict | List[Dict]], file_dir: str, file_name: str) -> str:

    file_dir = os.path.join(get_original_cwd(), file_dir)
    filepath = os.path.join(file_dir, file_name)
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, "w") as outfile:
        json.dump(data, outfile)
    return filepath


def read_json(file_dir: str, file_name: str) -> Dict | List[Dict]:

    filepath = os.path.join(get_original_cwd(), file_dir, file_name)
    assert os.path.exists(filepath), f"File not found at {filepath}"
    assert os.path.getsize(filepath) > 0, f"File is empty at {filepath}"

    with open(filepath, "r") as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError as e:
            log.error(f"File {filepath} is not in JSON format: {e}")
            raise e
    return data


class TqdmToLogger(tqdm):
    """File-like object to redirect tqdm output to a logger."""
    def __init__(self, logger, level=logging.INFO, *args, **kwargs):
        self.logger = logger
        self.level = level
        super().__init__(*args, **kwargs)

    def write(self, s):
        # Only log if the message is not empty or just a newline
        if s.rstrip() != '':
            self.logger.log(self.level, s.rstrip())

    def flush(self):
        pass