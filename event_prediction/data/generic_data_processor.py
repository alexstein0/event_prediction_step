import pandas as pd
from typing import List

from .data_utils import convert_to_str, remove_spaces, convert_to_bool


class GenericDataProcessor:
    def __init__(self, data_cfg):
        self._index_columns = list(data_cfg.index_columns)
        self._categorical_columns = list(data_cfg.categorical_columns)
        self._numeric_columns = list(data_cfg.numeric_columns)
        self._binary_columns = list(data_cfg.binary_columns)
        # self._static_numeric_columns = list(data_cfg.static_numeric_columns) if data_cfg.static_numeric_columns is not None else [] # prob dont need if statement
        self._label_columns = list(data_cfg.label_columns)
        self._raw_columns = list(data_cfg.raw_selected_columns)

        all_cols = []
        all_cols.extend(self._index_columns)
        all_cols.extend(self._categorical_columns)
        all_cols.extend(self._numeric_columns)
        all_cols.extend(self._binary_columns)
        # all_cols.extend([static_col["name"] for static_col in self._static_numeric_columns])
        all_cols.extend(self._label_columns)
        self._all_cols = []
        [self._all_cols.append(x) for x in all_cols if x not in self._all_cols]

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def select_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # only keep used columns and indexes
        cols = self.get_data_cols() + [x for x in self.get_index_columns() if x not in self.get_data_cols()]
        data = data[cols]
        return data

    def arrange_columns(self, data: pd.DataFrame, sort_col: str = None) -> pd.DataFrame:
        sort_columns = self.get_index_columns().copy()
        if sort_col is not None:
            sort_columns.append(sort_col)
        data = data.sort_values(by=sort_columns)
        return data

    def convert_columns_to_types(self, data: pd.DataFrame) -> pd.DataFrame:
        for col_name in data.columns:
            if col_name in self.get_numeric_columns():
                data[col_name] = data[col_name].astype(float)
            elif col_name in self.get_categorical_columns():
                data[col_name] = convert_to_str(data[col_name])
            elif col_name in self.get_binary_columns():
                data[col_name] = convert_to_bool(data[col_name])
            else:
                pass
                # print(f"Ignoring column {col_name}")
        return data

    def clean_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        for col_name in self.get_categorical_columns():
            data[col_name] = remove_spaces(data[col_name])
        return data

    def get_raw_columns(self) -> List[str]:
        return self._raw_columns

    def get_all_cols(self):
        return self._all_cols

    def get_data_cols(self):
        res = []
        [res.append(x) for x in self.get_all_cols() if x not in self.get_index_columns() and x not in res]  # + self.get_label_columns()
        return res

    def get_non_index_columns(self):
        res = []
        [res.append(x) for x in self.get_data_cols() if x not in self.get_index_columns() and x not in res]  # + self.get_label_columns()
        return res

    def get_index_columns(self):
        return self._index_columns

    def get_label_columns(self):
        return self._label_columns

    def get_numeric_columns(self):
        return self._numeric_columns

    def get_static_numeric_columns(self):
        return []

    def get_categorical_columns(self):
        return self._categorical_columns

    def get_binary_columns(self):
        return self._binary_columns

    def summarize_dataset(self, data: pd.DataFrame, bucket_amount: int = -1):
        print(f"Dataset has {len(data)} rows and columns: {data.columns}")
        for col in self.get_index_columns():
            unique_values = data[col].unique()
            print(f"Index col {col} has {len(unique_values)} unique values")

        for col in self.get_categorical_columns():
            unique_values = data[col].unique()
            print(f"Categorical col {col} has {len(unique_values)} unique values")

        for col in self.get_binary_columns():
            unique_values = data[col].unique()
            assert len(unique_values) == 2
            print(f"Binary col {col}")

        for col in self.get_numeric_columns():
            dc = data[col]
            mean = float(dc.mean())
            std = float(dc.mean())
            min_value = float(dc.min())
            max_value = float(dc.max())
            print(f"Numeric col {col} has stats: mean: {mean:6.3}, std: {std:6.3}, min_value: {min_value:6.3}, max_value: {max_value:6.3}")
            if bucket_amount > 0:
                print(f"\tWill create ~{2**bucket_amount} buckets")

        for col in self.get_label_columns():
            unique_values = data[col].unique()
            print(f"Label col {col} has {len(unique_values)} unique values")
