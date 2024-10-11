from .generic_data_processor import GenericDataProcessor
import pandas as pd
from .data_utils import add_hours_total_minutes, convert_dollars_to_floats, add_minutes_from_last, add_is_online, get_timestamps_from_str, add_static_fields
import numpy as np


class IbmFraudTransactionDataProcessor(GenericDataProcessor):
    def __init__(self, data_cfg):
        super(IbmFraudTransactionDataProcessor, self).__init__(data_cfg)

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a normalized dataframe"""

        # Pre conversion string cleaning
        data["Amount"] = convert_dollars_to_floats(data["Amount"],  log_scale=False)

        # convert to right datatype
        data = self.convert_columns_to_types(data)

        # add missing columns
        data = add_hours_total_minutes(data, get_timestamps_from_str(data, time_col="Time"))
        data["is_online"] = add_is_online(data["Merchant City"])

        # sort
        data = self.arrange_columns(data, "total_minutes")

        # add sort dependent columns
        data = add_minutes_from_last(data, "total_minutes", self.get_index_columns())

        # add static data columns (aggregate data per user)
        data = add_static_fields(data, reference_df=data, groupby_columns=["User"])

        # clean up columns
        data = self.clean_columns(data)

        # only keep used columns
        # data = data[self.get_all_cols()]
        data = self.select_data(data)

        return data
