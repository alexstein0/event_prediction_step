from .generic_data_processor import GenericDataProcessor
import pandas as pd
from .data_utils import add_hours_total_minutes, get_timestamps_from_str, add_minutes_from_last


class BeijingPollutionDataProcessor(GenericDataProcessor):
    def __init__(self, data_cfg):
        super(BeijingPollutionDataProcessor, self).__init__(data_cfg)

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a normalized dataframe"""

        # data["User"] = data["reviewerID"]

        # convert to right datatype
        data = self.convert_columns_to_types(data)

        # add missing columns
        data = add_hours_total_minutes(data, get_timestamps_from_str(data, year_col="year", day_col="day", month_col="month", time_col="Time"))

        # data = add_hours_total_minutes(data, timestamps=pd.to_datetime(data["unixReviewTime"], unit='s'))

        # sort
        data = self.arrange_columns(data, "total_minutes")

        # add sort dependent columns
        data = add_minutes_from_last(data, "total_minutes", self.get_index_columns())

        # clean up columns
        data = self.clean_columns(data)

        # only keep used columns and indexes
        cols = self.get_data_cols() + [x for x in self.get_index_columns() if x not in self.get_data_cols()]
        data = data[cols]

        return data
