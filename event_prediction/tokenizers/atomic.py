from .generic_tokenizer import GenericTokenizer
from event_prediction import data_utils
from typing import List
import pandas as pd

class Atomic(GenericTokenizer):
    def __init__(self, tokenizer_cfgs, data_cfgs):
        super().__init__(tokenizer_cfgs, data_cfgs)

    def pretokenize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        indexes = dataset[self.data_processor.get_index_columns()]
        # prepended_tokens, special_tokens_added = data_utils.get_prepended_tokens(indexes)
        # dataset = data_utils.interweave_series([prepended_tokens, dataset])
        all_tokens = dataset[self.data_processor.get_data_cols()]

        labels = dataset[self.data_processor.get_label_columns()]


        # todo this is code ot add next row/next column but probably isnt necessary
        # for st in special_tokens_added:
        #     self.special_tokens_dict[st] = st
        # normal_rows = dataset["spec"].isnull()
        # special_rows = dataset["spec"].notnull()
        # special_table = dataset[special_rows]["spec"].to_frame()
        #
        # row_token = "ROW"
        # col_token = "COL"
        # main_table = dataset[normal_rows][self.data_processor.get_data_cols()]
        # dataset = data_utils.add_special_tabular_tokens(main_table, add_col_sep=col_token, add_row_sep=row_token)
        # self.special_tokens_dict[row_token] = row_token
        # self.special_tokens_dict[col_token] = col_token
        #
        # all_tokens = data_utils.cols_to_words(dataset, special_table)
        return pd.concat([indexes, all_tokens, labels], axis=1)


    def post_process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(columns=["metadata", "text", "labels"])
        output["metadata"] = dataset[self.data_processor.get_index_columns()].to_dict('records')
        data_col = dataset[self.data_processor.get_data_cols()]
        text = data_utils.concat_dataframe_cols(data_col, " ")  # todo, store separately?
        output["text"] = text
        output["labels"] = dataset[self.data_processor.get_label_columns()]
        return output


    def get_token_cols(self):
        cols = self.data_processor.get_label_columns().copy()
        cols.extend(self.data_processor.get_data_cols())
        return cols
