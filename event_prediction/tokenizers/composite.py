from .generic_tokenizer import GenericTokenizer
import pandas as pd
import numpy as np
from typing import Tuple, Set, List, Dict

from event_prediction import data_utils

class Composite(GenericTokenizer):
    def __init__(self, tokenizer_cfgs, data_cfgs):
        super().__init__(tokenizer_cfgs, data_cfgs)

    # def pretokenize(self, dataset):
    #     data_col = dataset[self.data_processor.get_data_cols()]
    #     index = dataset[self.data_processor.get_index_columns()]
    #     all_tokens = data_utils.concat_dataframe_cols(data_col)
    #     if len(self.data_processor.get_label_columns()) > 0:
    #         labels = dataset[self.data_processor.get_label_columns()]
    #     else:
    #         labels = None
    #
    #     return all_tokens, index, labels

    def pretokenize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data_col = dataset[self.data_processor.get_data_cols()]
        all_tokens = data_utils.concat_dataframe_cols(data_col)
        all_tokens.name = "tokens"
        indexes = dataset[self.data_processor.get_index_columns()]
        labels = dataset[self.data_processor.get_label_columns()]

        # prepended_tokens, special_tokens_added = data_utils.get_prepended_tokens(indexes)
        # for st in special_tokens_added:
        #     self.special_tokens_dict[st] = st
        # return pd.concat([prepended_tokens, all_tokens, labels], axis=1).sort_index()
        return pd.concat([indexes, all_tokens, labels], axis=1)


    # def model(self, dataset):
    #     # todo check if this is right way to do composite? words are the concat of the whole sentence
    #     # this is effectively a "word level" tokenizer.  most of the work is done by the pretokenizer and this simply maps inputs to IDS
    #     self.add_special_tokens()
    #     for col_name in self.get_token_cols():
    #         self.add_all_tokens(set(dataset[col_name].values.tolist()))
    #     return dataset


    def post_process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # todo move this to pretokenize? (see atomic.pretokenize)
        # indexes = dataset[self.data_processor.get_index_columns()]
        #
        # dataset, special_tokens_added = data_utils.add_index_tokens(dataset["tokens"].to_frame(), indexes)
        # dataset.loc[dataset['tokens'].isnull(), 'tokens'] = dataset.loc[dataset['tokens'].isnull(), 'spec']
        # dataset = dataset["tokens"]
        #
        # special_tokens_added = []
        # for st in special_tokens_added:
        #     self.add_special_token(st)

        # dataset = dataset.astype(str)
        # prepended = dataset[dataset["prepended_tokens"].notnull()]["prepended_tokens"]
        # new_df = pd.DataFrame(prepended.to_list(), index=prepended.index)
        # todo labels also
        # combined = pd.concat([new_df, dataset["tokens"]], axis=1).sort_index()
        # todo maybe a better way to flatten in correct order
        # combined['i'] = combined.index
        # flattened = combined.melt(value_vars=[0, 1, 'tokens'], id_vars="i").sort_values('i')["value"]
        # flattened = flattened[~flattened.isna()].reset_index(drop=True)
        # return flattened

        # index = dataset[self.data_processor.get_index_columns()].to_dict('records')
        # index = [{'metadata': x} for x in index]
        # labels = dataset[self.data_processor.get_label_columns()].to_dict('records')
        # labels = [{'labels': x} for x in labels]
        # text = dataset[['tokens']].to_dict('records')
        # text = [{'text': x} for x in text]
        # output = []
        # for i in range(len(text)):
        #     this_dict = {}
        #     this_dict.update(index[i])
        #     this_dict.update(labels[i])
        #     this_dict.update(text[i])
        #     output.append(this_dict)
        # todo clean the below
        output = pd.DataFrame(columns=["metadata", "text", "labels"])
        output["metadata"] = dataset[self.data_processor.get_index_columns()].to_dict('records')
        output["text"] = dataset['tokens']
        output["labels"] = dataset[self.data_processor.get_label_columns()]
        return output

    def get_token_cols(self) -> List[str]:
        cols = self.data_processor.get_label_columns().copy()
        cols.append("tokens")
        return cols

    # TODO:
    # 1. Add a step that converts floats to ints, and probably buckets them. We can
    #    maybe normalize all numerical values to something like between 0 and 10 and
    #    then just use that as the buckets. (i.e. normalize all floats to 0-1.0, multiply
    #    by 10, then convert int.)
    # 2. There are currently 18 columns, and we can cut that way down. Most of the time-based
    #    columns should go. They specifically created an hour column, so maybe that is the
    #    only useful one.  All of the "static" columns should probably stay since they specifically
    #    created them.  Tracing through the preprocessing code in data_utils.py, can give us a
    #    sense of what they thought was useful.
    def tokenize(self, dataset: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[Set, Set]:
        trainset, testset = dataset
        trainset_tokens = data_utils.concat_dataframe_cols(trainset)
        testset_tokens = data_utils.concat_dataframe_cols(testset)
        return trainset_tokens, testset_tokens
