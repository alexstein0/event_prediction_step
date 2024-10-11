from tokenizers import Tokenizer
from typing import Dict, List


def get_classification_options(tokenizer: Tokenizer, label_in_last_col: bool = False, label_col_prefix: str = None) -> Dict:
    """
    Get info needed for doing labeled-data style classification from next-token-prediction generated tokens.
    Assuming a vocabulary of tokens produced from tabular data in rows and columns, get the column number that
    contains the label for a given row, and get the token ids that correspond to the possible values of that label.
    """
    vocab = tokenizer.get_vocab()
    column_names = set([x.split('_')[0] for x, _ in vocab.items()])
    column_names = [int(x) for x in column_names if x.isnumeric()]
    if label_in_last_col:
        label_col = str(max(column_names))
    elif label_col_prefix is not None:
        label_col = label_col_prefix
    else:
        # We could have the label in some other column than last, but it probably makes the most sense to have
        # relevant data for a prediction in preceeding columns (and thus preceeding tokens).
        raise NotImplementedError("Only label_in_last_col=True or contains_row_token=True is supported right now")
    # ids = get_tokens_by_columns(tokenizer, [label_col,])  # [label_col]
    ids = get_tokens_by_columns(tokenizer)
    return {"num_cols": len(column_names), "label_ids": ids}


def get_tokens_by_columns(tokenizer: Tokenizer, return_columns: List[str] = None) -> Dict[str, Dict[str, int]]:
    """
    Get the ids of each token, where tokens come from columns of tabular data in the form "colname_val".
    For example "15_True" is a token from column 15 with the value True. The resulting dict will have
    each column and every possible token value for that column with id corresponding to that value.
    Ex: {"15": {"True": 14, "False": 7}}
    """
    vocab = tokenizer.get_vocab()
    special_tokens = set([x for x, _ in vocab.items() if len(x.split('_')) == 1])
    column_names = set([x.split('_')[0] for x, _ in vocab.items() if x not in special_tokens])
    if return_columns is None:
        return_columns = column_names
    mapping = {}
    for col_name in column_names:
        # columns we want to get ids for
        if col_name in return_columns:
            ids = {}
            for token, id in vocab.items():
                # token has format like "15_True", where 15 is the col_name and True is the value
                if token in special_tokens:
                    continue
                col = token.split('_')[0]
                if col_name == col:
                    val = token.split('_')[1]
                    ids[val] = id
            mapping[col_name] = ids
    return mapping
