import os
from typing import List, Dict

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stop_words_set = set(stopwords.words("english"))

QUESTION_MARKER = " Q: "
COMPQ_MARKER = " QC: "
SIMPQ_MARKER = " QS: "
INTERQ_MARKER = " QI: "
ANSWER_MARKER = " A: "
EOQ_MARKER = "[EOQ]"
LIST_JOINER = " + "
BLANK = "__"
WH_WORDS = set(["who", "what", "where", "how", "why", "when", "which"])


def get_sequence_representation(
    origq: str,
    question_seq: List[str],
    answer_seq: List[str],
    compq_marker: str = COMPQ_MARKER,
    interq_marker: str = INTERQ_MARKER,
    answer_marker: str = ANSWER_MARKER,
    simpq_marker: str = SIMPQ_MARKER,
):
    ret_seq = compq_marker + origq
    if len(question_seq) != len(answer_seq):
        raise ValueError(
            "Number of generated questions and answers should match before"
            "question generation. Qs: {} As: {}".format(question_seq, answer_seq)
        )

    for aidx in range(len(answer_seq)):
        ret_seq += interq_marker
        ret_seq += question_seq[aidx]
        ret_seq += answer_marker + answer_seq[aidx]
    ret_seq += simpq_marker
    return ret_seq


def tokenize_str(input_str):
    return word_tokenize(input_str)


def stem_tokens(token_arr):
    return [stemmer.stem(token) for token in token_arr]


def filter_stop_tokens(token_arr):
    return [token for token in token_arr if token not in stop_words_set]


def stem_filter_tokenization(input_str):
    return stem_tokens(filter_stop_tokens(tokenize_str(input_str.lower())))


# functions borrowed from AllenNLP to parse JSONNET with env vars
def get_environment_variables() -> Dict[str, str]:
    """
    Wraps `os.environ` to filter out non-encodable values.
    """
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def _is_encodable(value: str) -> bool:
    """
    We need to filter out environment variables that can't
    be unicode-encoded to avoid a "surrogates not allowed"
    error in jsonnet.
    """
    # Idiomatically you'd like to not check the != b""
    # but mypy doesn't like that.
    return (value == "") or (value.encode("utf-8", "ignore") != b"")
