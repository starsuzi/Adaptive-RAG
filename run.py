"""
Instantiates a base config with various HP combinations.
"""

import re
import os
os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(os.getcwd()) + '/cache' # '/data/soyeong/cache'
import copy
import shutil
import json
import subprocess
import statistics
import itertools
from typing import Dict, List
import pandas as pd
import _jsonnet
import argparse 

from lib import (
    get_retriever_address,
    get_llm_server_address,
    infer_source_target_prefix,
    get_config_file_path_from_name_or_path,
)

import logging
import time

from functools import wraps

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper

dataset_to_prompt_set_to_qids = {
"squad": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    }, 
"trivia": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    }, 
"sciq": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },   
"tydiqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },  
"cpgqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },  
"sleepqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    }, 
"popqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },   
"nq": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },    
    "hotpotqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },
"temp": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },    
    "hotpotqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07",
            "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680",
            "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd",
            "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2",
            "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a",
            "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0",
            "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8",
            "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508",
            "5a88f9d55542995153361218",
            "5a758ea55542992db9473680",
            "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25",
            "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d",
            "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819",
            "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9",
            "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55",
            "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76",
            "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a",
            "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25",
            "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426",
            "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d",
            "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },
    "2wikimultihopqa": {
        "1": [
            "228546780bdd11eba7f7acde48001122",
            "97954d9408b011ebbd84ac1f6bf848b6",
            "a5995da508ab11ebbd82ac1f6bf848b6",
            "1ceeab380baf11ebab90acde48001122",
            "35bf3490096d11ebbdafac1f6bf848b6",
            "f86b4a28091711ebbdaeac1f6bf848b6",
            "f44939100bda11eba7f7acde48001122",
            "e5150a5a0bda11eba7f7acde48001122",
            "c6805b2908a911ebbd80ac1f6bf848b6",
            "13cda43c09b311ebbdb0ac1f6bf848b6",
            "f1ccdfee094011ebbdaeac1f6bf848b6",
            "028eaef60bdb11eba7f7acde48001122",
            "8727d1280bdc11eba7f7acde48001122",
            "79a863dc0bdc11eba7f7acde48001122",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
        ],
        "2": [
            "c6805b2908a911ebbd80ac1f6bf848b6",
            "5897ec7a086c11ebbd61ac1f6bf848b6",
            "028eaef60bdb11eba7f7acde48001122",
            "af8c6722088b11ebbd6fac1f6bf848b6",
            "1ceeab380baf11ebab90acde48001122",
            "5811079c0bdc11eba7f7acde48001122",
            "228546780bdd11eba7f7acde48001122",
            "e5150a5a0bda11eba7f7acde48001122",
            "f44939100bda11eba7f7acde48001122",
            "f1ccdfee094011ebbdaeac1f6bf848b6",
            "13cda43c09b311ebbdb0ac1f6bf848b6",
            "79a863dc0bdc11eba7f7acde48001122",
            "a5995da508ab11ebbd82ac1f6bf848b6",
            "cdbb82ec0baf11ebab90acde48001122",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
        ],
        "3": [
            "028eaef60bdb11eba7f7acde48001122",
            "8727d1280bdc11eba7f7acde48001122",
            "79a863dc0bdc11eba7f7acde48001122",
            "4724c54e08e011ebbda1ac1f6bf848b6",
            "e5150a5a0bda11eba7f7acde48001122",
            "35bf3490096d11ebbdafac1f6bf848b6",
            "a5995da508ab11ebbd82ac1f6bf848b6",
            "228546780bdd11eba7f7acde48001122",
            "97954d9408b011ebbd84ac1f6bf848b6",
            "f44939100bda11eba7f7acde48001122",
            "1ceeab380baf11ebab90acde48001122",
            "f86b4a28091711ebbdaeac1f6bf848b6",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
            "af8c6722088b11ebbd6fac1f6bf848b6",
            "5897ec7a086c11ebbd61ac1f6bf848b6",
        ],
    },
    "musique": {
        "1": [
            "2hop__804754_52230",
            "2hop__292995_8796",
            "2hop__496817_701819",
            "2hop__154225_727337",
            "2hop__642271_608104",
            "2hop__439265_539716",
            "2hop__195347_20661",
            "2hop__131516_53573",
            "2hop__427213_79175",
            "3hop1__443556_763924_573834",
            "2hop__782642_52667",
            "2hop__861128_15822",
            "4hop3__703974_789671_24078_24137",
            "3hop1__61746_67065_43617",
            "4hop3__463724_100414_35260_54090",
        ],
        "2": [
            "2hop__292995_8796",
            "2hop__154225_727337",
            "2hop__642271_608104",
            "2hop__195347_20661",
            "3hop1__61746_67065_43617",
            "2hop__861128_15822",
            "3hop1__753524_742157_573834",
            "2hop__496817_701819",
            "4hop3__703974_789671_24078_24137",
            "3hop1__858730_386977_851569",
            "2hop__804754_52230",
            "2hop__782642_52667",
            "2hop__102217_58400",
            "2hop__387702_20661",
            "3hop1__443556_763924_573834",
        ],
        "3": [
            "2hop__427213_79175",
            "3hop1__753524_742157_573834",
            "2hop__782642_52667",
            "2hop__496817_701819",
            "3hop1__443556_763924_573834",
            "4hop3__463724_100414_35260_54090",
            "2hop__292995_8796",
            "2hop__804754_52230",
            "3hop1__858730_386977_851569",
            "2hop__131516_53573",
            "2hop__387702_20661",
            "4hop3__703974_789671_24078_24137",
            "2hop__154225_727337",
            "3hop1__61746_67065_43617",
            "2hop__642271_608104",
        ],
    },
    "iirc": {
        "1": [
            "q_10344",
            "q_10227",
            "q_9591",
            "q_3283",
            "q_8776",
            "q_8981",
            "q_9518",
            "q_1672",
            "q_9499",
            "q_8173",
            "q_9433",
            "q_8350",
            "q_3268",
            "q_8736",
            "q_389",
        ],
        "2": [
            "q_9499",
            "q_10236",
            "q_2466",
            "q_10270",
            "q_8776",
            "q_9591",
            "q_10227",
            "q_8981",
            "q_9518",
            "q_3290",
            "q_8173",
            "q_8736",
            "q_10344",
            "q_389",
            "q_1672",
        ],
        "3": [
            "q_10344",
            "q_10227",
            "q_8776",
            "q_3268",
            "q_3283",
            "q_10270",
            "q_10236",
            "q_8736",
            "q_1672",
            "q_3208",
            "q_9433",
            "q_8350",
            "q_9591",
            "q_8981",
            "q_3290",
        ],
    },
}

instantiation_schemes = {
    "nor_qa": {},
    "oner": {"bm25_retrieval_count": ["15"]}, # gpt: ['6'] flan: ["15"]
    "oner_qa": {
        "bm25_retrieval_count": ["15"],
        "distractor_count": ['"1"'],
    },
    "ircot": {
        "bm25_retrieval_count": ["4", "6", "8"],
        "distractor_count": ['"1"', '"2"', '"3"'],
    },
    "ircot_qa": {
        "bm25_retrieval_count": ["6"], # gpt: ['3'] flan: ["6"]
        "distractor_count": ['"1"'],
    },
}


def hash_str(string: str) -> str:
    import hashlib

    return str(int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 10**8)


def verify_config(config_file_path: str) -> bool:
    # Verifies that all the file_paths used in the config are available.
    # If not, it prints a message of what's not available. This is to be run
    # before the predict.py command.
    import pandas as pd

    env_variables = {}
    retriever_address = get_retriever_address()
    env_variables["RETRIEVER_HOST"] = str(retriever_address["host"])
    env_variables["RETRIEVER_PORT"] = str(retriever_address["port"])
    llm_server_address = get_llm_server_address(args.llm_port_num)
    env_variables["LLM_SERVER_HOST"] = str(llm_server_address["host"])
    env_variables["LLM_SERVER_PORT"] = str(llm_server_address["port"])
    

    config_json = json.loads(_jsonnet.evaluate_file(config_file_path, ext_vars=env_variables))
    flattened_config_df = pd.json_normalize(config_json, sep=".")
    flattened_config_dict = flattened_config_df.to_dict(orient="records")[0]

    validate_paths = []
    for key, value in flattened_config_dict.items():
        assert isinstance(key, str)
        if key.endswith(".prompt_file"):
            if isinstance(value, str):
                validate_paths.append(value)
            else:
                validate_paths += value

    missing_paths = [validate_path for validate_path in validate_paths if not os.path.exists(validate_path)]
    if missing_paths:
        print(f"\nMissing path error in config: {config_file_path}")
        for missing_path in missing_paths:
            print(f"Missing prompt file_path    : {missing_path}")

    return not missing_paths


def instatiate_config(
    content: str,
    variable_replacements: Dict[str, str],  # NOTE: It's updated in-place with evaluated replacements when required.
) -> str:

    for variable_name, variable_value in variable_replacements.items():
        assert isinstance(variable_value, str)

        for invoked_variable_name in re.findall(r"\$[a-zA-Z0-9-_]+", variable_value):

            invoked_variable_name = invoked_variable_name.lstrip("$")
            assert (
                invoked_variable_name in variable_replacements
            ), "Invoked a variable name replacement within another that's not available in the passed dict."

            invoked_variable_value = variable_replacements[invoked_variable_name]
            variable_value = variable_value.replace("$" + invoked_variable_name, invoked_variable_value)

        if re.match(r"eval\(.+\)", variable_value):
            # means it's a python expression that needs to be evaluated.
            variable_value_ = re.sub(r"eval\((.+)\)", r"\1", variable_value)
            variable_value = str(eval(variable_value_))

        regex = re.compile(f"(.*local {variable_name} =) (.+?)(;.*)", re.DOTALL)
        if not regex.match(content):
            raise Exception(f"Variable name {variable_name} defined in the file.")

        original_variable_value = re.sub(regex, r"\2", content)
        if original_variable_value.strip() == "null":
            raise Exception(
                f"Looks like you're trying to replace variable ({variable_name}) that is set to be none. Likely an error."
            )
        variable_replacements[variable_name] = variable_value  # Updated inplace.
        content = re.sub(regex, r"\1 " + variable_value + r"\3", content)

    return content


def infer_dataset(content: str) -> str:
    regex = re.compile('.*local dataset = "(\w+)";.*', re.DOTALL)
    if not regex.match(content):
        raise Exception("Couldn't infer dataset from the config.")
    return re.sub(regex, r"\1", content).strip()


def summarize_and_results(hyperparameter_metrics_data: List[Dict]) -> None:
    for datum in hyperparameter_metrics_data:
        complete = datum.pop("complete")
        if not complete and datum["metric_value"] != "n/a":
            datum["metric_value"] = "** " + str(datum["metric_value"]) + " **"
    dataframe = pd.DataFrame(hyperparameter_metrics_data)
    print(dataframe)


def are_file_contents_equal(file_path_1: str, file_path_2: str) -> bool:
    with open(file_path_1, "r") as file:
        content_1 = file.read().strip()
    with open(file_path_2, "r") as file:
        content_2 = file.read().strip()
    return content_1 == content_2


def is_experiment_complete(
    original_experiment_file_path: str, prediction_file_path: str, metrics_file_path: str, variable_replacements: str
):
    if not os.path.exists(original_experiment_file_path):
        return False
    if not os.path.exists(prediction_file_path):
        return False
    if not os.path.exists(metrics_file_path):
        return False

    used_variable_replacements_file_path = os.path.join(
        os.path.dirname(prediction_file_path),
        os.path.splitext(os.path.split(prediction_file_path)[1])[0] + "_variable_replacements.json",
    )
    if not os.path.exists(used_variable_replacements_file_path):
        used_variable_replacements = ""
    else:
        with open(used_variable_replacements_file_path, "r") as file:
            used_variable_replacements = file.read().strip()

    if json.loads(variable_replacements or "{}") != json.loads(used_variable_replacements or "{}"):
        return False

    used_experiment_file_path = os.path.join(
        os.path.dirname(prediction_file_path),
        "config__"
        + os.path.splitext(os.path.split(prediction_file_path)[1])[0].replace("prediction__", "")
        + ".jsonnet",
    )
    if os.path.exists(used_experiment_file_path):
        if not are_file_contents_equal(original_experiment_file_path, used_experiment_file_path):
            return False

    with open(prediction_file_path, "r") as file:
        # This is necessary because sometimes retriever stops working and as a result
        # the whole thing returns empty results. (actually, need better detection for 'answer' type) .
        predictions = json.load(file)
        num_complete_items = sum([bool(value) for key, value in predictions.items()])
    return num_complete_items / len(predictions) > 0.9

@timed
def main():

    parser = argparse.ArgumentParser(description="Manager script for dealing with HP tuning.")
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("experiment_name_or_path", type=str, help="experiment_name_or_path")
    base_parser.add_argument(
        "--instantiation_scheme",
        type=str,
        help="instantiation_scheme",
        choices=instantiation_schemes.keys(),
        required=True,
    )
    base_parser.add_argument(
        "--prompt_set", type=str, help="prompt_set", choices={"1", "2", "3", "aggregate"}, required=True
    )
    base_parser.add_argument(
        '--set_name', type=str, help="set_name", required=True
    )
    # TODO
    # llm_port_num
    base_parser.add_argument(
        '--llm_port_num', type=str, help="llm_port_num", required=True
    )
    subparsers = parser.add_subparsers(title="Commands", metavar="", dest="command")
    write_subparser = subparsers.add_parser(
        "write", description="write files.", help="write files.", parents=[base_parser]
    )
    write_subparser.add_argument("--no_diff", action="store_true", help="don't show diff after writing the files.")
    write_subparser.add_argument("--best", action="store_true", help="pick and write best performing HP config.")
    write_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    write_subparser.add_argument(
        "--variable_replacements",
        type=str,
        help="json string for jsonnet local variable replacements.",
        default="",
    )
    subparsers.add_parser(
        "diff",
        description="show diff between original and new files.",
        help="show diff between original and new files.",
        parents=[base_parser],
    )
    subparsers.add_parser(
        "verify",
        description="verify all required prompt files exist.",
        help="verify all required prompt files exist.",
        parents=[base_parser],
    )
    print_subparser = subparsers.add_parser(
        "print", description="print file names.", help="print file names.", parents=[base_parser]
    )
    print_subparser.add_argument("--best", action="store_true", help="print the best HP config.")
    print_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    backup_subparser = subparsers.add_parser(
        "backup", description="backup output directory.", help="backup output directory.", parents=[base_parser]
    )
    backup_subparser.add_argument("--force", action="store_true", help="remove backup first if it exists.")
    print_backup_subparser = subparsers.add_parser(
        "print_backup",
        description="print backup output directory if it exists.",
        help="print backup output directory if it exists.",
        parents=[base_parser],
    )
    print_backup_subparser.add_argument("--force", action="store_true", help="remove original first if it exists.")
    recover_backup_subparser = subparsers.add_parser(
        "recover_backup",
        description="recover backup output directory to original directory.",
        help="recover backup output directory to original directory.",
        parents=[base_parser],
    )
    recover_backup_subparser.add_argument("--force", action="store_true", help="remove original first if it exists.")
    predict_subparser = subparsers.add_parser(
        "predict",
        description="run prediction on eval files.",
        help="run prediction on eval files.",
        parents=[base_parser],
    )
    predict_subparser.add_argument(
        "--variable_replacements",
        type=str,
        help="json string for jsonnet local variable replacements.",
        default="",
    )
    predict_subparser.add_argument("--silent", action="store_true", help="silent")
    predict_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    predict_subparser.add_argument("--skip_if_exists", action="store_true", help="skip if final metrics/output exist.")
    predict_subparser.add_argument("--use_backup", action="store_true", help="use backup output directory.")
    predict_subparser.add_argument("--best", action="store_true", help="predict on the best HP config.")
    predict_subparser.add_argument("--force", action="store_true", default=False, help="force predict if it exists")
    subparsers.add_parser(
        "delete_predictions",
        description="delete predictions directory.",
        help="delete predictions directory.",
        parents=[base_parser],
    )
    track_subparser = subparsers.add_parser(
        "track",
        description="track progress of completion.",
        help="track progress of completion.",
        parents=[base_parser],
    )
    track_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    track_subparser.add_argument("--use_backup", action="store_true", help="use backup output directory.")

    evaluate_subparser = subparsers.add_parser(
        "evaluate",
        description="evaluate prediction on eval files.",
        help="evaluate prediction on eval files.",
        parents=[base_parser],
    )
    evaluate_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    evaluate_subparser.add_argument("--use_backup", action="store_true", help="use backup output directory.")
    evaluate_subparser.add_argument(
        "--question-type-key-value", type=str, help="':' separated question-type-key-value.", default=None
    )
    evaluate_subparser.add_argument("--best", action="store_true", help="evaluate on the best HP config.")
    evaluate_subparser.add_argument("--skip_if_exists", action="store_true", help="skip if final metrics/output exist.")
    evaluate_subparser.add_argument(
        "--only_print", action="store_true", default=False, help="only print don't run evaluation"
    )
    evaluate_subparser.add_argument(
        "--official", action="store_true", default=False, help="use official eval scripts when available."
    )
    summarize_subparser = subparsers.add_parser(
        "summarize",
        description="summarize results of evaluation runs.",
        help="summarize results of evaluation runs.",
        parents=[base_parser],
    )
    summarize_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    summarize_subparser.add_argument("--use_backup", action="store_true", help="use backup output directory.")
    summarize_subparser.add_argument("--best", action="store_true", help="summarize only the best HP config.")
    summarize_subparser.add_argument(
        "--variable_replacements",
        type=str,
        help="json string for jsonnet local variable replacements.",
        default="",
    )
    summarize_subparser.add_argument(
        "--official", action="store_true", default=False, help="use official eval scripts when available."
    )
    ground_truth_check_subparser = subparsers.add_parser(
        "ground_truth_check",
        description="prints head of all ground-turths to check equality.",
        help="prints head of all ground-turths to check equality.",
        parents=[base_parser],
    )
    ground_truth_check_subparser.add_argument("--evaluation_path", type=str, help="evaluation_path", required=False)
    ground_truth_check_subparser.add_argument("--use_backup", action="store_true", help="use backup output directory.")
    ###

    args = parser.parse_args()

    # NOTE: The writing of best config for prompt_set 1 and the rest is fundamentally different.
    # I always select the best hp-config based on prompt_set 1, but replace the prompt examples to use
    # based on what prompt_set is passed. For prompt_set_1 I'll have __best suffix. For the rest, I'll have
    # __best_p1_to_p2, and __best_p1_to_p3 suffixes. I know the naming is a bit odd, but due to legacy reasons.

    config_filepath = get_config_file_path_from_name_or_path(args.experiment_name_or_path)
    base_config_name = os.path.splitext(os.path.split(config_filepath)[1])[0]
    hyperparameter_variations_directory = os.path.join("instantiated_configs")
    os.makedirs(hyperparameter_variations_directory, exist_ok=True)

    instantiation_scheme = instantiation_schemes[args.instantiation_scheme]

    with open(config_filepath, "r") as file:
        file_content = file.read().strip()

    _prompt_set = "1" if args.prompt_set == "aggregate" else args.prompt_set

    inferred_dataset = infer_dataset(file_content)
    valid_qids = dataset_to_prompt_set_to_qids[inferred_dataset][_prompt_set]
    variable_replacements = {}
    if valid_qids is not None:
        variable_replacements = {"valid_qids": json.dumps(valid_qids)}

    # First replace the prompt set
    names = [f"prompt_set_{_prompt_set}"] if valid_qids is not None else [""]
    file_content = instatiate_config(content=file_content, variable_replacements=variable_replacements)

    max_metric_value = float("-inf")
    best_config_file_path = None
    best_hyperparameters = None

    complete_experiment_names = []
    incomplete_experiment_names = []
    hyperparameter_metrics_data = []

    args_best_is_passed = hasattr(args, "best") and args.best
    

    if args.prompt_set == "aggregate":
        assert args.command == "summarize" and args_best_is_passed

    # Then institate scheme-based variable replacements and update them.
    local_file_contents = []

    for values in [e for e in itertools.product(*list(instantiation_scheme.values()))]:

        local_file_content = copy.deepcopy(file_content)

        variable_replacements = {}
        for key, value in zip(instantiation_scheme.keys(), values):
            variable_replacements[key] = copy.deepcopy(value)

        local_file_content = instatiate_config(content=local_file_content, variable_replacements=variable_replacements)

        local_names = copy.deepcopy(names)
        for key, value in variable_replacements.items():
            # TODO: assert value doesn't have non-alphanumeric chars. If so, remove them.
            value_name = value.replace('"', "")
            local_names.append(f"{key}__{value_name}")

        local_file_contents.append(local_file_content)
        overall_local_name = "___".join(local_names).lstrip("_")

        local_file_path = os.path.join(
            hyperparameter_variations_directory, "____".join([base_config_name, overall_local_name]) + ".jsonnet"
        )

        local_file_path_basename = os.path.basename(local_file_path)
        if len(local_file_path_basename) > 255:
            local_file_path = os.path.join(
                hyperparameter_variations_directory,
                os.path.splitext(local_file_path_basename)[0][:237]
                + "__"
                + hash_str(local_file_path_basename)
                + ".jsonnet",
            )

        experiment_name = os.path.splitext(os.path.split(local_file_path)[1])[0]

        prediction_directory = os.path.join("predictions", experiment_name)


        evaluation_path = args.evaluation_path if hasattr(args, "evaluation_path") else None

        if hasattr(args, "use_backup") and args.use_backup:
            prediction_directory += "__backup"
        metrics_file_path = ""  # o/w vscode pylance complains.
        if args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check") or args_best_is_passed:
            if evaluation_path is None:
                exit("Pass evaluation_path or set a default one in the hp_manager.")

            prediction_file_name = os.path.splitext(os.path.basename(evaluation_path))[0]
            prediction_file_name = infer_source_target_prefix(config_filepath, evaluation_path) + prediction_file_name
            prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")
            evaluation_file_name = os.path.splitext(os.path.split(evaluation_path)[1])[0]
            evaluation_file_name = infer_source_target_prefix(config_filepath, evaluation_path) + evaluation_file_name
            ground_truth_file_path = os.path.join(
                prediction_directory, "ground_truth__" + evaluation_file_name + ".json"
            )
            metrics_file_path = os.path.join(
                prediction_directory, "evaluation_metrics__" + evaluation_file_name + ".json"
            )

        if args.command == "ground_truth_check" and not args_best_is_passed:
            if not os.path.exists(ground_truth_file_path):
                print(f"ground_turth file_path {ground_truth_file_path} not found.", flush=True)
                continue
            with open(ground_truth_file_path, "r") as file:
                lines = file.readlines()
            print("".join(lines[:10]), flush=True)

        if args.command == "write" and not args.best:
            print(f"Writing in {local_file_path}")
            with open(local_file_path, "w") as file:
                file.write(local_file_content)

        if ((args.command == "diff") or (args.command == "write" and not args.no_diff)) and not args.best:
            print("\n")
            message = ">>> " + os.path.split(local_file_path)[1]
            print("-" * len(message))
            subprocess.call(f"colordiff {config_filepath} {local_file_path}", shell=True)

        elif args.command == "print" and not args.best:
            print(local_file_path)

        elif args.command == "backup" and not args_best_is_passed:
            assert not prediction_directory.endswith("__backup")
            original_prediction_directory = prediction_directory
            if not os.path.exists(original_prediction_directory):
                print("original prediction directory doesn't exist. Skipping backup.")
                continue
            backup_prediction_directory = original_prediction_directory + "__backup"
            if os.path.exists(backup_prediction_directory) and not args.force:
                exit(f"Backup {backup_prediction_directory} already exists. Remove it or pass force to replace it.")
            shutil.rmtree(backup_prediction_directory, ignore_errors=True)
            shutil.move(original_prediction_directory, backup_prediction_directory)
            print(f"Backing up {original_prediction_directory}")

        elif args.command == "recover_backup" and not args_best_is_passed:
            assert not prediction_directory.endswith("__backup")
            original_prediction_directory = prediction_directory
            backup_prediction_directory = original_prediction_directory + "__backup"
            if not os.path.exists(backup_prediction_directory):
                print("backup prediction directory doesn't exist. Skipping recovery.")
                continue
            if os.path.exists(original_prediction_directory) and not args.force:
                exit(f"Original {original_prediction_directory} already exists. Remove it or pass force to replace it.")
            shutil.rmtree(original_prediction_directory, ignore_errors=True)
            shutil.move(backup_prediction_directory, original_prediction_directory)
            print(f"Recovering backup {backup_prediction_directory}")

        elif args.command == "print_backup" and not args_best_is_passed:
            assert not prediction_directory.endswith("__backup")
            original_prediction_directory = prediction_directory
            backup_prediction_directory = original_prediction_directory + "__backup"
            if os.path.exists(backup_prediction_directory):
                print(backup_prediction_directory)

        elif args.command == "verify" and not args_best_is_passed:
            if not os.path.exists(local_file_path):
                raise Exception("Looks like the instantiated config is not available. Make sure to 'write' it first.")
            verify_config(local_file_path)

        elif args.command == "predict" and not args_best_is_passed:
            run_command = f"python predict.py {local_file_path} {evaluation_path} --set_name {args.set_name} --llm_port_num {args.llm_port_num}"

            if args.silent:
                run_command += " --silent"
            if args.variable_replacements:
                run_command += f" --variable-replacements '{args.variable_replacements}'"
            if args.force:
                run_command += " --force"

            if not args.skip_if_exists:
                print(run_command)
                subprocess.call(run_command, shell=True)
            else:
                if not is_experiment_complete(
                    local_file_path, prediction_file_path, metrics_file_path, args.variable_replacements
                ):
                    print(run_command)
                    subprocess.call(run_command, shell=True)

        elif args.command == "track" and not args_best_is_passed:
            if is_experiment_complete(
                local_file_path, prediction_file_path, metrics_file_path, args.variable_replacements
            ):
                complete_experiment_names.append(experiment_name)
            else:
                incomplete_experiment_names.append(experiment_name)

        elif args.command == "summarize" or args_best_is_passed:

            if os.path.exists(metrics_file_path):
                with open(metrics_file_path, "r") as file:
                    metrics = json.load(file)

                hyperparameter_metrics_datum = {key: value for key, value in variable_replacements.items()}
                if "para_recall" in metrics and "avg_predicted_paras" in metrics:
                    metric_value = "@".join(
                        [str(round(metrics["title_recall"] * 100, 1)), str(round(metrics["avg_predicted_titles"], 1))]
                    )
                    metric_value += " | "
                    metric_value += "@".join(
                        [str(round(metrics["para_recall"] * 100, 1)), str(round(metrics["avg_predicted_paras"], 1))]
                    )
                else:
                    metric_value = " | ".join(
                        [  # Note that "|" is used to identify the first metric if required.
                            str(round(metrics["f1"] * 100, 1)),  # First item will be used for knowing best HP.
                            str(round(metrics["precision"] * 100, 1)) if "precision" in metrics else "----",
                            str(round(metrics["recall"] * 100, 1)) if "recall" in metrics else "----",
                            str(metrics["count"]).rjust(5, " "),
                        ]
                    )
                hyperparameter_metrics_datum["metric_value"] = metric_value
                hyperparameter_metrics_datum["complete"] = is_experiment_complete(
                    local_file_path, prediction_file_path, metrics_file_path, args.variable_replacements
                )
                hyperparameter_metrics_data.append(hyperparameter_metrics_datum)
            else:

                if args.command == "write" and args.best and "prompt_set_1" in metrics_file_path:
                    exit("The best HP config can't be identified as all exps are not complete yet.")

                hyperparameter_metrics_datum = {key: value for key, value in variable_replacements.items()}
                hyperparameter_metrics_datum["metric_value"] = "n/a"
                hyperparameter_metrics_datum["complete"] = False
                hyperparameter_metrics_data.append(hyperparameter_metrics_datum)

            if args.command == "write" and args.best:

                if "prompt_set_1" not in metrics_file_path:
                    # the best HP is only to be set based on prompt_set_1
                    best_config_file_path = None
                    best_hyperparameters = None
                    continue

                metric_value = hyperparameter_metrics_datum["metric_value"]
                if isinstance(metric_value, str):
                    metric_value_ = [e.strip() for e in metric_value.split("|")][0]
                    if "@" in metric_value_:
                        metric_value_ = metric_value_.split("@")[0].strip()
                    try:
                        metric_value_ = float(metric_value_)
                    except:
                        metric_value_ = metric_value
                if not isinstance(metric_value_, (float, int)):
                    exit("The best HP config can't be identified as metric value is not a float/int.")
                if metric_value_ > max_metric_value:
                    best_config_file_path = local_file_path
                    best_hyperparameters = copy.deepcopy(hyperparameter_metrics_datum)
                    best_hyperparameters.pop("metric_value")
                    max_metric_value = metric_value_

        elif args.command == "evaluate" and not args_best_is_passed:
            run_command = f"python evaluate.py {local_file_path} {evaluation_path} --set_name {args.set_name} --llm_port_num {args.llm_port_num}"
            if os.path.exists(metrics_file_path) and args.skip_if_exists:
                print(f"Skipping as the metrics file already exists here: {metrics_file_path}.")
                continue
            if args.question_type_key_value is not None:
                run_command += f" --question-type-key-value {args.question_type_key_value}"
            if args.only_print:
                run_command += " --only-print"
            if args.official:
                run_command += " --official"
            print(run_command)
            subprocess.call(run_command, shell=True)

        elif args.command == "summarize" and not args_best_is_passed:
            pass

        elif args.command == "delete_predictions":
            print(f"Removing predictions for {experiment_name}")
            shutil.rmtree(prediction_directory, ignore_errors=True)

    if len(set(local_file_contents)) != len(local_file_contents):
        raise Exception("Looks like some of the HP variations didn't lead to different output file content.")

    if args.command == "track" and not args_best_is_passed:
        all_experiment_names = complete_experiment_names + incomplete_experiment_names
        completion_percentage = (
            len(complete_experiment_names) / len(all_experiment_names) if all_experiment_names else 0.0
        )
        print("\nComplete Experiment Names:\n--------------------------")
        print("\n".join(complete_experiment_names))
        print("\nInComplete Experiment Names:\n----------------------------")
        print("\n".join(incomplete_experiment_names))
        print("\nOverall Stats:\n--------------")
        print(
            f"{len(complete_experiment_names)} / {len(all_experiment_names)} " f"({completion_percentage}) completed."
        )

    if args.command == "summarize" and not args_best_is_passed:
        summarize_and_results(hyperparameter_metrics_data)

    if args_best_is_passed and str(args.prompt_set) in ("1", "2", "3"):
        source_target_prefix = infer_source_target_prefix(config_filepath, evaluation_path)

        source_best_experiment_name = "____".join(
            [base_config_name, args.instantiation_scheme, source_target_prefix + "best"]
        )
        if str(args.prompt_set) == "1":
            target_best_experiment_name = source_best_experiment_name
        else:
            target_best_experiment_name = "____".join(
                [base_config_name, args.instantiation_scheme, source_target_prefix + f"best_p1_to_p{args.prompt_set}"]
            )
        source_write_best_config_file_path = os.path.join(
            hyperparameter_variations_directory, source_best_experiment_name + ".jsonnet"
        )
        target_write_best_config_file_path = os.path.join(
            hyperparameter_variations_directory, target_best_experiment_name + ".jsonnet"
        )

    if args.command == "summarize" and args_best_is_passed and str(args.prompt_set) in ("1", "2", "3"):

        official_prefix = "official_" if args.official else ""
        metrics_file_path = os.path.join(
            "predictions",
            target_best_experiment_name,
            official_prefix + "evaluation_metrics__" + evaluation_file_name + ".json",
        )
        if not os.path.exists(metrics_file_path):
            metric_value = "n/a"
        else:
            with open(metrics_file_path, "r") as file:
                metrics = json.load(file)
            if "para_recall" in metrics and "avg_predicted_paras" in metrics:
                metric_value = "@".join(
                    [str(round(metrics["title_recall"] * 100, 1)), str(round(metrics["avg_predicted_titles"], 1))]
                )
                metric_value += " | "
                metric_value += "@".join(
                    [str(round(metrics["para_recall"] * 100, 1)), str(round(metrics["avg_predicted_paras"], 1))]
                )
            else:
                metric_value = " | ".join(
                    [  # Note that "|" is used to identify the first metric if required.
                        str(round(metrics["f1"] * 100, 1)),  # First item will be used for knowing best HP.
                        str(round(metrics["precision"] * 100, 1)) if "precision" in metrics else "----",
                        str(round(metrics["recall"] * 100, 1)) if "recall" in metrics else "----",
                        str(metrics["count"]).rjust(5, " "),
                    ]
                )
        prediction_file_name = os.path.splitext(os.path.basename(evaluation_path))[0]
        prediction_file_name = infer_source_target_prefix(config_filepath, evaluation_path) + prediction_file_name
        prediction_directory = os.path.join("predictions", target_best_experiment_name)
        prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")
        if not is_experiment_complete(
            target_write_best_config_file_path, prediction_file_path, metrics_file_path, args.variable_replacements
        ):
            metric_value = "** " + str(metric_value) + " **"
        print(f"Best score => {metric_value}")

    if args.command == "summarize" and args_best_is_passed and str(args.prompt_set) == "aggregate":

        source_target_prefix = infer_source_target_prefix(config_filepath, evaluation_path)

        prediction_file_name = os.path.splitext(os.path.basename(evaluation_path))[0]
        prediction_file_name = infer_source_target_prefix(config_filepath, evaluation_path) + prediction_file_name

        metric_values = []
        has_incomplete_experiment = False
        for _prompt_set in ("1", "2", "3"):

            if str(_prompt_set) == "1":
                target_best_experiment_name = "____".join(
                    [base_config_name, args.instantiation_scheme, source_target_prefix + "best"]
                )
            else:
                target_best_experiment_name = "____".join(
                    [base_config_name, args.instantiation_scheme, source_target_prefix + f"best_p1_to_p{_prompt_set}"]
                )

            target_write_best_config_file_path = os.path.join(
                hyperparameter_variations_directory, target_best_experiment_name + ".jsonnet"
            )

            official_prefix = "official_" if args.official else ""
            metrics_file_path = os.path.join(
                "predictions",
                target_best_experiment_name,
                official_prefix + "evaluation_metrics__" + evaluation_file_name + ".json",
            )

            if not os.path.exists(metrics_file_path):
                metric_value = "n/a"
            else:
                with open(metrics_file_path, "r") as file:
                    metrics = json.load(file)
                if "para_recall" in metrics and "avg_predicted_paras" in metrics:
                    metric_value = round(metrics["para_recall"] * 100, 1)
                else:
                    metric_value = round(metrics["f1"] * 100, 1)
            metric_values.append(metric_value)

            prediction_directory = os.path.join("predictions", target_best_experiment_name)
            prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")
            if not is_experiment_complete(
                target_write_best_config_file_path, prediction_file_path, metrics_file_path, args.variable_replacements
            ):
                has_incomplete_experiment = True

        assert len(metric_values) == 3
        if "n/a" in metric_values:
            metric_value = "n/a"
        else:
            mean = round(statistics.mean(metric_values), 1)
            std = round(statistics.stdev(metric_values), 1)
            metric_value = f"{mean} | {std}"

        if has_incomplete_experiment:
            metric_value = "** " + str(metric_value) + " **"

        print(f"Best score stats (mean|str) => {metric_value}   <<   {metric_values}")

    if args_best_is_passed and args.command == "write" and str(args.prompt_set) in ("1", "2", "3"):

        if str(args.prompt_set) == "1":
            print("Setting Best HP:")
            for key, value in best_hyperparameters.items():
                print(f"    {key} => {value}")
            print(f"Max metric value: {max_metric_value}\n")
            print("Copying best HP config file_path")
            print(f"    from: {best_config_file_path}")
            print(f"      to: {source_write_best_config_file_path}")
            shutil.copy(best_config_file_path, source_write_best_config_file_path)

        else:
            if not os.path.exists(source_write_best_config_file_path):
                exit("Please save the best hp config with the prompt_set_1 first.")
            assert best_hyperparameters is None
            assert best_config_file_path is None
            print(f"Copying Best HP for prompt-1 to prompt-{args.prompt_set} in:")
            print(target_write_best_config_file_path)
            variable_replacements = {"valid_qids": json.dumps(valid_qids)}
            with open(source_write_best_config_file_path, "r") as file:
                file_content = file.read().strip()
            file_content = instatiate_config(content=file_content, variable_replacements=variable_replacements)
            with open(target_write_best_config_file_path, "w") as file:
                file.write(file_content)

    # Most of the the code below is duplication, abstract it out into a function.
    if args_best_is_passed and args.command == "print" and str(args.prompt_set) in ("1", "2", "3"):
        print(f"Best HP filepath: {target_write_best_config_file_path}")

    if args_best_is_passed and args.command in ("predict", "evaluate") and str(args.prompt_set) in ("1", "2", "3"):
        if not os.path.exists(target_write_best_config_file_path):
            exit(f"Best HP filepath {target_write_best_config_file_path} not found.")

        prediction_directory = os.path.join("predictions", target_best_experiment_name)
        if hasattr(args, "use_backup") and args.use_backup:
            prediction_directory += "__backup"
        if evaluation_path is None:
            exit("Pass evaluation_path or set a default one in the hp_manager.")

        prediction_file_name = os.path.splitext(os.path.basename(evaluation_path))[0]
        prediction_file_name = infer_source_target_prefix(config_filepath, evaluation_path) + prediction_file_name
        prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")
        evaluation_file_name = os.path.splitext(os.path.split(evaluation_path)[1])[0]
        evaluation_file_name = infer_source_target_prefix(config_filepath, evaluation_path) + evaluation_file_name
        ground_truth_file_path = os.path.join(prediction_directory, "ground_truth__" + evaluation_file_name + ".json")
        official_prefix = "official_" if hasattr(args, "official") and args.official else ""
        metrics_file_path = os.path.join(
            prediction_directory, official_prefix + "evaluation_metrics__" + evaluation_file_name + ".json"
        )
        assert target_write_best_config_file_path is not None

        if args.command == "predict":

            run_command = f"python predict.py {target_write_best_config_file_path} {evaluation_path} --llm_port_num {args.llm_port_num}"

            if args.silent:
                run_command += " --silent"
            if args.variable_replacements:
                run_command += f" --variable-replacements '{args.variable_replacements}'"
            if args.force:
                run_command += " --force"

            if not args.skip_if_exists:
                print(run_command)
                subprocess.call(run_command, shell=True)
            else:
                if not is_experiment_complete(
                    local_file_path, prediction_file_path, metrics_file_path, args.variable_replacements
                ):
                    print(run_command)
                    subprocess.call(run_command, shell=True)

        if args.command == "evaluate":
            if os.path.exists(prediction_file_path):
                #import pdb; pdb.set_trace()
                run_command = f"python evaluate.py {target_write_best_config_file_path} {evaluation_path} --set_name {args.set_name} --llm_port_num {args.llm_port_num}"
                if os.path.exists(metrics_file_path) and args.skip_if_exists:
                    print(f"Skipping as the metrics file already exists here: {metrics_file_path}.")
                else:
                    if args.question_type_key_value is not None:
                        run_command += f" --question-type-key-value {args.question_type_key_value}"
                    if args.only_print:
                        run_command += " --only-print"
                    if args.official:
                        run_command += " --official"
                    print(run_command)
                    subprocess.call(run_command, shell=True)
            else:
                print(f"Best HP prediction path {prediction_file_path} is not available.")


if __name__ == "__main__":
    main()
