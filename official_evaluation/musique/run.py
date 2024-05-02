#!/usr/bin/env python
import logging
import os
import sys

LEVEL = logging.INFO  # alternative: logging.DEBUG
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "allennlp"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "transformers", "src"))

from allennlp.common.util import import_module_and_submodules
import_module_and_submodules("allennlp_lib")

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position
import transformers

# Following for handling this issue: https://github.com/allenai/allennlp/issues/4847
# The solution is taken from https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# If the issue persists, increase 2048 and/or decrease max_instances_in_memory
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

if __name__ == "__main__":
    if "setting" not in os.environ:
        os.environ["setting"] = "full"
        print(f'Running in {os.environ["setting"]} setting.')
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    logging.info(f"Transformers version: {transformers.__version__}")
    logging.info(f"Transformers path: {transformers.__file__}")
    main(prog="allennlp")