import json
import logging
import re
from json import JSONDecodeError

from commaqa.execution.model_executer import ModelExecutor

logger = logging.getLogger(__name__)


class MathModel(ModelExecutor):
    def __init__(self, **kwargs):
        self.func_regex = {
            "is_greater\((.+) \| (.+)\)": self.greater_than,
            "is_smaller\((.+) \| (.+)\)": self.smaller_than,
            "diff\((.+) \| (.+)\)": self.diff,
            "belongs_to\((.+) \| (.+)\)": self.belongs_to,
            "max\((.+)\)": self.max,
            "min\((.+)\)": self.min,
            "count\((.+)\)": self.count,
        }
        super(MathModel, self).__init__(**kwargs)

    @staticmethod
    def get_number(num):
        # can only extract numbers from strings
        if isinstance(num, int) or isinstance(num, float):
            return num
        if not isinstance(num, str):
            return None
        try:
            item = json.loads(num)
        except JSONDecodeError:
            logger.debug("Could not JSON parse: " + num)
            return None
        if isinstance(item, list):
            if (len(item)) != 1:
                logger.debug("List of values instead of single number in {}".format(num))
                return None
            item = item[0]
            if isinstance(item, list):
                logger.debug("Could not parse float from list within the list: {}".format(item))
                return None
        try:
            return float(item)
        except ValueError:
            logger.debug("Could not parse float from: " + item)
            return None

    def max(self, groups):
        if len(groups) != 1:
            raise ValueError("Incorrect regex for max. " "Did not find 1 group: {}".format(groups))
        try:
            entity = json.loads(groups[0])

            if isinstance(entity, list):
                numbers = []
                for x in entity:
                    num = MathModel.get_number(x)
                    if num is None:
                        if self.ignore_input_mismatch:
                            logger.debug("Cannot parse as number: {}".format(x))
                            return None, []
                        else:
                            raise ValueError("Cannot parse as number: {} in {}".format(x, entity))
                    numbers.append(num)
            else:
                logger.debug("max can only handle list of entities. Arg: " + str(entity))
                return None, []
        except JSONDecodeError:
            logger.error("Could not parse: {}".format(groups[0]))
            raise
        return max(numbers), []

    def min(self, groups):
        if len(groups) != 1:
            raise ValueError("Incorrect regex for min. " "Did not find 1 group: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                numbers = []
                for x in entity:
                    num = MathModel.get_number(x)
                    if num is None:
                        if self.ignore_input_mismatch:
                            logger.debug("Cannot parse as number: {}".format(x))
                            return None, []
                        else:
                            raise ValueError("Cannot parse as number: {} in {}".format(x, entity))
                    numbers.append(num)
            else:
                logger.debug("min can only handle list of entities. Arg: " + str(entity))
                return None, []
        except JSONDecodeError:
            logger.debug("Could not parse: {}".format(groups[0]))
            if self.ignore_input_mismatch:
                return None, []
            else:
                raise
        return min(numbers), []

    def count(self, groups):
        if len(groups) != 1:
            raise ValueError("Incorrect regex for max. " "Did not find 1 group: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                return len(entity), []
            else:
                logger.debug("count can only handle list of entities. Arg: " + str(entity))
                return None, []
        except JSONDecodeError:
            logger.debug("Could not parse: {}".format(groups[0]))
            if self.ignore_input_mismatch:
                return None, []
            else:
                raise

    def belongs_to(self, groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for belongs_to. " "Did not find 2 groups: {}".format(groups))
        try:
            entity = json.loads(groups[0])
            if isinstance(entity, list):
                if len(entity) > 1:
                    logger.debug("belongs_to can only handle single entity as 1st arg. Args:" + str(groups))
                    return None, []
                else:
                    entity = entity[0]
        except JSONDecodeError:
            entity = groups[0]
        try:
            ent_list = json.loads(groups[1])
        except JSONDecodeError:
            logger.debug("Could not JSON parse: " + groups[1])
            raise

        if not isinstance(ent_list, list):
            logger.debug("belongs_to can only handle lists as 2nd arg. Args:" + str(groups))
            return None, []
        if entity in ent_list:
            return "yes", []
        else:
            return "no", []

    def diff(self, groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for diff. " "Did not find 2 groups: {}".format(groups))
        num1 = MathModel.get_number(groups[0])
        num2 = MathModel.get_number(groups[1])
        if num1 is None or num2 is None:
            if self.ignore_input_mismatch:
                # can not compare with Nones
                return None, []
            else:
                raise ValueError("Cannot answer diff with {}".format(groups))
        if num2 > num1:
            return round(num2 - num1, 3), []
        else:
            return round(num1 - num2, 3), []

    def greater_than(self, groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for greater_than. " "Did not find 2 groups: {}".format(groups))
        num1 = MathModel.get_number(groups[0])
        num2 = MathModel.get_number(groups[1])
        if num1 is None or num2 is None:
            if self.ignore_input_mismatch:
                # can not compare with Nones
                return None, []
            else:
                raise ValueError("Cannot answer gt with {}".format(groups))
        if num1 > num2:
            return "yes", []
        else:
            return "no", []

    def smaller_than(self, groups):
        if len(groups) != 2:
            raise ValueError("Incorrect regex for smaller_than. " "Did not find 2 groups: {}".format(groups))
        num1 = MathModel.get_number(groups[0])
        num2 = MathModel.get_number(groups[1])
        if num1 is None or num2 is None:
            if self.ignore_input_mismatch:
                # can not compare with Nones
                return None, []
            else:
                raise ValueError("Cannot answer lt with {}".format(groups))
        if num1 < num2:
            return "yes", []
        else:
            return "no", []

    def ask_question_predicate(self, question_predicate):
        for regex, func in self.func_regex.items():
            m = re.match(regex, question_predicate)
            if m:
                return func(m.groups())
        raise ValueError("Could not parse: {}".format(question_predicate))
