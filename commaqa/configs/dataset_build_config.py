from commaqa.configs.entities_config import EntitiesConfig
from commaqa.configs.predicate_config import PredicateConfig
from commaqa.configs.predicate_language_config import PredicateLanguageConfig
from commaqa.configs.theory_config import TheoryConfig


class DatasetBuildConfig:
    def __init__(self, input_json):
        self.version = input_json["version"]
        self.entities = EntitiesConfig(input_json["entities"])
        self.predicates = [PredicateConfig(x) for x in input_json["predicates"].items()]
        self.theories = [TheoryConfig(x) for x in input_json["theories"]]
        self.pred_lang_config = PredicateLanguageConfig(input_json["predicate_language"])
