from commaqa.configs.step_config import StepConfig
from commaqa.dataset.utils import get_predicate_args


class ModelQuestionConfig:
    def __init__(self, config_json):
        self.steps = [StepConfig(x) for x in config_json["steps"]] if "steps" in config_json else []
        self.questions = config_json.get("questions")
        self.init = config_json["init"]
        self.model = config_json["model"]
        self.predicate = config_json["predicate"]

    def to_json(self):
        return {
            "steps": [x.to_json() for x in self.steps],
            "questions": self.questions,
            "init": self.init,
            "model": self.model,
            "predicate": self.predicate,
        }


class PredicateLanguageConfig:
    def __init__(self, pred_lang_config):
        # import json
        # print(json.dumps(pred_lang_config, indent=2))
        self.predicate_config = {}
        self.model_config = {}
        for predicate, config in pred_lang_config.items():
            config["predicate"] = predicate
            question_config = ModelQuestionConfig(config)
            self.predicate_config[predicate] = question_config
            model = config["model"]
            if model not in self.model_config:
                self.model_config[model] = []
            self.model_config[model].append(question_config)

    def model_config_as_json(self):
        return {model: [config.to_json() for config in configs] for model, configs in self.model_config.items()}

    def find_model(self, question_predicate):
        matching_configs = self.find_valid_configs(question_predicate)
        if len(matching_configs) == 0:
            return None
        matching_models = {x.model for x in matching_configs}
        if len(matching_models) != 1:
            raise ValueError(
                "Unexpected number of matching models: {} for {}. "
                "Expected one model".format(matching_models, question_predicate)
            )
        return matching_models.pop()

    def find_valid_configs(self, question_predicate):
        qpred, qargs = get_predicate_args(question_predicate)
        matching_configs = []
        for key, config in self.predicate_config.items():
            config_qpred, config_qargs = get_predicate_args(key)
            if config_qpred == qpred:
                assert len(qargs) == len(config_qargs), "{} {}\n{}".format(qargs, config_qargs, question_predicate)
                mismatch = False
                for qarg, cqarg in zip(qargs, config_qargs):
                    if (cqarg == "?") ^ (qarg == "?"):
                        mismatch = True
                if not mismatch:
                    matching_configs.append(config)
        return matching_configs
