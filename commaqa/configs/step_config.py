class StepConfig:
    def __init__(self, step_json):
        self.operation = step_json["operation"]
        self.question = step_json["question"]
        self.answer = step_json["answer"]

    def to_json(self):
        return self.__dict__
