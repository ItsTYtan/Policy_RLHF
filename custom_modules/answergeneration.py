from typing import List
from distilabel.steps import Step, StepInput

class FormatQuestion(Step):
    template: str
    guidelines: dict[str, str]

    @property
    def inputs(self) -> List[str]:
        return ["topic", "question"]

    @property
    def outputs(self) -> List[str]:
        return ["instruction"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                question = row["question"]
                guideline = self.guidelines[row["topic"]]
                prompt = self.template.format(question=question, guideline=guideline)
                row["instruction"] = prompt
                result.append(row)
            yield result