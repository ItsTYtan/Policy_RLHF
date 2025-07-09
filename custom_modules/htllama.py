import random
from typing import List
from distilabel.steps import Step, StepInput

from templates.htllama_templates import JETT_TEMPLATE

class FormatHtllamaQuestion(Step):
    template: str
    refinements: list[str]

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "output", "output2"]

    @property
    def outputs(self) -> List[str]:
        return ["original_instruction", "output", "output2", "prompt"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                instruction = row["instruction"]
                random.shuffle(self.refinements)
                refinements = '\n'.join(self.refinements[:3])
                prompt = self.template.format(
                    input=instruction,
                    refinements = refinements
                )
                del row["instruction"]
                row["original_instruction"] = instruction
                row["prompt"] = prompt
                result.append(row)
            yield result

class FormatHtllamaAnswer(Step):
    template: str

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "output", "output2"]

    @property
    def outputs(self) -> List[str]:
        return ["original_instruction", "output", "output2", "prompt"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                instruction = row["instruction"]
                output = row["output"]
                output2 = row["output2"]
                prompt = self.template.format(
                    instruction=instruction,
                    output=output,
                    output2=output2
                )
                del row["instruction"]
                row["original_instruction"] = instruction
                row["prompt"] = prompt
                result.append(row)
            yield result

class FormatJett(Step):
    @property
    def inputs(self) -> List[str]:
        return ["instruction", "output", "output2"]

    @property
    def outputs(self) -> List[str]:
        return ["instruction", "output", "output2", "text"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                text = JETT_TEMPLATE.format(
                    instruction=row["instruction"],
                    response=row["output"]
                )
                row["text"] = text
                result.append(row)
            yield result