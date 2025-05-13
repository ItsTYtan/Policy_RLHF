from typing import List
from distilabel.steps import Step, StepInput

class Formathtllama(Step):
    template: str

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "output", "output2", "text"]

    @property
    def outputs(self) -> List[str]:
        return ["prompt"]

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
                row["prompt"] = prompt
                result.append(row)
            yield result