from math import inf
from typing import List
from distilabel.steps import StepInput, Step

class EvaluateLogprobs(Step):
    @property
    def inputs(self) -> List[str]:
        return ["logprobs"]

    @property
    def outputs(self) -> List[str]:
        return ["avg-logprob", "max-token", "max-logprob", "min-token", "min-logprob"]
    
    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                logprobs = row["logprobs"]
                maxtoken = ""
                mintoken = ""
                maxlogprob = -inf
                minlogprob = inf
                totallogprob = 0
                for logprob in logprobs:
                    token = logprob[0]
                    value = float(logprob[1])
                    totallogprob += value
                    if (value > maxlogprob):
                        maxtoken = token
                        maxlogprob = value
                    if (value < minlogprob):
                        mintoken = token
                        minlogprob = value
                avglogprob = totallogprob / len(logprobs)
                result.append(row |
                    {
                        "avg-logprob": str(avglogprob),
                        "max-token": maxtoken,
                        "max-logprob": str(maxlogprob),
                        "min-token": mintoken,
                        "min-logprob": str(minlogprob)
                    }
                )
            yield result

