import json
from typing import List
from distilabel.steps import Step, StepInput, GlobalStep
import numpy as np

class EvaluatePerplexity(Step):
    @property
    def inputs(self) -> List[str]:
        return ["logprobs"]
    
    @property
    def outputs(self) -> List[str]:
        return ["perplexity_score"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            results = []
            for row in batch:
                logprobs = row["logprobs"]
                n = len(logprobs)
                perplexity = np.exp(-np.sum(logprobs) / n)
                results.append(row | {"perplexity_score" : perplexity})
            yield results

class AggregateResultToJson(GlobalStep):
    filepath: str

    @property
    def inputs(self) -> List[str]:
        return ["perplexity_score"]
    
    def process(self, *inputs: StepInput):
        result_dict = dict() 
        total_perplexity = None
        n_inputs = 0
        for batch in inputs:
            for row in batch:
                n_inputs += 1
                total_perplexity += row["perplexity"]
            
        result_dict["avg_perplexity"] = total_perplexity / n_inputs
        with open(self.filepath + "/ablation_results.json") as f:
            json.dump(result_dict)
        yield inputs