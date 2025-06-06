import json
import math
import os
import random
import re
from typing import Dict, List, Optional
from distilabel.steps import Step, StepInput, GeneratorStep

class LoadHansard(GeneratorStep):
    hansard_filepath: str
    num_examples: Optional[int] = None
    _filepaths: list[str] = []
    max_length: int = math.inf

    def load(self):
        for filename in os.listdir(self.hansard_filepath):
            full_path = os.path.join(self.hansard_filepath, filename)
            try: 
                with open(full_path, "r") as f:
                    rawText = json.load(f)["text"]
                    if not rawText:
                        continue
                    if len(rawText) < self.max_length:
                        self._filepaths.append(full_path)
            except:
                continue
        self._filepaths = sorted(self._filepaths, reverse=True)[: self.num_examples]
        if not self.num_examples:
            self.num_examples = len(self._filepaths)
        super().load()

    @property
    def outputs(self):
        return ["file", "length", "hansard"]

    def process(self, offset: int = 0):
        if offset:
            self._filepaths = self._filepaths[offset:]
        while self._filepaths:
            batch = []
            for file in self._filepaths[: self.batch_size]:
                with open(file, "r") as f:
                    rawText = json.load(f)["text"]
                    batch.append({"file": file, "length": len(rawText), "hansard": rawText})
            self._filepaths = self._filepaths[self.batch_size :]
            yield (
                batch,
                True if len(self._filepaths) == 0 else False,
            )

class FormatPolicyExtract(Step):
    template: str

    @property
    def inputs(self) -> List[str]:
        return ["hansard"]
    
    @property
    def outputs(self) -> List[str]:
        return ["instruction"]
    
    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                input = row["hansard"]
                prompt = self.template.format(
                    hansard=input,
                )
                result.append(row | {"instruction": prompt})
            yield result

class FormatDecisionExtract(Step):
    template: str

    @property
    def inputs(self) -> List[str]:
        return ["policy", "hansard"]
    
    @property
    def outputs(self) -> List[str]:
        return ["instruction"]
    
    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                policy = row["policy"]
                hansard = row["hansard"]
                prompt = self.template.format(
                    policy=policy,
                    hansard=hansard,
                )
                result.append(row | {"instruction": prompt})
            yield result

