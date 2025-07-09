import random
import re
from typing import Dict, List, Optional
from distilabel.steps import Step, StepInput, GeneratorStep

class FromTopicArray(GeneratorStep):
    topics: List[str]
    
    @property
    def outputs(self):
        return ["topic"]

    def process(self, offset: int = 0):
        if offset:
            self.topics = self.topics[offset:]
        while self.topics:
            batch = [
                { "topic": topic } for topic in self.topics[: self.batch_size]
            ]
            self.topics = self.topics[self.batch_size :]
            yield (
                batch,
                True if len(self.topics) == 0 else False,
            )

class TopicToPrompt(Step):
    template: str
    questionTypes: Dict[str, List[str]]
    questionPhrasings: List[str]
    phrasingSelectProb: float

    @property
    def inputs(self):
        return ["topic"]
    
    @property
    def outputs(self):
        return ["topic", "instruction", "question_type", "question_phrasings"]
    
    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for row in batch:
                for category, _types in self.questionTypes.items():
                    for _type in _types:
                        phrasings = ""
                        for phrasing in self.questionPhrasings:
                            if random.random() < self.phrasingSelectProb:
                                phrasings = phrasings + phrasing + "\n"
                        instruction = self.template.format(
                            topic=row["topic"],
                            category=category,
                            type=_type,
                            phrasings=phrasings
                        )
                        result.append(row | {
                            "instruction": instruction,
                            "question_type": _type,
                            "question_phrasings": phrasings
                        })
        yield result

class Extract(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generation"]

    @property
    def outputs(self) -> List[str]: 
        return ["extract"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for entry in batch:
                match = re.search(r"<extract>(.*?)</extract>", entry["generation"], re.DOTALL)
                text = match.group(1) if match else ""
                extract = text.splitlines()
                chunk = map(lambda question: {
                    "extract": question,
                }, extract)
                for qnEntry in list(chunk):
                    if qnEntry["extract"] == "":
                        continue
                    result.append(entry | qnEntry)
            yield result 