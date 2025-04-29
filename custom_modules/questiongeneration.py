import re
from typing import List, Optional
from distilabel.steps import Step, StepInput, GeneratorStep

class FormatTopic(GeneratorStep):
    topics: List[str]
    template: Optional[str] = None
        
    @property
    def outputs(self):
        return ["topic", "instruction"]

    def process(self, offset: int = 0):
        if offset:
            self.topics = self.topics[offset:]

        while self.topics:
            batch = [
                {
                    "topic": topic,
                    "instruction": topic if not self.template else self.template.format(topic=topic)
                } for topic in self.topics[: self.batch_size]
            ]
            self.topics = self.topics[self.batch_size :]
            yield (
                batch,
                True if len(self.topics) == 0 else False,
            )

class ExtractQuestion(Step):
    @property
    def inputs(self) -> List[str]:
        return ["topic", "generation", "model"]

    @property
    def outputs(self) -> List[str]:
        return ["topic", "question", "model"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for entry in batch:
                match = re.search(r"<questions>(.*?)</questions>", entry["generation"], re.DOTALL)
                text = match.group(1) if match else ""
                questions = text.splitlines()
                chunk = map(lambda question: {
                    "topic": entry["topic"],
                    "question": question,
                    "model": entry["model"],
                }, questions)
                for qnEntry in list(chunk):
                    if qnEntry["question"] == "":
                        continue
                    result.append(qnEntry)
            yield result 