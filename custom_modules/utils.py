import json
from distilabel.steps import GlobalStep, StepInput

class ToJsonFile(GlobalStep):
    filename: str
    filepath: str

    def process(self, inputs: StepInput): 
        full_path = f"{self.filepath}/{self.filename}"
        with open(full_path, "w", encoding="utf-8") as f:
            obj = []
            for input in inputs:
                record = {}
                for key, value in input.items():
                    record[key] = value
                obj.append(record)
            json.dump(obj, f, ensure_ascii=False, indent=2)
        yield inputs