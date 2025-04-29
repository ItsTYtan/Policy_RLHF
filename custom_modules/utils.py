import json
from typing import List
from distilabel.steps import GlobalStep, StepInput, GeneratorStep
from pydantic import PrivateAttr

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

class FromJsonFile(GeneratorStep):
    filename: str
    filepath: str
    _file: any = PrivateAttr(None)

    def load(self):
        super().load()
        full_path = f"{self.filepath}/{self.filename}"
        with open(full_path, "r", encoding="utf-8") as f:
            self._file = json.load(f)  

    @property
    def outputs(self) -> List[str]:
        full_path = f"{self.filepath}/{self.filename}"
        with open(full_path, "r", encoding="utf-8") as f:
            lst = json.load(f)  
        return list(lst[0].keys())

    def process(self, offset: int = 0):
        if offset:
            self._file = self._file[offset:]
        
        while self._file:
            batch = self._file[: self.batch_size]
            self._file = self._file[self.batch_size :]
            yield(
                batch,
                True if len(self._file) == 0 else False,
            )