import json
import re
from typing import Dict, List, Optional
from distilabel.steps import GlobalStep, StepInput, GeneratorStep, Step
from pydantic import PrivateAttr

def jsonlToJson(filepath):
    jsonl_data = []
    with open(filepath, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            jsonl_data.append(json.loads(line))
    
    return jsonl_data

def jsonToJsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

class ToJsonFile(GlobalStep):
    filename: str
    filepath: str
    jsonl: bool = False

    def process(self, inputs: StepInput): 
        full_path = f"{self.filepath}/{self.filename}"
        if (self.jsonl):
            yield jsonToJsonl(full_path)
        else:
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
    _file: any = PrivateAttr(List)
    startIdx: Optional[int] = None
    endIdx: Optional[int] = None

    def load(self):
        super().load()

        full_path = f"{self.filepath}/{self.filename}"
        if self.filename.endswith(".jsonl"):
            self._file = self.jsonlToJson(full_path)
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                self._file = json.load(f) 
        startIdx = self.startIdx if self.startIdx else 0
        endIdx = self.endIdx if self.endIdx else len(self._file)
        self._file = self._file[startIdx: endIdx]

    @property
    def outputs(self) -> List[str]:
        full_path = f"{self.filepath}/{self.filename}"
        if self.filename.endswith(".jsonl"):
            lst = jsonlToJson(full_path)
        else:
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

class FormatSFT(Step):
    system_prompt: str

    @property
    def inputs(self):
        return ["instructions", "generations"]
    
    @property
    def outputs(self):
        return ["messages"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for row in batch:
                messages = [{"role": "system", "content": self.system_prompt}]
                instructions = row["instructions"]
                generations = row["generations"]
                for instruction, generation in zip(instructions, generations):
                    messages.extend([
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": generation}
                    ])
                row["messages"] = messages
                result.append(row)
            yield result

class AddColumns(Step):
    columnDict: Dict[str, str]
    
    @property
    def outputs(self):
        return list(self.columnDict.keys())

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for row in batch:
                result.append(row | self.columnDict)
            yield result    

class PolicyDPOtoSFT(Step):
    @property
    def inputs(self):
        return ["question", "dpo_response_type", "generation"]
    
    @property
    def outputs(self):
        return ["instruction", "generation"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for row in batch:
                resTypes = row["dpo_response_type"]
                generations = row["generation"]
                for type, gen in zip(resTypes, generations):
                    if (type == "ok-response"):
                        result.append({"instruction": row["question"], "generation": gen})
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

class ExtractJson(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generation"]

    @property
    def outputs(self) -> List[str]: 
        return ["json"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for entry in batch:
                match = re.search(r"```json\s*(\{.*?\})\s*```", entry["generation"], re.DOTALL)
                jsonStr = match.group(1) if match else ""
                result.append(entry | {"json": json.loads(jsonStr)})
            yield result 

class ExtractPythonArray(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generation"]

    @property
    def outputs(self) -> List[str]: 
        return ["array"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for entry in batch:
                match = re.search(r"\[\s*([\s\S]*?)\s*\]", entry["generation"], re.DOTALL)
                arr = "[" + (match.group(1) if match else "") + "]"
                result.append(entry | {"array": json.loads(arr)})
            yield result 