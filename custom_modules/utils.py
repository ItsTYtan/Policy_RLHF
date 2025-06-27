import json
import re
import sqlite3
from typing import Any, Dict, List, Optional
from distilabel.steps import GlobalStep, StepInput, GeneratorStep, Step
from pydantic import PrivateAttr

def jsonlToJson(filepath):
    jsonl_data = []
    with open(filepath, "r", encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            jsonl_data.append(json.loads(line))
    
    return jsonl_data

def jsonToJsonl(filepath):
    with open(filepath, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

class ToJsonFile(GlobalStep):
    filename: str
    filepath: str
    jsonl: bool = False

    def process(self, inputs: StepInput): 
        ext = ".jsonl" if self.jsonl else ".json"
        full_path = f"{self.filepath}/{self.filename}" + ext
        with open(full_path, "w", encoding="utf-8") as f:
            if (self.jsonl):
                for input in inputs:
                    f.write(json.dumps(input) + "\n")
            else:
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
            self._file = jsonlToJson(full_path)
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
                if not match:
                    result.append(entry | {"json": {}})
                    continue
                result.append(entry | {"json": json.loads(match.group(1))})
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

class TemplateFormatter(Step):
    template: str
    template_inputs: list[str]

    @property
    def inputs(self) -> List[str]:
        return self.template_inputs

    @property
    def outputs(self) -> List[str]: 
        return ["instruction"]
    
    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                formatDict = dict()
                for template_input in self.template_inputs:
                    formatDict[template_input] = row[template_input]
                prompt = self.template.format(**formatDict)
                result.append(row | {"instruction": prompt})
            yield result

class FromDb(GeneratorStep):
    dbPath: str = "./db/axiom.db"
    sql: str
    _cursor: Any = None
    _columns: List[str] = []
        
    @property
    def outputs(self):
        if not self._columns:
            # open, execute & read description just once
            conn = sqlite3.connect(self.dbPath)
            cur  = conn.cursor()
            cur.execute(self.sql)
            self._columns = [d[0] for d in cur.description]
            conn.close()
        return self._columns

    def process(self, offset: int = 0):
        conn = sqlite3.connect(self.dbPath)
        self._cursor = conn.cursor()
        self._cursor.execute(self.sql)
        self._columns = [desc[0] for desc in self._cursor.description]
        if offset:
            self._cursor.fetchmany(offset)
        data = self._cursor.fetchmany(self.batch_size)
        while data:
            batch = []
            for entry in data:
                batch.append(
                    dict(zip(self._columns, entry))
                )
            data = self._cursor.fetchmany(self.batch_size)                        
            yield (
                batch,
                False if data else True,
            )
        conn.close()

class GeneralSqlExecutor(Step):
    dbPath: str = "./db/axiom.db"
    sql_template: str
    sql_inputs: list[str]
    output_columns: list[str] = []
        
    @property
    def outputs(self):
        return self.output_columns

    def process(self, *inputs: StepInput):
        conn = sqlite3.connect(self.dbPath)
        cursor = conn.cursor()
        for input in inputs:
            result = []
            for row in input:
                formatDict = dict()
                for template_input in self.sql_inputs:
                    formatDict[template_input] = row[template_input]
                sql = self.sql_template.format(**formatDict)
                cursor.execute(sql)
                db_result = cursor.fetchall()
                db_result_dict = {
                    col: [row[i] for row in db_result]
                    for i, col in enumerate(self.output_columns)
                }
                result.append(row | db_result_dict)
            yield result
        conn.close()

# class AddToTable(GlobalStep):
#     db_path: str = "./db/axiom.db"
#     tableName: str
#     columns_to_add: list[str]

#     extendSchemaSql: str
#     batchUpdateSql: str

#     @property
#     def inputs(self) -> List[str]:
#         return self.columns_to_add

#     def process(self, inputs: StepInput): 
#         conn = sqlite3.connect(self.dbPath)
#         cursor = conn.cursor()
#         cursor.execute(self.extendSchemaSql)

#         updates = []
#         for input in inputs:
#             update = []
#             for column in self.columns_to_add:

#         cursor.executemany(self.batchUpdateSql, )
#         for input in inputs:

#         yield inputs  