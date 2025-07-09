import json
import math
import os
import random
import re
import sqlite3
from typing import Any, Dict, List, Optional
import chromadb
from distilabel.steps import Step, StepInput, GeneratorStep
from dotenv import load_dotenv

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

class LoadHansardSections(GeneratorStep):
    hansard_filepath: str
    num_examples: Optional[int] = None
    _filepaths: list[str] = []

    def load(self):
        for filename in os.listdir(self.hansard_filepath):
            full_path = os.path.join(self.hansard_filepath, filename)
            try: 
                with open(full_path, "r") as f:
                    data = json.load(f)
                    if data:
                        self._filepaths.append(full_path)
            except Exception as e:
                print(e)
                continue
        self._filepaths = sorted(self._filepaths, reverse=True)[: self.num_examples]
        if not self.num_examples:
            self.num_examples = len(self._filepaths)
        super().load()

    @property
    def outputs(self):
        return ["file", "section_title", "content"]

    def process(self, offset: int = 0):
        if offset:
            self._filepaths = self._filepaths[offset:]
        while self._filepaths:
            batch = []
            for file in self._filepaths[: self.batch_size]:
                with open(file, "r") as f:
                    data = json.load(f)
                    for section in data[1:]:
                        batch.append({
                            "file": os.path.basename(file), 
                            "section_title": section["title"], 
                            "content": section["content"],
                        })
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

class ExtractSpeaker(Step):
    mpListFilePath: str
    _mps: list[str]

    def load(self):
        with open(self.mpListFilePath, "r") as f:
            data = json.load(f)
            self._mps = data
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["content"]
    
    @property
    def outputs(self) -> List[str]:
        return ["speaker", "speech"]
    
    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                content = row["content"]
                for mp in self._mps:
                    match1 = re.search(rf"{mp}\n:.*?\n[^\d]*?\n:", content, re.DOTALL)
                    match2 = re.search(rf"\({mp}\)[^\d]*?\n:.*?\n[^\d]*?\n:", content, re.DOTALL)
                    if match1:
                        result.append(row | {"speaker": mp, "speech": match1.group()})
                    if match2:
                        result.append(row | {"speaker": mp, "speech": match2.group()})
                    if not match1 and not match2:
                        matchEOF1 = re.search(rf"{mp}\n:.*", content, re.DOTALL)
                        matchEOF2 = re.search(rf"\({mp}\)[^\d]*?\n:.*", content, re.DOTALL)
                        if matchEOF1 and matchEOF2:
                            print("something wrong w ur regex ah", matchEOF1.group(), matchEOF2.group())
                        if matchEOF1:
                            result.append(row | {"speaker": mp, "speech": matchEOF1.group()})
                        if matchEOF2:
                            result.append(row | {"speaker": mp, "speech": matchEOF2.group()})
            yield result

# takes in the id for the speeches table and the id for the dataset table
class FormatInContextRAG(Step):
    template: str

    _dbPath: str = "/home/tytan216/volume/tzeyoung/Policy_RLHF/db/axiom.db"
    _sql_template_speeches: str = '''
        SELECT speech
        FROM speeches s
        WHERE id = ?
    '''
    _sql_template_dataset: str = '''
        SELECT question, generation
        FROM dataset d
        WHERE id = ?
    '''

    @property
    def inputs(self) -> List[str]:
        return ["dataset_id", "speeches_id"]
    
    @property
    def outputs(self) -> List[str]:
        return ["instruction", "question", "answer", "context"]
    
    def process(self, *inputs: StepInput):
        conn = sqlite3.connect(self._dbPath)
        cursor = conn.cursor()
        for batch in inputs:
            result = []
            for row in batch:
                dataset_id = int(row["dataset_id"])
                cursor.execute(self._sql_template_dataset, (dataset_id,))
                question, generation = cursor.fetchone()

                speeches_id = int(row["speeches_id"])
                cursor.execute(self._sql_template_speeches, (speeches_id,))
                (speech,) = cursor.fetchone()

                instruction = self.template.format(
                    question=question,
                    generation=generation,
                    context=speech
                )
                result.append(row | {
                    "instruction": instruction,
                    "question": question,
                    "answer": generation,
                    "context": speech
                })
            yield result

# takes in the id for the speeches table and the id for the dataset table
class ExpandClaims(Step):
    @property
    def inputs(self) -> List[str]:
        return ["speeches_ids", "claims"]
    
    @property
    def outputs(self) -> List[str]:
        return ["speeches_ids_expanded", "claims_expanded"]
    
    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for row in batch:
                speeches_ids = [int(x.strip()) for x in row["speeches_ids"].split(",")]
                claimsLstStr = row["claims"]
                speeches_ids_lst = []
                claimsLst = []
                for id, claims in zip(speeches_ids, claimsLstStr):
                    claimsarr = json.loads(claims)
                    if not claimsarr:
                        continue
                    speeches_ids_lst.extend([id] * len(claimsarr))
                    claimsLst.extend(claimsarr)
                result.append(row | {
                    "speeches_ids_expanded": speeches_ids_lst,
                    "claims_expanded": claimsLst
                })
            yield result