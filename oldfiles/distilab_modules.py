from collections import defaultdict
import json
import os
import re

from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING
import concurrent
from distilabel.steps import Step, StepInput, GlobalStep, GeneratorStep
from dotenv import load_dotenv
from openai import OpenAI
from templates.templates import answer_template_dict
from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, GeneratorStepOutput

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class FormatPolicyQuestionRAG(Step):
    persist_directory: Optional[str] = Field(default=None)
    collection_name: Optional[str] = Field(default=None)
    _chroma_client: Any = PrivateAttr()

    def load(self):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self._chroma_client = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name,
        )
        super().load()
        

    @property
    def inputs(self) -> List[str]:
        return ["question"]

    @property
    def outputs(self) -> List[str]:
        return ["question", "instruction", "context", "source"]


    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                question = row["question"]
                doc = self._chroma_client.similarity_search(question, k=1)[0]
                context = doc.page_content
                source = doc.metadata["source"]
                prompt = answer_template_dict["rag"].format(question=question, context=context)
                result.append({"question": question, "instruction": prompt, "context": context, "source": source})
            yield result

class FormatPolicyQuestion(Step):
    template: str

    @property
    def inputs(self) -> List[str]:
        return ["question"]

    @property
    def outputs(self) -> List[str]:
        return ["question", "instruction"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                question = row["question"]
                prompt = self.template.format(question=question)
                result.append({"question": question, "instruction": prompt})
            yield result

class ExtractPolicyAnswerRAG(Step):
    @property
    def inputs(self) -> List[str]:
        return ["question", "generations", "context", "source"]

    @property
    def outputs(self) -> List[str]:
        return ["question", "answers", "context", "source"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for pair in batch:
                answers = []
                for generation in pair["generations"]:
                    match = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
                    text = match.group(1) if match else ""
                    answers.append(text)
                result.append({"question": pair["question"], "answers": answers, "context": pair["context"], "source": pair["source"]})
            yield result

class ExtractPolicyAnswer(Step):
    method: str

    @property
    def inputs(self) -> List[str]:
        return ["question", "generations"]

    @property
    def outputs(self) -> List[str]:
        return ["question", "answers"]

    def process(self, *inputs: StepInput):
        for batch in inputs:
            result = []
            for pair in batch:
                answers = []
                for generation in pair["generations"]:
                    if (self.method == "direct"):
                        match1 = re.search(r"<answer1>(.*?)</answer1>", generation, re.DOTALL)
                        text1 = match1.group(1) if match1 else ""
                        match2 = re.search(r"<answer2>(.*?)</answer2>", generation, re.DOTALL)
                        text2 = match2.group(1) if match2 else ""
                        answers = [text1, text2]               
                result.append({"question": pair["question"], "answers": answers})
            yield result

class GeneratePolicyQuestion(GeneratorStep):
    politicalTopics: List[str]
    policyTemplate: str

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        if offset:
            self.politicalTopics = self.politicalTopics[offset:]

        while self.politicalTopics:
            batch = [
                {
                    "topic": topic,
                    "instruction": self.policyTemplate.format(topic=topic)
                } for topic in self.politicalTopics[: self.batch_size]
            ]
            self.politicalTopics = self.politicalTopics[self.batch_size :]
            yield (
                batch,
                True if len(self.politicalTopics) == 0 else False,
            )

    @property
    def outputs(self) -> "StepColumns":
        return ["topic", "instruction"]

class ExtractPolicyQuestion(Step):
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

class OpenRouterLLM(Step):
    _client: Any = None  # Will be set in load()
    model: str
    max_tokens: int
    temperature: float = 0.9
    system_prompt: str = "You are a helpful assistant."

    def load(self):
        load_dotenv()
        apikey = os.getenv("OPENROUTER_API_KEY") 
        baseurl = "https://openrouter.ai/api/v1"
        self._client = OpenAI(
            api_key=apikey,
            base_url=baseurl
        )
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    @property
    def outputs(self) -> List[str]:
        return ["generation"]

    def _call_api(self, prompt: str) -> str:
        """
        Synchronous wrapper around your chat completion call.
        Returns the generated text (or empty string on failure).
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(e)
            return ""

    def process(self, *inputs: StepInput):
        """
        For each input batch (an iterable of rows), runs all API calls in parallel
        using a thread pool, then yields the list of results.
        """
        for batch in inputs:
            # You can tune max_workers to suit your rate‑limits / CPU
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Schedule one future per row
                futures = {
                    executor.submit(self._call_api, row["instruction"]): row
                    for row in batch
                }

                results: List[Dict[str, str]] = []
                # As each finishes, collect its result
                for future in concurrent.futures.as_completed(futures):
                    text = future.result()
                    results.append({"generation": text})
            yield results


class ToJsonFile(GlobalStep):
    filename: str
    filepath: str

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
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