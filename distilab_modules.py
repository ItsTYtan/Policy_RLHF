import os
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
)
from distilabel.models.llms import OpenAILLM, TransformersLLM
from distilabel.steps.tasks import TextGeneration, UltraFeedback

from typing import Any, List
from distilabel.steps import Step, StepInput
from templates import PROMPT_TEMPLATE_ANSWER
from pydantic import Field, PrivateAttr

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class FormatPolicyQuestionRAG(Step):
    persist_directory: str = Field(...)
    collection_name: str = Field(...)
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
        return ["topic", "question"]

    @property
    def outputs(self) -> List[str]:
        return ["topic", "instruction", "context", "source"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                question = row["question"]
                doc = self._chroma_client.similarity_search(question, k=1)[0]
                context = doc.page_content
                source = doc.metadata["source"]
                prompt = PROMPT_TEMPLATE_ANSWER.format(question=question, context=context)
                result.append({"topic": row["topic"], "instruction": prompt, "context": context, "source": source})
            yield result

class FormatPolicyQuestionNoRAG(Step):
    @property
    def inputs(self) -> List[str]:
        return ["topic", "question"]

    @property
    def outputs(self) -> List[str]:
        return ["topic", "instruction", "context", "source"]

    def process(self, *inputs: StepInput):
        for input in inputs:
            result = []
            for row in input:
                question = row["question"]
                prompt = PROMPT_TEMPLATE_ANSWER.format(question=question, context="")
                result.append({"topic": row["topic"], "instruction": prompt, "context": None, "source": None})
            yield result