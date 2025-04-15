import asyncio
import os
import re

from typing import Any, List, Optional, TYPE_CHECKING
from distilabel.steps import Step, StepInput, GlobalStep, GeneratorStep
from dotenv import load_dotenv
from openai import AsyncOpenAI
from templates import PROMPT_TEMPLATE_ANSWER
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
                prompt = PROMPT_TEMPLATE_ANSWER.format(question=question, context=context)
                result.append({"question": question, "instruction": prompt, "context": context, "source": source})
            yield result

class FormatPolicyQuestionNoRAG(Step):
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
                prompt = PROMPT_TEMPLATE_ANSWER.format(question=question, context="")
                result.append({"question": question, "instruction": prompt})
            yield result

class ExtractPolicyAnswer(Step):
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
                    match = re.search(r"<neutral>(.*?)</neutral>", generation, re.DOTALL)
                    text = match.group(1) if match else ""
                    answers.append(text)
                result.append({"question": pair["question"], "answers": answers, "context": pair["context"], "source": pair["source"]})
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

    def load(self):
        load_dotenv()
        apikey = os.getenv("OPENROUTER_API_KEY") 
        baseurl = "https://openrouter.ai/api/v1"
        # Initialize the AsyncOpenAI client with OpenRouter base URL
        # (Assuming AsyncOpenAI is imported from a relevant library)
        self._client = AsyncOpenAI(
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

    def process(self, *inputs: StepInput):
        for input in inputs:
            # Define an asynchronous function to run tasks concurrently
            async def process_batch():
                tasks = []
                for row in input:
                    prompt = row["instruction"]
                    # Prepare the API call for each prompt
                    task = self._client.chat.completions.create(
                        model="qwen/qwen2.5-vl-72b-instruct",  # Replace with your desired model
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1024,
                        temperature=0.7
                    )
                    tasks.append(task)
                # Gather all tasks concurrently
                responses = await asyncio.gather(*tasks)
                results = []
                for res in responses:
                    content = res.choices[0].message.content
                    resStr = content if content else ""
                    results.append({
                        "generation": resStr
                    })
                return results

            # Run the asynchronous batch process (this uses asyncio.run to start the event loop)
            batch_results = asyncio.run(process_batch())
            yield batch_results
