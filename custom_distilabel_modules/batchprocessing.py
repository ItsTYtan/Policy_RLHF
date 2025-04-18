import asyncio
import os

from typing import Any, List, Optional, TYPE_CHECKING
from distilabel.steps import Step, StepInput, GlobalStep, GeneratorStep
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import Field, PrivateAttr

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
