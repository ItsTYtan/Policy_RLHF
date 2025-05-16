import os
from typing import Any, Dict, List

import concurrent
from dotenv import load_dotenv
from openai import OpenAI
from distilabel.steps import StepInput, GlobalStep

class OpenRouterLLM(GlobalStep):
    _client: Any = None
    model: str
    max_tokens: int
    temperature: float = 0.9
    system_prompt: str = "You are a helpful assistant."
    max_workers: int = 100
    logprobs: bool = False

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
        if self.logprobs:
            return ["generation", "model_name", "logprobs"]
        else: 
            return ["generation", "model_name"]

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
                temperature=self.temperature,
                logprobs=self.logprobs
            )

            if self.logprobs:
                logprobsRaw = response.choices[0].logprobs.content
                logprobs = map(lambda completion: (completion.token, str(completion.logprob)), logprobsRaw)
                return response.choices[0].message.content or "", list(logprobs)

            return response.choices[0].message.content or "", None
            
        except Exception as e:
            print(e)
            return ""

    def process(self, *inputs: StepInput):
        """
        For each input batch (an iterable of rows), runs all API calls in parallel
        using a thread pool, then yields the list of results.
        """
        # You can tune max_workers to suit your rate‑limits / CPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Schedule one future per row
            futures = {
                executor.submit(self._call_api, row["instruction"]): row
                for batch in inputs
                for row in batch
            }

            results = []
            count = 0
            total = len(futures)
            # As each finishes, collect its result
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                text, logprobs = future.result()
                resultRow = row | {"generation": text, "model_name": self.model}
                if self.logprobs:
                    resultRow = resultRow | {"logprobs": logprobs}
                results.append(resultRow)
                count += 1
                if (count % 100 == 0):
                    print(str(count) + "/" + str(total) + " generated")
        yield results