import json
import os
from typing import Any, Dict, List, Optional

import concurrent
import boto3
from dotenv import load_dotenv
from openai import OpenAI
from distilabel.steps import StepInput, GlobalStep

class OpenRouterLLM(GlobalStep):
    _client: Any = None
    model: str
    max_tokens: int
    temperature: float = 0.9
    system_prompt: Optional[str] = None
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
            if self.system_prompt:
                msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt}
                ]
            else:
                msgs = [
                    {"role": "user",   "content": prompt}
                ]
            response = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
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

class SageMakerLLM(GlobalStep):
    _client: Any = None
    model: str
    max_tokens: int
    temperature: float = 0.7
    system_prompt: Optional[str] = None # Not implemented in this class
    max_workers: int = 100
    logprobs: bool = False # Not implemented in this class

    def load(self):
        load_dotenv()
        self._client = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='ap-southeast-1'
        ).client('sagemaker-runtime')
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    @property
    def outputs(self) -> List[str]:
        return ["generation", "model_name"]

    def _call_api(self, prompt: str) -> str:
        """
        Synchronous wrapper around your chat completion call.
        Returns the generated text (or empty string on failure).
        """
        try:
            if self.system_prompt:
                msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt}
                ]
            else:
                msgs = [
                    {"role": "user",   "content": prompt}
                ]
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "return_full_text": False,
                    "repetition_penalty": 1.1,
                }
            }

            payload_json = json.dumps(payload)
            response = self._client.invoke_endpoint(
                EndpointName=self.model,
                ContentType='application/json',
                Body=payload_json
            )
            response_body = json.loads(response['Body'].read().decode('utf-8'))
            return response_body.get('generated_text', 'No text found')
            
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
                text = future.result()
                resultRow = row | {"generation": text, "model_name": self.model}
                results.append(resultRow)
                count += 1
                if (count % 100 == 0):
                    print(str(count) + "/" + str(total) + " generated")
        yield results