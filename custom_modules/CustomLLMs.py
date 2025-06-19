import json
import os
from typing import Annotated, Any, Dict, List, Optional, TypeVar, Union

import concurrent
import boto3
import botocore
from dotenv import load_dotenv
from openai import OpenAI
from distilabel.steps import StepInput, GlobalStep

from pydantic import Field
import torch
import torch.nn.functional as F

from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"
RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]

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
        ).client('sagemaker-runtime',config=botocore.config.Config(read_timeout=120, connect_timeout=60))
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

class Qwen3Embedder(GlobalStep):
    modelName: RuntimeParameter[str] = "Qwen/Qwen3-Embedding-0.6B"
    _tokenizer: Any = None
    _model: Any = None
    max_length: RuntimeParameter[int] = 8192
    batch_size: RuntimeParameter[int] = 10

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.modelName, padding_side='left')
        self._model = AutoModel.from_pretrained(self.modelName)
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["text_to_embed"]

    @property
    def outputs(self) -> List[str]:
        return ["embedding"]
    
    def _last_token_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def process(self, *inputs: StepInput):
        inputs_flattened = []
        for batch in inputs:
            for row in batch:
                inputs_flattened.append(row)

        input_texts = [row["text_to_embed"] for row in inputs_flattened]

        results = []
        for i in tqdm(range(0, len(input_texts), self.batch_size), desc="Embedding progress"):
            if len(input_texts) - i < self.batch_size:
                batch_input_texts = input_texts[i:]
                batch_inputs_flattened = inputs_flattened[i:]
            else:
                batch_input_texts = input_texts[i:i+self.batch_size]
                batch_inputs_flattened = inputs_flattened[i:i+self.batch_size]
            
            # Tokenize the input texts
            batch_dict = self._tokenizer(
                batch_input_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict.to(self._model.device)
            outputs = self._model(**batch_dict)
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            for embedding, row in zip(embeddings, batch_inputs_flattened):
                results.append(row | {"embedding": embedding.tolist()})
        
        yield results

class Qwen3Reranker(GlobalStep):
    modelName: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: RuntimeParameter[int] = 8192
    k: RuntimeParameter[int] = 1
    _tokenizer: Any = None
    _model: Any = None
    _token_false_id: Any = None
    _token_true_id: Any = None
    _prefix_tokens: Any = None
    _suffix_tokens: Any = None
    

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.modelName, padding_side='left')
        self._model = AutoModel.from_pretrained(self.modelName)
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["query", "documents", "metadatas"]

    @property
    def outputs(self) -> List[str]:
        return ["query", "documents", "metadatas"]
    
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self._tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self._prefix_tokens + ele + self._suffix_tokens
        inputs = self._tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self._model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self._token_true_id]
        false_vector = batch_scores[:, self._token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def process(self, *inputs: StepInput):
        result = []
        for batch in inputs:
            for row in batch:
                pairs = [self.format_instruction(None, row["query"], doc) for doc in row["documents"]]
                inputs = self.process_inputs(pairs)
                scores = self.compute_logits(inputs)
                
                sortedData = sorted(zip(scores, row["documents"], row["metadatas"]), reverse=True)[:self.k]
                row["scores"] = [x[0] for x in sortedData]
                row["documents"] = [x[1] for x in sortedData]
                row["metadatas"] = [x[2] for x in sortedData]

                result.append(row)
        yield result