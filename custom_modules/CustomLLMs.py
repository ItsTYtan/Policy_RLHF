import json
import os
import subprocess
import time
from typing import Annotated, Any, Dict, List, Optional, TypeVar, Union
from huggingface_hub import login
import os
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
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from vllm import LLM, SamplingParams
import math
from vllm.inputs.data import TokensPrompt

_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"
RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]

class OpenRouterLLM(GlobalStep):
    model: str
    max_tokens: int
    temperature: float = 0.9
    max_workers: int = 100
    logprobs: bool = False

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
        load_dotenv()
        apikey = os.getenv("OPENROUTER_API_KEY") 
        baseurl = "https://openrouter.ai/api/v1"
        client = OpenAI(
            api_key=apikey,
            base_url=baseurl
        )

        """
        Synchronous wrapper around your chat completion call.
        Returns the generated text (or empty string on failure).
        """
        try:
            msgs = [
                {"role": "user",   "content": prompt}
            ]
            response = client.chat.completions.create(
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
            # As each finishes, collect its result
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Data generated", total=len(futures)):
                row = futures[future]
                text, logprobs = future.result()
                resultRow = row | {"generation": text, "model_name": self.model}
                if self.logprobs:
                    resultRow = resultRow | {"logprobs": logprobs}
                results.append(resultRow)
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
            # As each finishes, collect its result
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Data generated", total=len(futures)):
                row = futures[future]
                text = future.result()
                resultRow = row | {"generation": text, "model_name": self.model}
                results.append(resultRow)
        yield results

class Qwen3Embedder(GlobalStep):
    model: RuntimeParameter[str] = "Qwen/Qwen3-Embedding-8B"
    _tokenizer: Any = None
    _model: Any = None
    max_length: RuntimeParameter[int] = 8192
    batch_size: RuntimeParameter[int] = 10

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, padding_side='left')
        self._model = AutoModel.from_pretrained(self.model)
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

class Qwen3Embeddervllm(GlobalStep):
    model: str ="Qwen/Qwen3-Embedding-8B"
    max_workers: int = 100

    def load(self):
        # Create the log directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
        
        # Define log file paths
        stdout_log_file = os.path.join(log_dir, "vllm_stdout.log")
        stderr_log_file = os.path.join(log_dir, "vllm_stderr.log")

        # Run the vllm server and redirect logs to files
        with open(stdout_log_file, 'w') as stdout_file, open(stderr_log_file, 'w') as stderr_file:
            command = [
                "vllm", "serve", self.model,
                "--dtype", "auto",
                "--api-key", "token-abc123",
                "--gpu-memory-utilization", "0.4",
                "--task", "embed"
            ]
            
            # Start the vllm server process
            subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)

        load_dotenv()
        login(os.getenv("HUGGINGFACE_TOKEN"))
        apikey = "token-abc123" 
        baseurl = "http://localhost:8000/v1"
        test_client = OpenAI(
            api_key=apikey,
            base_url=baseurl
        )

        # Periodically check server status (ping)
        while True:
            print("checking server status...")

            try:
                # Make the request to check if server is ready
                response = test_client.embeddings.create(
                    model=self.model,
                    input="test",
                )
                
                # Safely check if the response is None or empty
                if response is not None and response != {} and response != []:
                    print("vLLM server started!")
                    break
            
            except Exception as e:
                time.sleep(5)
                continue
            
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["text_to_embed"]

    @property
    def outputs(self) -> List[str]:
        return ["embedding"]

    def _call_api(self, prompt: str) -> str:
        apikey = "token-abc123" 
        baseurl = "http://localhost:8000/v1"
        client = OpenAI(
            api_key=apikey,
            base_url=baseurl
        )
        """
        Synchronous wrapper around your chat completion call.
        Returns the generated text (or empty string on failure).
        """
        try:
            response = client.embeddings.create(
                model=self.model,
                input=prompt,
            )

            return response.data[0].embedding
            
        except Exception as e:
            print(e)
            print(response)
            return []
        
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def process(self, *inputs: StepInput):
        """
        For each input batch (an iterable of rows), runs all API calls in parallel
        using a thread pool, then yields the list of results.
        """
        # You can tune max_workers to suit your rate‑limits / CPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Schedule one future per row
            futures = {
                executor.submit(self._call_api, self.get_detailed_instruct('Given a web search query, retrieve relevant passages that answer the query', row["text_to_embed"])): row
                for batch in inputs
                for row in batch
            }

            results = []
            # As each finishes, collect its result
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Data generated", total=len(futures)):
                row = futures[future]
                text = future.result()
                resultRow = row | {"embedding": text}
                results.append(resultRow)
        yield results

class Qwen3Reranker(GlobalStep):
    modelName: str = "Qwen/Qwen3-Reranker-8B"
    max_length: int = 8192
    k: int = 1
    _tokenizer: Any = None
    _model: Any = None
    _token_false_id: Any = None
    _token_true_id: Any = None
    _prefix_tokens: Any = None
    _suffix_tokens: Any = None
    

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.modelName, padding_side='left')
        self._model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["query", "documents", "ids"]

    @property
    def outputs(self) -> List[str]:
        return ["query", "documents", "ids"]
    
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
        flattenedRows = [row for batch in inputs for row in batch]
        for row in tqdm(flattenedRows, desc="Raranking batches"):
            docs = row["documents"]
            ids = row["ids"]
            pairs = [self.format_instruction(None, row["query"], doc) for doc in docs]
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
            
            sortedData = sorted(zip(scores, docs, ids), reverse=True)[:self.k]
            row["documents"] = [x[1] for x in sortedData]
            row["ids"] = [x[2] for x in sortedData]

            result.append(row)
        yield result

class Qwen3Rerankervllm(GlobalStep):
    modelName: str = "Qwen/Qwen3-Reranker-8B"
    max_length: int = 8192
    k: int = 1
    
    def load(self):
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["query", "documents", "ids"]

    @property
    def outputs(self) -> List[str]:
        return ["query", "documents", "ids"]
    
    def format_instruction(self, instruction, query, doc):
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text


    def process_inputs(self, pairs, instruction, max_length, suffix_tokens, tokenizer):
        messages = [self.format_instruction(instruction, query, doc) for query, doc in pairs]
        messages = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:max_length] + suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        return messages

    @torch.no_grad()
    def compute_logits(self, model, inputs, sampling_params, true_token, false_token):
        outputs = model.generate(inputs, sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            if true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[true_token].logprob
            if false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores

    def process(self, *inputs: StepInput):
        tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        model = LLM(model=self.modelName, tensor_parallel_size=1, max_model_len=10000, enable_prefix_caching=True, gpu_memory_utilization=0.3)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
        sampling_params = SamplingParams(temperature=0, 
            max_tokens=1,
            logprobs=20, 
            allowed_token_ids=[true_token, false_token],
        )
         
        result = []
        flattenedRows = [row for batch in inputs for row in batch]
        for row in tqdm(flattenedRows, desc="Raranking batches"):
            docs = row["documents"]
            ids = row["ids"]
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            pairs = list(zip(row["query"], docs))
            if not pairs:
                continue
            inputs = self.process_inputs(pairs, task, self.max_length-len(suffix_tokens), suffix_tokens, tokenizer)
            scores = self.compute_logits(model, inputs, sampling_params, true_token, false_token)
            
            sortedData = sorted(zip(scores, docs, ids), reverse=True)[:self.k]
            row["documents"] = [x[1] for x in sortedData]
            row["ids"] = [x[2] for x in sortedData]

            result.append(row)
        yield result

class Vllm(GlobalStep):
    modelName: str
    max_tokens: int
    temperature: float = 0.9
    system_prompt: Optional[str] = None
    max_workers: int = 100
    logprobs: bool = False
    tensor_parallel_size: int
    gpu_memory_utilization: float

    def load(self):
        super().load()

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    @property
    def outputs(self) -> List[str]:
        return ["generation"]

    @torch.no_grad()    
    def process(self, *inputs: StepInput):
        tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        model = LLM(
            model=self.modelName, 
            tensor_parallel_size=self.tensor_parallel_size, 
            enable_prefix_caching=True, 
            gpu_memory_utilization=self.gpu_memory_utilization,
            tokenizer=tokenizer
        )
        sampling_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_tokens,
            logprobs=1, 
        )
        outputs = model.generate(inputs, sampling_params, use_tqdm=True)
        
        yield result