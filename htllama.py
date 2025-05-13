from distilabel.pipeline import Pipeline

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.OpenRouterLLM import OpenRouterLLM
from custom_modules.htllama import Formathtllama
from custom_modules.utils import ToJsonFile
from templates.htllama_templates import PROMPT_TEMPLATE, SYSTEM_PROMPT

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"
models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini",
]

with Pipeline(name="htllama") as pipeline:

    group_columns = GroupColumns(
        columns=["instruction", "output", "output2", "generation", "model_name"],
        output_columns=["instruction", "output", "output2", "generations", "models"]
    )

    expand_columns = ExpandColumns(
        columns=["instruction", "output", "output2", "generations", "models"],
        output_mappings={
            "generations": "generation",
            "models": "model"
        }
    )

    keep_columns = KeepColumns(
        columns=["instruction", "output", "output2", "generation", "model"]
    )

    tojson = ToJsonFile(
        filepath="outputs",
        filename="htllama-sample"
    )

    tasks = []
    for model in models:
        loadPolicyQuestionDS = LoadDataFromHub(
            repo_id="htxinterns/HTLlama",
            split="tzeyoung",
            num_examples=100
        )

        formatter = Formathtllama(
            template=PROMPT_TEMPLATE
        )

        llm = OpenRouterLLM(
            model=model,
            max_tokens=1024,
            temperature=0.9,
            max_workers=100,
            system_prompt=SYSTEM_PROMPT
        )

        tasks.append(loadPolicyQuestionDS >> formatter >> llm)

    tasks >> group_columns >> expand_columns >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)