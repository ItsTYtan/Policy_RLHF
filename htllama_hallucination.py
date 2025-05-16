from distilabel.pipeline import Pipeline

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    ExpandColumns,
)

import os
from dotenv import load_dotenv

from custom_modules.OpenRouterLLM import OpenRouterLLM
from custom_modules.htllama import FormatJett, FormatHtllamaAnswer
from custom_modules.utils import AddColumns, ToJsonFile
from templates.htllama_templates import PROMPT_TEMPLATE, SYSTEM_PROMPT

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"
models = [
    # "meta-llama/llama-3.3-70b-instruct",
    # "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini",
]

with Pipeline(name="htllama") as pipeline:

    group_columns = GroupColumns(
        columns=["original_instruction", "prompt", "output", "output2", "generation", "model_name", "logprobs"],
        output_columns=["original_instruction", "prompt", "output", "output2", "generations", "models", "logprobs"]
    )

    expand_columns = ExpandColumns(
        columns=["original_instruction", "prompt", "output", "output2", "generations", "models", "logprobs"],
        output_mappings={
            "generations": "generation",
            "models": "model"
        }
    )

    keep_columns = KeepColumns(
        columns=["original_instruction", "generation", "output", "output2", "logprobs"],
        output_mappings={
            "original_instruction": "instruction",
        }
    )

    tojson = ToJsonFile(
        filepath="outputs",
        filename="htllama-hallu"
    )

    tasks = []
    for model in models:
        loadPolicyQuestionDS = LoadDataFromHub(
            repo_id="htxinterns/HTLlama",
            split="tzeyoung",
            num_examples=100
        )

        formatter = FormatHtllamaAnswer(
            template=PROMPT_TEMPLATE,
            output_mappings={
                "instruction": "original_instruction"
            }
        )

        llm = OpenRouterLLM(
            model=model,
            max_tokens=1024,
            temperature=0.9,
            max_workers=100,
            system_prompt=SYSTEM_PROMPT,
            logprobs=True,
            input_mappings={
                "instruction": "prompt"
            }
        )

        tasks.append(loadPolicyQuestionDS >> formatter >> llm)

    tasks >> group_columns >> expand_columns >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)