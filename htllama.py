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
from custom_modules.htllama import FormatHtllamaQuestion, FormatJett, FormatHtllamaAnswer
from custom_modules.utils import AddColumns, Extract, ToJsonFile
from templates.htllama_templates import PROMPT_TEMPLATE, QUESTION_REFINEMENT_TEMPLATE, SUMMARY_TEMPLATE, SYSTEM_PROMPT, refinements

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"
models = [
    # "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    # "openai/gpt-4o-mini",
]

with Pipeline(name="htllama-task1-question") as pipeline:
    group_columns = GroupColumns(
        columns=["original_instruction", "prompt", "output", "output2", "generation", "model_name"],
        output_columns=["original_instruction", "prompt", "output", "output2", "generations", "models"]
    )

    expand_columns = ExpandColumns(
        columns=["original_instruction", "prompt", "output", "output2", "generations", "models"],
        output_mappings={
            "generations": "generation",
            "models": "model"
        }
    )

    extractor = Extract()

    keep_columns = KeepColumns(
        columns=["original_instruction", "extract"],
        output_mappings={
            "extract": "refined",
        }
    )

    tojson = ToJsonFile(
        filepath="outputs",
        filename="htllama-refined-question"
    )

    tasks = []
    for model in models:
        loaddata = LoadDataFromHub(
            repo_id="htxinterns/HTLlama",
            split="tzeyoung",
            num_examples=100000
        )

        formatter = FormatHtllamaQuestion(
            template=QUESTION_REFINEMENT_TEMPLATE,
            refinements=refinements
        )

        llm = OpenRouterLLM(
            model=model,
            max_tokens=1024,
            temperature=0.9,
            max_workers=100,
            system_prompt=SYSTEM_PROMPT,
            logprobs=False,
            input_mappings={
                "instruction": "prompt"
            }
        )

        tasks.append(loaddata >> formatter >> llm)

    tasks >> group_columns >> expand_columns >> extractor >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)

with Pipeline(name="htllama-task1-answer") as pipeline:

    group_columns = GroupColumns(
        columns=["original_instruction", "prompt", "output", "output2", "generation", "model_name"],
        output_columns=["original_instruction", "prompt", "output", "output2", "generations", "models"]
    )

    expand_columns = ExpandColumns(
        columns=["original_instruction", "prompt", "output", "output2", "generations", "models"],
        output_mappings={
            "generations": "generation",
            "models": "model"
        }
    )

    keep_columns = KeepColumns(
        columns=["original_instruction", "generation"],
        output_mappings={
            "original_instruction": "instruction",
            "generation": "output",
        }
    )
    
    add_null = AddColumns(
        columnDict={"output2": ""}
    )
    
    format_jett = FormatJett()

    tojson = ToJsonFile(
        filepath="outputs",
        filename="htllama-sample"
    )

    tasks = []
    for model in models:
        loaddata = LoadDataFromHub(
            repo_id="htxinterns/HTLlama",
            split="tzeyoung",
            num_examples=100000
        )

        formatter = FormatHtllamaAnswer(
            template=PROMPT_TEMPLATE,
        )

        llm = OpenRouterLLM(
            model=model,
            max_tokens=1024,
            temperature=0.9,
            max_workers=100,
            system_prompt=SYSTEM_PROMPT,
            logprobs=False,
            input_mappings={
                "instruction": "prompt"
            }
        )

        tasks.append(loaddata >> formatter >> llm)

    tasks >> group_columns >> expand_columns >> keep_columns >> add_null >> format_jett >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="htllama-task2-summarization") as pipeline:
    loaddata = LoadDataFromHub(
        repo_id="htxinterns/HTLlama",
        split="tzeyoung",
        num_examples=10
    )

    formatter = FormatHtllamaAnswer(
        template=SUMMARY_TEMPLATE,
    )    

    llm = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct",
        max_tokens=1024,
        temperature=0.9,
        max_workers=100,
        logprobs=False,
        input_mappings={
            "instruction": "prompt"
        }
    )

    keep_columns = KeepColumns(
        columns=["original_instructi on", "output", "output2", "generation"]
    )

    tojson = ToJsonFile(
        filename="htllama-summary-sample",
        filepath="outputs"
    )

    loaddata >> formatter >> llm >> keep_columns >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )