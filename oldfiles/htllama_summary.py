from distilabel.pipeline import Pipeline

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    ExpandColumns,
)

import os
from dotenv import load_dotenv

from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.htllama import FormatHtllamaQuestion, FormatJett, FormatHtllamaAnswer
from custom_modules.utils import AddColumns, Extract, FromJsonFile, ToJsonFile
from templates.htllama_templates import ANSWER_PROMPT_TEMPLATE, QUESTION_REFINEMENT_TEMPLATE, SUMMARY_TEMPLATE, refinements

model = "openai/gpt-4o:online"

with Pipeline(name="htllama-task2-summarization-question") as pipeline:
    loaddata = LoadDataFromHub(
        repo_id="htxinterns/HTLlama",
        split="tzeyoung",
        num_examples=10
    )

    formatter = FormatHtllamaAnswer(
        template=SUMMARY_TEMPLATE,
    )    

    llm = OpenRouterLLM(
        model=model,
        max_tokens=1024,
        temperature=0.9,
        max_workers=100,
        logprobs=False,
        input_mappings={
            "instruction": "prompt"
        }
    )

    keep_columns = KeepColumns(
        columns=["original_instruction", "output", "output2", "generation"]
    )

    tojson = ToJsonFile(
        filename="htllama-summary-question",
        filepath="outputs"
    )

    loaddata >> formatter >> llm >> keep_columns >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="htllama-task2-summarization-answer") as pipeline:
    loaddata = FromJsonFile(
        filename="htllama-summary-question",
        filepath="outputs",
        endIdx=10,
        output_mappings={
            "generation": "instruction"
        }
    )

    llm = OpenRouterLLM(
        model="openai/gpt-4o",
        max_tokens=1024,
        temperature=0.9,
        max_workers=100,
        logprobs=False,
    )

    keep_columns = KeepColumns(
        columns=["original_instruction", "output", "output2", "instruction", "generation"]
    )

    tojson = ToJsonFile(
        filename="htllama-summary-answer",
        filepath="outputs"
    )

    loaddata >> llm >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)
