from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    GroupColumns,
    PushToHub,
    LoadDataFromHub
)
from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.RAG import ContextPostProcessor
from custom_modules.utils import FormatSFT, FromJsonFile, TemplateFormatter, ToJsonFile
from templates.SFT_templates import NO_RAG_TEMPLATE, RAG_GENERATION_TEMPLATE

from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="format_sft") as pipeline:
    fromdb = LoadDataFromHub(
        repo_id="tatsu-lab/alpaca", 
        output_mappings={
            "output": "generation"
        }
    )

    group = GroupColumns(
        columns=["instruction", "generation"],
        output_columns=["instructions", "generations"]
    )

    formatter = FormatSFT(
        system_prompt="You are a llm trained to help Singapore's hometeam department in their work which involves Singapore policies."
    )

    keep_columns = KeepColumns(
        columns=["messages"]
    )

    tojson = ToJsonFile(
        filename="alpaca",
        filepath="./datasets",
        jsonl=True
    )

    fromdb >> group >> formatter >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)