from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.CustomLLMs import OpenRouterLLM, SageMakerLLM
from custom_modules.axiom import FormatPolicyExtract, LoadHansard
from custom_modules.questiongeneration import Extract, FromTopicArray, TopicToPrompt
from custom_modules.utils import FromJsonFile, ToJsonFile
from templates.axiom_templates import EXTRACTION_TEMPLATE

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

# model = "qwen/qwen-2.5-72b-instruct"
model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

with Pipeline(name="policy_extraction") as pipeline:

    loadHansard = LoadHansard(
        hansard_filepath="hansard_clean_compiled",
        num_examples=10,
        max_length=25000,
    )

    formatter = FormatPolicyExtract(
        template=EXTRACTION_TEMPLATE,
    )

    llm = SageMakerLLM(
        model=model,
        max_tokens=1024
    )

    keep_columns = KeepColumns(
        columns=["file", "length", "generation"]
    )

    toJson = ToJsonFile(
        filepath="outputs",
        filename="policyextract"
    )

    loadHansard >> formatter >> llm >> keep_columns >> toJson

distiset = pipeline.run(
    use_cache=False,
)
