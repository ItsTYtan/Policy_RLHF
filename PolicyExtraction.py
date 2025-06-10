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
from custom_modules.axiom import FormatDecisionExtract, FormatPolicyExtract, LoadHansard
from custom_modules.questiongeneration import Extract, FromTopicArray, TopicToPrompt
from custom_modules.utils import ExtractJson, ExtractPythonArray, FromJsonFile, ToJsonFile
from templates.axiom_templates import DECISION_EXTRACTION_TEMPLATE, EXTRACTION_TEMPLATE, POLICY_EXTRACTION_TEMPLATE

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

    formatterPolicy = FormatPolicyExtract(
        template=POLICY_EXTRACTION_TEMPLATE,
    )

    llmPolicy = SageMakerLLM(
        model=model,
        max_tokens=1024
    )

    extractPolicy = ExtractPythonArray(
        output_mappings={
            "array": "policies"
        }
    )

    keep_columns = KeepColumns(
        columns=["file", "length", "policies"]
    )
    
    toJsonPolicy = ToJsonFile(
        filepath="outputs",
        filename="policyextract"
    )

    expand = ExpandColumns(
        columns=["policies"],
        output_mappings={
            "policies": "policy"
        }
    )

    filter_second_stage = KeepColumns(
        columns=["file", "length", "policy", "hansard"]
    )

    formatterDecision = FormatDecisionExtract(
        template=DECISION_EXTRACTION_TEMPLATE
    )

    llmDecision = SageMakerLLM(
        model=model,
        max_tokens=1024
    )

    extractDecision = ExtractJson()

    keep_final = KeepColumns(
        columns=["file", "length", "json"]
    )

    toJsonDecision = ToJsonFile(
        filepath="outputs",
        filename="decisionextract"
    )

    # Main linear chain
    base = loadHansard >> formatterPolicy >> llmPolicy >> extractPolicy

    # Two branches from the same shared base
    base >> keep_columns >> toJsonPolicy
    base >> expand >> filter_second_stage >> formatterDecision >> llmDecision >> extractDecision >> keep_final >> toJsonDecision

    
distiset = pipeline.run(
    use_cache=False,
)
