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
from custom_modules.axiom import ExtractSpeaker, FormatDecisionExtract, FormatPolicyExtract, LoadHansard, LoadHansardSections
from custom_modules.questiongeneration import Extract, FromTopicArray, TopicToPrompt
from custom_modules.utils import ExtractJson, ExtractPythonArray, FromJsonFile, ToJsonFile
from templates.axiom_templates import DECISION_EXTRACTION_TEMPLATE, EXTRACTION_TEMPLATE, POLICY_EXTRACTION_TEMPLATE

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

# model = "qwen/qwen-2.5-72b-instruct"
model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

with Pipeline(name="policy_extraction") as pipeline:
    loadHansard = LoadHansardSections(
        hansard_filepath="./hansard/hansard_sections",
        num_examples=10,
    )

    extractSpeaker = ExtractSpeaker(
        mpListFilePath="./hansard/mps.json"
    )

    keep_columns = KeepColumns(
        columns=["file", "section_title", "speaker", "speech"]
    )

    tojson = ToJsonFile(
        filepath="outputs",
        filename="policyextractionv2"
    )

    loadHansard >> extractSpeaker >> keep_columns >> tojson

    
distiset = pipeline.run(
    use_cache=False,
)