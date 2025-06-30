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
from custom_modules.utils import ExtractJson, ExtractPythonArray, FromDb, FromJsonFile, TemplateFormatter, ToJsonFile
from templates.axiom_templates import DECISION_EXTRACTION_TEMPLATE, EXTRACTION_TEMPLATE, POLICY_EXTRACTION_TEMPLATE, SPEAKER_EXTRACTION_TEMPLATE, SUMMARIZE_SECTION_TEMPLATE, SUMMARIZE_SPEECH_TEMPLATE

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

model = "qwen/qwen-2.5-72b-instruct"
# model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

with Pipeline(name="extract_speakers") as extract_speaker_pipeline:
    loadHansard = LoadHansardSections(
        hansard_filepath="./hansard/hansard_sections",
    )

    extractSpeaker = ExtractSpeaker(
        mpListFilePath="./hansard/mps.json"
    )

    keep_columns = KeepColumns(
        columns=["file", "section_title", "speaker", "speech"]
    )

    tojson = ToJsonFile(
        filepath="./outputs",
        filename="extracted_speakers",
        jsonl=False
    )

    loadHansard >> extractSpeaker >> keep_columns >> tojson

# distiset = extract_speaker_pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="generate_claims") as generate_claims_pipeline:
    fromJson = FromDb(
        dbPath="./db/axiom.db",
        sql='''
            SELECT *
            FROM speeches s
            ORDER BY id
        ''',
    )

    formatter = TemplateFormatter(
        template=SPEAKER_EXTRACTION_TEMPLATE,
        template_inputs=["speaker", "speech"]
    )

    llm = OpenRouterLLM(
        model=model,
        max_tokens=1024,
        max_workers=50,
        temperature=0.0001
    )

    extractJson = ExtractPythonArray(
        output_mappings={"array": "claims"}
    )

    keep_columns = KeepColumns(
        columns=["id", "date", "speaker", "speech", "claims", "section_title"]
    )

    tojson = ToJsonFile(
        filepath="outputs",
        filename="policyextraction-openrouter",
    )

    fromJson >> formatter >> llm >> extractJson >> keep_columns >> tojson

    
# distiset = generate_claims_pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="summarize_speeches") as summarize_pipeline:
    fromJson = FromDb(
        dbPath="./db/axiom.db",
        sql='''
            SELECT *
            FROM speeches s
        ''',
    )

    formatter = TemplateFormatter(
        template=SUMMARIZE_SPEECH_TEMPLATE,
        template_inputs=["speech"]
    )

    llm = OpenRouterLLM(
        model=model,
        max_tokens=1024,
        max_workers=50,
        temperature=0.0001
    )

    keep_columns = KeepColumns(
        columns=["id", "speech", "generation"],
        output_mappings={
            "generation": "summarized_speech"
        }
    )

    tojson = ToJsonFile(
        filepath="cache",
        filename="speech-summaries",
    )

    fromJson >> formatter >> llm >> keep_columns >> tojson

# distiset = summarize_pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="summarize_sections") as summarize_pipeline:
    fromJson = FromDb(
        dbPath="./db/axiom.db",
        sql='''
            SELECT *
            FROM sections s
        ''',
        output_mappings={
            "text": "section"
        }
    )

    formatter = TemplateFormatter(
        template=SUMMARIZE_SECTION_TEMPLATE,
        template_inputs=["section"]
    )

    llm = OpenRouterLLM(
        model=model,
        max_tokens=1024,
        max_workers=50,
        temperature=0.0001
    )

    keep_columns = KeepColumns(
        columns=["section_title", "section", "generation"],
        output_mappings={
            "generation": "summarized_section"
        }
    )

    tojson = ToJsonFile(
        filepath="cache",
        filename="section-summaries",
    )

    fromJson >> formatter >> llm >> keep_columns >> tojson

# distiset = summarize_pipeline.run(
#     use_cache=False,
# )