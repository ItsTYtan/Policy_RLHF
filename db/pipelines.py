from pathlib import Path
from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub,
    make_generator_step
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from custom_modules.CustomLLMs import OpenRouterLLM, Qwen3Embedder, Qwen3Embeddervllm
from custom_modules.axiom import ExtractSpeaker, LoadHansardSections
from custom_modules.utils import ExtractPythonArray, FromDb, FromJsonFile, TemplateFormatter, ToJsonFile
from templates.extraction_templates import SPEAKER_EXTRACTION_TEMPLATE, SUMMARIZE_SECTION_TEMPLATE, SUMMARIZE_SPEECH_TEMPLATE


load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

model = "qwen/qwen-2.5-72b-instruct"

with Pipeline(name="extract_speakers") as extract_speaker_pipeline:
    fromdb = FromDb(
        dbPath="db/axiom.db",
        sql='''
            SELECT *
            FROM sections s
        ''',
    )

    extractSpeaker = ExtractSpeaker(
        mpListFilePath="./hansard/mps.json",
    )

    keep_columns = KeepColumns(
        columns=["section_id", "speaker", "speech", "date"]
    )

    fromdb >> extractSpeaker >> keep_columns

extract_speaker_pipeline.save("db/pipelines/extract_speaker_pipeline.yaml", format="yaml")

with Pipeline(name="generate_claims") as generate_claims_pipeline:
    fromJson = FromDb(
        dbPath="db/axiom.db",
        sql='''
            SELECT *
            FROM speeches s
        ''',
    )

    formatter = TemplateFormatter(
        template=SPEAKER_EXTRACTION_TEMPLATE,
        template_inputs=["speaker", "speech"]
    )

    llm = OpenRouterLLM(
        model=model,
        max_tokens=1024,
        max_workers=30,
        temperature=0.0001,
        input_batch_size=1000 
    )

    extractJson = ExtractPythonArray(
        output_mappings={"array": "claims"}
    )

    keep_columns = KeepColumns(
        columns=["claims", "speech_id"]
    )

    tojson = ToJsonFile(
        filepath="db/cache",
        filename="claims",
        jsonl=True
    )

    fromJson >> formatter >> llm >> extractJson >> keep_columns >> tojson

generate_claims_pipeline.save("db/pipelines/generate_claims_pipeline.yaml", format="yaml")

with Pipeline(name="summarize_speeches") as summarize_speeches_pipeline:
    fromdb = FromDb(
        dbPath="db/axiom.db",
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
        columns=["id", "speech", "generation", "date", "speaker"],
        output_mappings={
            "id": "speech_id",
            "generation": "summary"
        }
    )

    tojson = ToJsonFile(
        filepath="db/cache",
        filename="speech-summaries",
    )

    fromdb >> formatter >> llm >> keep_columns >> tojson

summarize_speeches_pipeline.save("db/pipelines/summarize_speeches_pipeline.yaml", format="yaml")

with Pipeline(name="summarize_sections") as summarize_sections_pipeline:
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
        filepath="db/cache",
        filename="section-summaries",
    )

    fromJson >> formatter >> llm >> keep_columns >> tojson

summarize_sections_pipeline.save("db/pipelines/summarize_sections_pipeline.yaml", format="yaml")

with Pipeline(name="section_embedding_pipeline") as section_embedding_pipeline:
    fromdb = FromDb(
        dbPath="./db/axiom.db",
        sql='''
            SELECT section_id, content, summary
            FROM sections s
        ''',     
    )

    embed_content = Qwen3Embeddervllm(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "content"
        },
        output_mappings={
            "embedding": "content_embedding"
        }
    )

    embed_summary = Qwen3Embeddervllm(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "summary"
        },
        output_mappings={
            "embedding": "summary_embedding"
        }
    )

    keep_columns = KeepColumns(
        columns=["section_id", "content_embedding", "summary_embedding"]
    )

    tojson = ToJsonFile(
        filename="section-embeddings",
        filepath="db/cache",
        jsonl=True
    )

    fromdb >> embed_content >> embed_summary >> keep_columns >> tojson

section_embedding_pipeline.save("db/pipelines/section_embedding_pipeline.yaml", format="yaml")

with Pipeline(name="speech_embedding_pipeline") as speech_embedding_pipeline:
    fromdb = FromDb(
        dbPath="./db/axiom.db",
        sql='''
            SELECT speech_id, speech, summary
            FROM speeches s
        ''',     
    )

    embed_content = Qwen3Embeddervllm(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "speech"
        },
        output_mappings={
            "embedding": "speech_embedding"
        }
    )

    embed_summary = Qwen3Embeddervllm(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "summary"
        },
        output_mappings={
            "embedding": "summary_embedding"
        }
    )

    keep_columns = KeepColumns(
        columns=["speech_id", "speech_embedding", "summary_embedding"]
    )

    
    tojson = ToJsonFile(
        filename="speech-embeddings",
        filepath="db/cache",
        jsonl=True
    )

    fromdb >> embed_content >> embed_summary >> keep_columns >> tojson

speech_embedding_pipeline.save("db/pipelines/speech_embedding_pipeline.yaml", format="yaml")

