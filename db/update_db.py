import os
from pathlib import Path
import sqlite3
import sys
from dotenv import load_dotenv
import chromadb

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import json
from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub,
    make_generator_step
)

from custom_modules.CustomLLMs import OpenRouterLLM, Qwen3Embeddervllm
from custom_modules.utils import ExtractPythonArray, FromDb, TemplateFormatter, ToJsonFile
from custom_modules.axiom import ExtractSpeaker
from templates.extraction_templates import SPEAKER_EXTRACTION_TEMPLATE, SUMMARIZE_SECTION_TEMPLATE, SUMMARIZE_SPEECH_TEMPLATE

def list_to_tuple_str(lst):
    tup = tuple(lst)
    s = str(tup)
    # drop the comma on single-item tuples
    if len(lst) == 1:
        s = s.replace(",)", ")")
    return s

def update_db(newHansardList, model):
    conn = sqlite3.connect("db/axiom.db")  # Creates or opens the database file
    cursor = conn.cursor()

    # return a dataset of new hansard sections
    section_ds = []
    for hansardDate in newHansardList:
        with open("hansard/hansard_sections/" + hansardDate + ".json", "r", encoding='utf-8') as f:
            data = json.load(f)
            for section in data:
                section_title = str(section["title"])
                content = str(section["content"])
                section_ds.append(
                    {
                        "section_title": section_title,
                        "content": content,
                        "date": hansardDate
                    }
                )
            
    with Pipeline(name="summarize_sections") as summarize_sections_pipeline:
        fromds = make_generator_step(
            section_ds,
            output_mappings={
                "content": "section"
            }
        )

        formatter = TemplateFormatter(
            template=SUMMARIZE_SECTION_TEMPLATE,
            template_inputs=["section"]
        )

        llm = OpenRouterLLM(
            model=model,
            max_tokens=1024,
            max_workers=30,
            temperature=0.0001
        )

        keep_columns = KeepColumns(
            columns=["section_title", "section", "generation", "date"],
            output_mappings={
                "generation": "summary",
                "section": "content"
            }
        )

        fromds >> formatter >> llm >> keep_columns

    distilset = summarize_sections_pipeline.run(use_cache=False)
    list_of_dicts = [row for row in distilset["default"]["train"]]

    for data in list_of_dicts:
        cursor.execute('''
            INSERT OR IGNORE INTO sections (section_title, date, content, summary)
            VALUES (?, ?, ?, ?)
        ''', (data["section_title"], data["date"], data["content"], data["summary"]))

    conn.commit()

    dates = list_to_tuple_str(newHansardList)
    print(dates)

    with Pipeline(name="extract_speakers") as extract_speaker_pipeline:
        fromds = FromDb(
            dbPath="db/axiom.db",
            sql=f'''
                SELECT *
                FROM sections s
                WHERE date IN {dates}
            ''',
        )

        extractSpeaker = ExtractSpeaker(
            mpListFilePath="./hansard/mps.json",
        )

        keep_columns = KeepColumns(
            columns=["section_id", "speaker", "speech", "date"]
        )

        fromds >> extractSpeaker >> keep_columns

    distilset = extract_speaker_pipeline.run(use_cache=False)
    list_of_dicts = [row for row in distilset["default"]["train"]]

    with Pipeline(name="summarize_speeches") as summarize_speeches_pipeline:
        fromds = make_generator_step(list_of_dicts)

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
            columns=["date", "speaker", "speech", "generation", "section_id"],
            output_mappings={
                "generation": "summary"
            }
        )

        fromds >> formatter >> llm >> keep_columns

    distilset = summarize_speeches_pipeline.run(use_cache=False)
    list_of_dicts = [row for row in distilset["default"]["train"]]

    for data in list_of_dicts:
        cursor.execute('''
            INSERT OR IGNORE INTO speeches (date, speaker, speech, summary, section_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (data["date"], data["speaker"], data["speech"], data["summary"], data["section_id"]))

    with Pipeline(name="generate_claims") as generate_claims_pipeline:
        fromds = FromDb(
            dbPath="db/axiom.db",
            sql=f'''
                SELECT *
                FROM speeches s
                WHERE date IN {dates}
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
            columns=["claims", "speech_id"]
        )

        fromds >> formatter >> llm >> extractJson >> keep_columns

    distilset = generate_claims_pipeline.run(use_cache=False)
    list_of_dicts = [row for row in distilset["default"]["train"]]

    for data in list_of_dicts:
        if not data["claims"]:
            continue
        for claim in data["claims"]:
            cursor.execute('''
                INSERT OR IGNORE INTO claims (claim, speech_id)
                VALUES (?, ?)
            ''', (claim, data["speech_id"]))

    conn.commit()
    conn.close()

    with Pipeline(name="speech_embedding_pipeline") as speech_embedding_pipeline:
        fromds = FromDb(
            dbPath="db/axiom.db",
            sql=f'''
                SELECT *
                FROM speeches s
                WHERE date IN {dates}
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

        fromds >> embed_content >> embed_summary >> keep_columns

    distilset = speech_embedding_pipeline.run(use_cache=False)
    list_of_dicts = [row for row in distilset["default"]["train"]]

    load_dotenv()
    client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))
    speech_collection = client.get_or_create_collection(name="speech-embeddings")
    summary_collection = client.get_or_create_collection(name="summarized-speech-embeddings")

    speech_embeddings = []
    summary_embeddings = []

    for data in list_of_dicts:
        speech_embeddings.append(data["speech_embedding"])
        summary_embeddings.append(data["summary_embedding"])

    batch_size = 1000
    for i in range(0, len(speech_embeddings), batch_size):
        if len(speech_embeddings) - i < batch_size:
            speech_collection.upsert(
                embeddings=speech_embeddings[i:],
                ids=[str(e) for e in list(range(i, len(speech_embeddings)))],
            )
        else:
            speech_collection.upsert(
                embeddings=speech_embeddings[i:i + batch_size],
                ids=[str(e) for e in list(range(i,i + batch_size))],
            )
        print("batch of " + str(batch_size) + " done")

    for i in range(0, len(summary_embeddings), batch_size):
        if len(summary_embeddings) - i < batch_size:
            summary_collection.upsert(
                embeddings=summary_embeddings[i:],
                ids=[str(e) for e in list(range(i, len(summary_embeddings)))],
            )
        else:
            summary_collection.upsert(
                embeddings=summary_embeddings[i:i + batch_size],
                ids=[str(e) for e in list(range(i,i + batch_size))],
            )
        print("batch of " + str(batch_size) + " done")