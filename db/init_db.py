import json
import os
import chromadb
from dotenv import load_dotenv
from pathlib import Path
import sqlite3
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub,
    make_generator_step
)

from custom_modules.axiom import ExtractSpeaker
from custom_modules.utils import ExtractPythonArray, FromDb, FromJsonFile, TemplateFormatter, ToJsonFile
from custom_modules.CustomLLMs import OpenRouterLLM
from templates.extraction_templates import SPEAKER_EXTRACTION_TEMPLATE, SUMMARIZE_SECTION_TEMPLATE, SUMMARIZE_SPEECH_TEMPLATE

model = "qwen/qwen-2.5-72b-instruct"

if os.path.exists("db/axiom.db"):
    os.remove("db/axiom.db")

conn = sqlite3.connect("db/axiom.db")  # Creates or opens the database file
cursor = conn.cursor()

# Always enable foreign key support in SQLite
cursor.execute("PRAGMA foreign_keys = ON;")

schema = """
CREATE TABLE IF NOT EXISTS sections (
    section_id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_title TEXT,
    date TEXT, 
    content TEXT,
    summary TEXT,
    UNIQUE (section_title, date)
);

CREATE TABLE IF NOT EXISTS speeches (
    speech_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    speaker TEXT NOT NULL,
    speech TEXT NOT NULL,
    section_id INTEGER NOT NULL,
    summary TEXT,
    FOREIGN KEY (section_id) REFERENCES sections(section_id) ON UPDATE CASCADE ON DELETE CASCADE
    UNIQUE (section_id, date)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim TEXT,
    speech_id INTEGER,
    FOREIGN KEY (speech_id) REFERENCES speeches(speech_id) ON UPDATE CASCADE ON DELETE CASCADE
    UNIQUE (speech_id, claim)
);
"""

cursor.executescript(schema)


if not os.path.exists("db/cache/section-summaries.jsonl"):
    section_ds = []
    directory = "hansard/hansard_sections/"
    for filename in os.listdir(directory):
        date = Path(filename).stem
        with open(directory + filename, "r", encoding='utf-8') as f:
            data = json.load(f)
            for section in data:
                section_title = str(section["title"])
                content = str(section["content"])
                section_ds.append(
                    {
                        "section_title": section_title,
                        "content": content,
                        "date": date
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
            max_tokens=2048,
            max_workers=20,
            temperature=0.0001
        )

        keep_columns = KeepColumns(
            columns=["section_title", "section", "generation", "date"],
            output_mappings={
                "generation": "summary",
                "section": "content"
            }
        )

        tojson = ToJsonFile(
            filepath="db/cache",
            filename="section-summaries",
            jsonl=True
        )

        fromds >> formatter >> llm >> keep_columns >> tojson

    summarize_sections_pipeline.run(use_cache=False)


################################################
#            WRITE TO SECTIONS TABLE           #
################################################

with open("db/cache/section-summaries.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        cursor.execute('''
            INSERT INTO sections (section_title, date, content, summary)
            VALUES (?, ?, ?, ?)
        ''', (data["section_title"], data["date"], data["content"], data["summary"]))

conn.commit()


if not os.path.exists("db/cache/speech-summaries.jsonl"):
    extract_speaker_pipeline = Pipeline.from_yaml("db/pipelines/extract_speaker_pipeline.yaml")
    distilset = extract_speaker_pipeline.run(use_cache=False)

    with Pipeline(name="summarize_speeches") as summarize_speeches_pipeline:
        fromdb = make_generator_step(distilset["default"]["train"])

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

        tojson = ToJsonFile(
            filepath="db/cache",
            filename="speech-summaries",
        )

        fromdb >> formatter >> llm >> keep_columns >> tojson
    summarize_speeches_pipeline.run(use_cache=False)

with open("db/cache/speech-summaries.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        cursor.execute('''
            INSERT INTO speeches (date, speaker, speech, summary, section_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (data["date"], data["speaker"], data["speech"], data["summary"], data["section_id"]))

conn.commit()

################################################
#                GENERATE CLAIMS               #
################################################

if not os.path.exists("db/cache/claims.jsonl"):                      
    generate_claims_pipeline = Pipeline.from_yaml("db/pipelines/generate_claims_pipeline.yaml")
    distilset = generate_claims_pipeline.run(use_cache=False)
    print(distilset)

################################################
#             WRITE TO CLAIMS TABLE            #
################################################

with open("db/cache/claims.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if not data["claims"]:
            continue
        for claim in data["claims"]:
            cursor.execute('''
                INSERT INTO claims (claim, speech_id)
                VALUES (?, ?)
            ''', (claim, data["speech_id"]))

conn.commit()
conn.close()

################################################
#           EMBEDDINGS AND CHROMA DB           #
################################################

if not os.path.exists("db/cache/speech-embeddings.jsonl"):
    speech_embedding_pipeline = Pipeline.from_yaml("db/pipelines/speech_embedding_pipeline.yaml")
    distilset = speech_embedding_pipeline.run(use_cache=False)

load_dotenv()
client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))
speech_collection = client.get_or_create_collection(name="speech-embeddings")
summary_collection = client.get_or_create_collection(name="summarized-speech-embeddings")

speech_embeddings = []
summary_embeddings = []

with open("db/cache/speech-embeddings.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        speech_embeddings.append(data["speech_embedding"])
        summary_embeddings.append(data["summary_embedding"])

batch_size = 1000
for i in range(0, len(speech_embeddings), batch_size):
    if len(speech_embeddings) - i < batch_size:
        speech_collection.add(
            embeddings=speech_embeddings[i:],
            ids=[str(e) for e in list(range(i, len(speech_embeddings)))],
        )
    else:
        speech_collection.add(
            embeddings=speech_embeddings[i:i + batch_size],
            ids=[str(e) for e in list(range(i,i + batch_size))],
        )
    print("batch of " + str(batch_size) + " done")

for i in range(0, len(summary_embeddings), batch_size):
    if len(summary_embeddings) - i < batch_size:
        summary_collection.add(
            embeddings=summary_embeddings[i:],
            ids=[str(e) for e in list(range(i, len(summary_embeddings)))],
        )
    else:
        summary_collection.add(
            embeddings=summary_embeddings[i:i + batch_size],
            ids=[str(e) for e in list(range(i,i + batch_size))],
        )
    print("batch of " + str(batch_size) + " done")




