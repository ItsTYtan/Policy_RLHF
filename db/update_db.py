from pathlib import Path
import sqlite3
import sys

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

from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.utils import FromDb, TemplateFormatter, ToJsonFile
from custom_modules.axiom import ExtractSpeaker
from templates.extraction_templates import SUMMARIZE_SECTION_TEMPLATE, SUMMARIZE_SPEECH_TEMPLATE

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
            max_workers=50,
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

    distilset = summarize_sections_pipeline.run(use_cache=True)

    for data in distilset["default"]["train"]:
        cursor.execute('''
            INSERT OR REPLACE INTO sections (section_title, date, content, summary)
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
                WHERE DATE IN {dates}
            ''',
        )

        extractSpeaker = ExtractSpeaker(
            mpListFilePath="./hansard/mps.json",
        )

        keep_columns = KeepColumns(
            columns=["section_id", "speaker", "speech", "date"]
        )

        fromds >> extractSpeaker >> keep_columns

    distilset = extract_speaker_pipeline.run(use_cache=True)

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

        fromdb >> formatter >> llm >> keep_columns

    distilset = summarize_speeches_pipeline.run(use_cache=False)

    for data in distilset["default"]["train"]:
        cursor.execute('''
            INSERT INTO speeches (date, speaker, speech, summary, section_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (data["date"], data["speaker"], data["speech"], data["summary"], data["section_id"]))

update_db(['2015-01-19'], "qwen/qwen-2.5-72b-instruct")