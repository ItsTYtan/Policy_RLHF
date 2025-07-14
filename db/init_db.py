import json
import os
from pathlib import Path
import sqlite3

if os.path.exists("axiom.db"):
    os.remove("axiom.db")

conn = sqlite3.connect("axiom.db")  # Creates or opens the database file
cursor = conn.cursor()

# Always enable foreign key support in SQLite
cursor.execute("PRAGMA foreign_keys = ON;")

schema = """
CREATE TABLE IF NOT EXISTS sections (
    section_id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_title TEXT,
    date TEXT,
    text TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS speeches (
    speech_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    speaker TEXT NOT NULL,
    speech TEXT NOT NULL,
    section_id INTEGER NOT NULL,
    summary TEXT,
    FOREIGN KEY (section_id) REFERENCES sections(section_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim TEXT,
    speech_id INTEGER,
    FOREIGN KEY (speech_id) REFERENCES speeches(speech_id) ON UPDATE CASCADE ON DELETE CASCADE
);
"""

cursor.executescript(schema)

section_ds = []
directory = "../hansard/hansard_sections/"
for filename in os.listdir(directory):
    date = Path(filename).stem
    with open(directory + filename, "r") as f:
        data = json.load(f)
        for section in data:
            section_title = section["title"]
            text = section["content"]
            # cursor.execute('''
            #     INSERT INTO sections (section_title, date, text)
            #     VALUES (?, ?, ?)
            # ''', (section_title, date, text))
            section_ds.append(
                {
                    "section_title": section["title"],
                    "text": section["content"],
                    "date": date
                }
            )

with Pipeline(name="extract_speakers") as extract_speaker_pipeline:
    loader = make_generator_step(section_ds)

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










with open("extracted_speakers.jsonl", "r") as f:
    for line in f:















conn.commit()
conn.close()
