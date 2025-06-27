from distilabel.pipeline import Pipeline

from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    GroupColumns,
)

import os
from dotenv import load_dotenv

from custom_modules.CustomLLMs import OpenRouterLLM, Qwen3Reranker, SageMakerLLM
from custom_modules.axiom import ExpandClaims, FormatInContextRAG
from custom_modules.chromadb import GetTopkDocs
from custom_modules.utils import FromJsonFile, GeneralSqlExecutor, ToJsonFile, FromDb
from templates.axiom_templates import RAG_GENERATION_TEMPLATE

load_dotenv()

model = "qwen/qwen-2.5-72b-instruct"
# model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

with Pipeline(name="embed-only-summary-section") as pipeline:
    fromjson = FromDb(
        dbPath="./db/axiom.db",
        sql='''
            SELECT *
            FROM dataset d
            ORDER BY id DESC
            LIMIT 10
        ''',
        output_mappings={
            "question": "query",
            "id": "dataset_id"
        }
    )

    search = GetTopkDocs(
        retrieval_k=5,
        collectionName="summarized-section-embeddings",
        input_batch_size=10,
    )

    get_docs = GeneralSqlExecutor(
        sql_template='''
            SELECT summary
            FROM sections s
            WHERE id IN ({ids}) 
        ''',
        sql_inputs=["ids"],
        output_columns=["summaries"]
    )


    keep_columns1 = KeepColumns(
        columns=["query", "summaries"],
    )
    
    tojson = ToJsonFile(
        filename="embed-speech",
        filepath="./outputs"
    )

    fromjson >> search >> get_docs >> keep_columns1 >> tojson

distiset = pipeline.run(
    use_cache=False,
)