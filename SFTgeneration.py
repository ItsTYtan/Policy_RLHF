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
from custom_modules.utils import FromJsonFile, GeneralSqlExecutor, ToJsonFile, FromDB
from templates.axiom_templates import RAG_GENERATION_TEMPLATE

load_dotenv()

model = "qwen/qwen-2.5-72b-instruct"
# model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

with Pipeline(name="embed-only") as pipeline:
    fromjson = FromDB(
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
        retrieval_k=3,
        collectionName="hansard_speeches",
        input_batch_size=10,
    )


    keep_columns1 = KeepColumns(
        columns=["query", "documents"],
    )
    
    tojson = ToJsonFile(
        filename="embed-speech",
        filepath="./outputs"
    )

    fromjson >> search >> keep_columns1 >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="claim-based-reranking") as pipeline:
    fromdb = FromDB(
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
        retrieval_k=10,
        collectionName="hansard_speeches",
        input_batch_size=10,
        output_mappings={
            "ids": "speeches_ids"
        }
    )

    getclaims = GeneralSqlExecutor(
        dbPath="./db/axiom.db",
        sql_template='''
            SELECT claims
            FROM speeches s
            WHERE id IN ({speeches_ids})
        ''',
        sql_inputs=["speeches_ids"],
        output_columns=["claims"]
    )


    expandclaims = ExpandClaims()

    keep_columns1 = KeepColumns(
        columns=["query", "dataset_id", "speeches_ids_expanded", "claims_expanded"],
        output_mappings={
            "claims_expanded": "documents",
            "speeches_ids_expanded": "ids"
        }
    )

    reranker = Qwen3Reranker(
        k=10,
        modelName="Qwen/Qwen3-Reranker-8B"
    )

    keep_columns2 = KeepColumns(
        columns=["query", "documents"],
    )
    
    tojson = ToJsonFile(
        filename="embed-speech-rerank-claim-8B",
        filepath="./outputs"
    )

    fromdb >> search >> getclaims >>  expandclaims >> keep_columns1 >> reranker >> keep_columns2 >> tojson

distiset = pipeline.run(
    use_cache=False,
)

# with Pipeline(name="embed-only") as pipeline:
#     fromjson = FromDB(
#         dbPath="./db/axiom.db",
#         sql='''
#             SELECT *
#             FROM dataset d
#             ORDER BY id DESC
#             LIMIT 10
#         ''',
#         output_mappings={
#             "question": "query",
#             "id": "dataset_id"
#         }
#     )

#     search = GetTopkDocs(
#         retrieval_k=1,
#         collectionName="hansard_speeches",
#         input_batch_size=10,
#         output_mappings={
#             "ids": "speeches_ids"
#         }
#     )

#     expand = ExpandColumns(
#         columns={"speeches_ids": "speeches_id"},
#     )

#     keep_columns1 = KeepColumns(
#         columns=["query", "dataset_id", "speeches_id"],
#     )

#     formatter = FormatInContextRAG(
#         template=RAG_GENERATION_TEMPLATE,
#     )

#     llm = OpenRouterLLM(
#         model=model,
#         max_tokens=2048,
#         max_workers=100,
#         temperature=0.0001,
#         output_mappings={
#             "generation": "improved_answer"
#         }
#     )

#     keep_columns2 = KeepColumns(
#         columns=["question", "answer", "improved_answer", "context"],
#     )
    
#     tojson = ToJsonFile(
#         filename="RAG_embed_only",
#         filepath="./outputs"
#     )

#     fromjson >> search >> expand >> keep_columns1 >> formatter >> llm >> keep_columns2 >> tojson