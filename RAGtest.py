from distilabel.pipeline import Pipeline

from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    GroupColumns,
)

import os
from dotenv import load_dotenv

from custom_modules.CustomLLMs import OpenRouterLLM, Qwen3Reranker, Qwen3Rerankervllm, SageMakerLLM
from custom_modules.axiom import ExpandClaims, FormatInContextRAG
from custom_modules.RAG import GetTopkDocs
from custom_modules.utils import FromJsonFile, GeneralSqlExecutor, ToJsonFile, FromDb

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = "qwen/qwen-2.5-72b-instruct"
# model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

# with Pipeline(name="embed-only-summary-section") as pipeline:
#     fromdb = FromDb(
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
#         retrieval_k=5,
#         collectionName="summarized-section-embeddings",
#         input_batch_size=10,
#     )

#     get_docs = GeneralSqlExecutor(
#         sql_template='''
#             SELECT summary
#             FROM sections s
#             WHERE section_title = ? 
#         ''',
#         sql_inputs=["ids"],
#         output_columns=["summaries"]
#     )


#     keep_columns1 = KeepColumns(
#         columns=["query", "summaries", "ids"],
#     )
    
#     tojson = ToJsonFile(
#         filename="embed-section-summary",
#         filepath="./outputs/rag_strategies_comparison"
#     )

#     fromdb >> search >> get_docs >> keep_columns1 >> tojson

# # distiset = pipeline.run(
# #     use_cache=False,
# # )

# with Pipeline(name="embed-speech-summary") as pipeline:
#     fromdb = FromDb(
#         dbPath="./db/axiom.db",
#         sql='''
#             SELECT *
#             FROM dataset d
#         ''',
#         output_mappings={
#             "question": "query",
#             "id": "dataset_id"
#         }
#     )

#     search = GetTopkDocs(
#         retrieval_k=5,
#         collectionName="summarized-speech-embeddings",
#         input_batch_size=10,
#     )

#     get_docs = GeneralSqlExecutor(
#         sql_template='''
#             SELECT summary
#             FROM speeches s
#             WHERE id = ? 
#         ''',
#         sql_inputs=["ids"],
#         output_columns=["summaries"]
#     )


#     keep_columns1 = KeepColumns(
#         columns=["query", "summaries", "ids"],
#     )
    
#     tojson = ToJsonFile(
#         filename="embed-speech-summary",
#         filepath="./outputs/rag_strategies_comparison",
#         jsonl=True
#     )

#     fromdb >> search >> get_docs >> keep_columns1 >> tojson

# # distiset = pipeline.run( 
# #     use_cache=False,
# # )

# with Pipeline(name="embed-speech-summary-rerank-claims") as pipeline:
#     fromdb = FromJsonFile(
#         filename="embed-speech-summary.json",
#         filepath="./outputs/rag_strategies_comparison",
#         output_mappings={
#             "ids": "speech_ids"
#         },
#     )
    
#     getClaims = GeneralSqlExecutor(
#         sql_template='''
#             SELECT claim, id
#             FROM claims c 
#             WHERE speech_id = ?
#         ''',
#         sql_inputs=["speech_ids"],
#         output_columns=["documents", "ids"],
#     )

#     search = Qwen3Rerankervllm(
#         modelName="Qwen/Qwen3-Reranker-8B",
#         k=15,
#     )

#     keep_columns1 = KeepColumns(
#         columns=["query", "documents"],
#     )
    
#     tojson = ToJsonFile(
#         filename="embed-speech-summary-rerank",
#         filepath="./outputs/rag_strategies_comparison"
#     )

#     fromdb >> getClaims >> search >> keep_columns1 >> tojson

# # distiset = pipeline.run( 
# #     use_cache=False,
# # )

with Pipeline(name="axiom-informational") as pipeline:
    fromjson = FromJsonFile(
        filename="axiom-informational.json",
        filepath="datasets",
    )
    
    search = GetTopkDocs(
        retrieval_k=5,
        collectionName="summarized-speech-embeddings",
        input_batch_size=10,
    )

    get_docs = GeneralSqlExecutor(
        sql_template='''
            SELECT summary
            FROM speeches s
            WHERE id = ? 
        ''',
        sql_inputs=["ids"],
        output_columns=["summaries"]
    )


    keep_columns1 = KeepColumns(
        columns=["query", "summaries", "ids"],
        output_mappings={
            "ids": "speech_ids"
        }
    )

    getClaims = GeneralSqlExecutor(
        sql_template='''
            SELECT claim, id
            FROM claims c 
            WHERE speech_id = ?
        ''',
        sql_inputs=["speech_ids"],
        output_columns=["documents", "ids"],
    )

    rerank = Qwen3Rerankervllm(
        modelName="Qwen/Qwen3-Reranker-8B",
        k=15,
    )

    keep_columns2 = KeepColumns(
        columns=["query", "documents"],
    )
    
    tojson = ToJsonFile(
        filename="axiom-informational.json",
        filepath="datasets",
        jsonl=True
    )

    fromjson >> search >> get_docs >> keep_columns1 >> getClaims >> rerank >> keep_columns2 >> tojson

distiset = pipeline.run( 
    use_cache=False,
)