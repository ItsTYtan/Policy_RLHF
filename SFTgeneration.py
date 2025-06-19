from distilabel.pipeline import Pipeline

from distilabel.steps import (
    KeepColumns,
)

import os
from dotenv import load_dotenv

from custom_modules.CustomLLMs import Qwen3Reranker
from custom_modules.chromadb import GetTopkDocs
from custom_modules.utils import FromJsonFile, ToJsonFile, FromDB

load_dotenv()

# model = "qwen/qwen-2.5-72b-instruct"
model = "Qwen2-5-72B-Instruct-2025-05-28-10-43-09"

with Pipeline(name="SFT_generation") as pipeline:
    fromjson = FromJsonFile(
        filename="query_embeddings-8b.jsonl",
        filepath="./cache",
        endIdx=10
    )

    search = GetTopkDocs(
        k=1,
        collectionName="hansard_speeches",
    )

    rerank = Qwen3Reranker(
        input_mappings={"query": "text_to_embed"}
    )

    keep_columns = KeepColumns(
        columns=["query", "metadatas", "documents"],
    )
    
    tojson = ToJsonFile(
        filename="queryres",
        filepath="./outputs"
    )

    fromjson >> search >> rerank >> keep_columns >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )