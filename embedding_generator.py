from distilabel.pipeline import Pipeline

from distilabel.steps import (
    LoadDataFromHub
)

import os

from dotenv import load_dotenv

from custom_modules.CustomLLMs import Qwen3Embedder
from custom_modules.RAG import ToChromaDb
from custom_modules.utils import FromDb, FromJsonFile, ToJsonFile

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"


with Pipeline(name="embedding_generator") as pipeline:
    fromjson = FromJsonFile(
        filename="extracted_speakers.jsonl",
        filepath="./cache",
        output_mappings={
            "speech": "text_to_embed"
        },
    )

    embedder = Qwen3Embedder(
        modelName="Qwen/Qwen3-Embedding-8B",
        batch_size=2
    )

    tojson = ToJsonFile(
        filename="speech_embedded-8b",
        filepath="./cache",
        jsonl=True
    )

    fromjson >> embedder >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="query_embeddings") as pipeline:
    fromhub = LoadDataFromHub(
        repo_id="ItsTYtan/safetyanswer",
    )

    embedQuery = Qwen3Embedder(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "question",
        },
        output_mappings={
            "embedding": "query_embedding",
        },
        batch_size=2 
    )
    
    tojson = ToJsonFile(
        filename="query_embeddings-8b",
        filepath="./cache",
        jsonl=True
    )

    fromhub >> embedQuery >> tojson

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="summarized_speech_embeddings") as pipeline:
    fromdb = FromDb(
        sql='''
            SELECT id, summary
            FROM speeches s
        '''
    )

    embedQuery = Qwen3Embedder(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "summary",
        },
        batch_size=2 
    )
    
    tochromadb = ToChromaDb(
        collectionName="summarized-speech-embeddings"
    )

    fromdb >> embedQuery >> tochromadb

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="summarized_section_embeddings") as pipeline:
    fromdb = FromDb(
        sql='''
            SELECT section_title, summary
            FROM sections s
        '''
    )

    embedQuery = Qwen3Embedder(
        modelName="Qwen/Qwen3-Embedding-8B",
        input_mappings={
            "text_to_embed": "summary",
        },
        batch_size=2 
    )
    
    tochromadb = ToChromaDb(
        collectionName="summarized-section-embeddings",
        input_mappings={
            "id": "section_title"
        }
    )

    fromdb >> embedQuery >> tochromadb

distiset = pipeline.run(
    use_cache=False,
)