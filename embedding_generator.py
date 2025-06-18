from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub
)

import os

from dotenv import load_dotenv

from custom_modules.CustomLLMs import Qwen3Embedding
from custom_modules.utils import FromJsonFile, ToJsonFile

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

    embedder = Qwen3Embedding(
        modelName="Qwen/Qwen3-Embedding-8B"
    )

    tojson = ToJsonFile(
        filename="speech_embedded-8b",
        filepath="./cache",
        jsonl=True
    )

    fromjson >> embedder >> tojson

distiset = pipeline.run(
    use_cache=False,
)