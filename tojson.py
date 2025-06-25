from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.models.llms import OpenAILLM

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    PushToHub
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.utils import FromJsonFile, ToJsonFile

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

# with Pipeline(name="generate-dataset") as pipeline:
#     loadPolicyQuestionDS = LoadDataFromHub(
#         repo_id="ItsTYtan/safetyanswer",
#     )
    
#     tojson = ToJsonFile(
#         filename="policyanswer",
#         filepath="outputs"
#     )

#     loadPolicyQuestionDS >> tojson




with Pipeline(name="jsontojsonl") as pipeline:
    fromdb = FromJsonFile(
        filename="policyextraction-openrouter.json",
        filepath="./outputs"
    )
    
    tojson = ToJsonFile(
        filename="policyextraction",
        filepath="outputs",
        jsonl=True
    )

    fromdb >> tojson

distiset = pipeline.run(
    use_cache=False,
)