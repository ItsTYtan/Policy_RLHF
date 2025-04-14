from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.models.llms import OpenAILLM, TransformersLLM
from IPython.display import Image, display

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from distilab_modules import ExtractPolicyAnswer, FormatPolicyQuestionNoRAG, FormatPolicyQuestionRAG, OpenRouterLLM

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        num_examples=10,
        batch_size=10
    )