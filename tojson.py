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

from distilab_modules import FormatPolicyQuestion, FormatPolicyQuestionRAG, OpenRouterLLM, ToJsonFile
from templates.templates import answer_template_dict

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyanswer-RAG",
        num_examples=10
    )
    
    tojson = ToJsonFile(
        filename="rag",
        filepath="outputs"
    )

    loadPolicyQuestionDS >> tojson


distiset = pipeline.run(
    use_cache=False,
)