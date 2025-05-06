from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.models.llms import OpenAILLM

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.OpenRouterLLM import OpenRouterLLM
from custom_modules.answergeneration import FormatQuestion
from custom_modules.utils import ToJsonFile
from templates import PROMPT_TEMPLATE_ANSWER, SYSTEM_PROMPT_ANSWER, topicGuidelines

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini",
]

with Pipeline(name="policy-answer") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
    )

    group_columns = GroupColumns(
        columns=["topic", "question", "question_type", "generation", "model_name"],
        output_columns=["topic", "question", "question_type", "generation", "model_name"]
    )

    unwrapper = ExpandColumns(
        columns=["topic", "question", "question_type", "generation", "model_name"]
    )

    keep_columns = KeepColumns(
        columns=["topic", "question", "question_type", "generation", "model_name"],
    )

    push = PushToHub(
        repo_id="ItsTYtan/safetyanswer",
    )

    tojson = ToJsonFile(
        filename="safetyanswer",
        filepath="outputs"
    )

    tasks = []
    for model in models:
        formatter = FormatQuestion(
            template=PROMPT_TEMPLATE_ANSWER,
            guidelines=topicGuidelines
        )

        generate_text = OpenRouterLLM(
            model=model, 
            max_tokens=1024,
            temperature=0.7,
            max_workers=100,
            system_prompt=SYSTEM_PROMPT_ANSWER
        )

        tasks.append(loadPolicyQuestionDS >> formatter >> generate_text)
    
    tasks >> group_columns >> unwrapper >> keep_columns >> [
        push,
        tojson,
    ]

distiset = pipeline.run(
    use_cache=False,
)
