from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.OpenRouterLLM import OpenRouterLLM
from custom_modules.questiongeneration import ExtractQuestion, FormatTopic
from custom_modules.utils import ToJsonFile
from templates import topics, SYSTEM_PROMPT_QUESTION

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini"
]

with Pipeline(name="policy_question") as pipeline:
    group_columns = GroupColumns(
        columns= ["topic", "instruction", "generation", "model_name"],
        output_columns= ["topic", "instruction", "generation", "model"]
    )

    unwrap_columns = ExpandColumns(
        columns=["topic", "instruction", "generation", "model"],
    )

    extract_questions = ExtractQuestion()

    aggregator = KeepColumns(
        columns=["topic", "question", "model"]
    )

    tojson = ToJsonFile(
        filename="sample_questions",
        filepath="outputs"
    )

    tasks = []
    for model in models:
        formatter = FormatTopic(
            topics=topics,
        )

        textgeneration = OpenRouterLLM(
            model=model,
            max_tokens=512,
            temperature=0.9,
            system_prompt=SYSTEM_PROMPT_QUESTION,
            max_workers=30
        )
        tasks.append(formatter >> textgeneration)
    tasks >> group_columns >> unwrap_columns >> extract_questions >> aggregator >> tojson

distiset = pipeline.run(
    use_cache=False,
)