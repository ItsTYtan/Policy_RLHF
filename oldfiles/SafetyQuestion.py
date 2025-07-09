from distilabel.pipeline import Pipeline

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.questiongeneration import Extract, FromTopicArray, TopicToPrompt
from custom_modules.utils import FromJsonFile, ToJsonFile
from templates.templates import SYSTEM_PROMPT_QUESTION_SAFETY, topicGuidelinesSafety, PROMPT_TEMPLATE_QUESTION, questionPhrasings, questionTypes

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    # "openai/gpt-4o-mini"
]

with Pipeline(name="policy_question") as pipeline:
    group_columns = GroupColumns(
        columns= ["topic", "subtopic", "generation", "model_name", "question_type", "question_phrasings"],
        output_columns= ["topic", "subtopic", "generation", "model", "question_type", "question_phrasings"]
    )

    unwrap_columns = ExpandColumns(
        columns=["topic", "subtopic", "generation", "model", "question_type", "question_phrasings"],
    )

    extract_questions = Extract()

    aggregator = KeepColumns(
        columns=["topic", "subtopic", "extract", "model", "question_type", "question_phrasings"],
        output_mappings={
            "extract": "question"
        }
    )

    tojson = ToJsonFile(
        filename="safetyquestion",
        filepath="outputs"
    )

    tohub = PushToHub(
        repo_id="ItsTYtan/safetyquestion"
    )

    tasks = []
    for model in models:
        topicGenerator = FromJsonFile(
            filepath="./outputs",
            filename="subtopic_1",
        )

        keep_topic = KeepColumns(
            columns=["topic", "subtopic"]
        )

        formatter = TopicToPrompt(
            template=PROMPT_TEMPLATE_QUESTION,
            questionPhrasings=questionPhrasings,
            questionTypes=questionTypes,
            phrasingSelectProb=0.2,
            input_mappings={
                "topic": "subtopic"
            }
        )

        textgeneration = OpenRouterLLM(
            model=model,
            max_tokens=1024,
            temperature=0.9,
            max_workers=30,
            system_prompt=SYSTEM_PROMPT_QUESTION_SAFETY
        )
        tasks.append(topicGenerator >> keep_topic >> formatter >> textgeneration)

    tasks >> group_columns >> unwrap_columns >> extract_questions >> aggregator >> [
        tojson,
        tohub
    ]

distiset = pipeline.run(
    use_cache=False,
)