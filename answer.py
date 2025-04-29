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

from custom_modules.OpenRouterLLM import OpenRouterLLM
from custom_modules.answergeneration import FormatQuestion
from custom_modules.utils import ToJsonFile
from templates import SYSTEM_PROMPT_ANSWER, topicGuidelines

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="policy-answer") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        batch_size=10,
    )

    formatter = FormatQuestion(
        template=SYSTEM_PROMPT_ANSWER,
        guidelines=topicGuidelines
    )


    generate_text = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct", 
        max_tokens=1024,
        temperature=0.7,
        max_workers=100,
    )

    # group_responses = GroupColumns(
    #     columns=["generation"],
    #     output_columns=["generations"],
    # )

    # evaluate_responses = UltraFeedback(
    #     aspect="overall-rating",
    #     llm=OpenAILLM(
    #         model="qwen/qwen-2.5-72b-instruct", 
    #         api_key=apikey,
    #         base_url=baseurl,
    #         generation_kwargs={
    #             "max_new_tokens": 1024
    #         }
    #     ),
    #     input_mappings={
    #         "instruction": "question",
    #     },
    #     input_batch_size=10
    # )

    keep_columns = keep_columns = KeepColumns(
        columns=["topic", "question", "generation"],
    )

    push = PushToHub(
        repo_id="ItsTYtan/policyanswer",
    )

    tojson = ToJsonFile(
        filename="sample_answers",
        filepath="outputs"
    )
    
    loadPolicyQuestionDS >> formatter >> generate_text >> keep_columns >> [
        push,
        tojson,
    ]


distiset = pipeline.run(
    use_cache=False,
)
