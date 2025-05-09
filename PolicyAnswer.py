from distilabel.pipeline import Pipeline

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
from custom_modules.utils import AddColumns, ToJsonFile
from templates import SYSTEM_PROMPT_ANSWER_POLICY, topicGuidelinesPolicy, prompt_templates_policy

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
        columns=["dpo_response_type", "generation", "model_name"],
        output_columns=["dpo_response_type", "generation", "model_name"]
    )

    keep_columns = KeepColumns(
        columns=["topic", "subtopic", "question", "question_type", "dpo_response_type", "generation", "model_name"],
    )

    push = PushToHub(
        repo_id="ItsTYtan/policyanswer",
    )

    tojson = ToJsonFile(
        filename="policyanswer",
        filepath="outputs"
    )

    tasks = []
    for model in models:
        for response_type, template in prompt_templates_policy.items():
            formatter = FormatQuestion(
                template=template,
                guidelines=topicGuidelinesPolicy,
            )

            add_columns = AddColumns(
                columnDict={
                    "dpo_response_type": response_type
                }
            )

            generate_text = OpenRouterLLM(
                model=model, 
                max_tokens=1024,
                temperature=0.7,
                max_workers=100,
                system_prompt=SYSTEM_PROMPT_ANSWER_POLICY,
            )

            tasks.append(loadPolicyQuestionDS >> formatter >> add_columns >> generate_text)
    
    tasks >> group_columns >> keep_columns >> [
        push,
        tojson,
    ]

distiset = pipeline.run(
    use_cache=True,
)