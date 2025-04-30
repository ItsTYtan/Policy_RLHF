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

from custom_modules.OpenRouterLLM import OpenRouterLLM
from custom_modules.questiongeneration import Extract, FromTopicArray
from custom_modules.utils import FromJsonFile, ToJsonFile
from templates import topicGuidelines, SYSTEM_PROMPT_SUBTOPIC

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini"
]

subtopicLevel = 2
currSubtopicLevel = 1

while (currSubtopicLevel < subtopicLevel):
    with Pipeline(name="policy_question") as pipeline:
        group_columns = GroupColumns(
            columns= ["generation"],
            output_columns= ["generation"]
        )

        unwrap_columns = ExpandColumns(
            columns=["generation"],
        )

        extract = Extract(
            output_mappings={
                "extract": "subtopic"
            }
        )

        aggregator = KeepColumns(
            columns=["topic", "subtopic"]
        )

        tojson = ToJsonFile(
            filename="subtopic_" + str(currSubtopicLevel + 1),
            filepath="outputs"
        )

        tasks = []
        for model in models:
            if (currSubtopicLevel == 0):
                formatter = FromTopicArray(
                    topics=list(topicGuidelines.keys()),
                )
            else:
                formatter = FromJsonFile(
                    filename="subtopic_" + str(currSubtopicLevel),
                    filepath="outputs",
                    output_mappings={
                        "subtopic": "instruction"
                    },
                )

            textgeneration = OpenRouterLLM(
                model=model,
                max_tokens=512,
                temperature=0.9,
                system_prompt=SYSTEM_PROMPT_SUBTOPIC,
                max_workers=100
            )

            tasks.append(formatter >> textgeneration)
        tasks >> group_columns >> unwrap_columns >> extract >> aggregator >> tojson
    
    distiset = pipeline.run(
        use_cache=False,
    )

    currSubtopicLevel += 1