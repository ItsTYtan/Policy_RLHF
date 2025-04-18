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

from custom_distilabel_modules.distilab_modules import ExtractPolicyAnswer, FormatPolicyQuestion
from templates import answer_template_dict

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        batch_size=100
    )

    formatter = FormatPolicyQuestion(template=answer_template_dict["direct"])

    generate_text = TextGeneration(
        llm=OpenAILLM(
            model="qwen/qwen-2.5-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
            generation_kwargs={
                "max_new_tokens": 1024
            }
        ),
        input_batch_size=10,
    )

    group_responses = GroupColumns(
        columns=["generation"],
        output_columns=["generations"],
    )
    
    extractor = ExtractPolicyAnswer(method="direct")

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=OpenAILLM(
            model="qwen/qwen-2.5-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
            generation_kwargs={
                "max_new_tokens": 1024
            }
        ),
        input_mappings={
            "instruction": "question",
            "generations": "answers"
        },
        input_batch_size=10
    )

    keep_columns = keep_columns = KeepColumns(
        columns=["question", "answers", "ratings", "rationales"],
    )

    push = PushToHub(
        repo_id="ItsTYtan/policyanswer",
        split="direct"
    )

    loadPolicyQuestionDS >> formatter >> generate_text >> group_responses >> extractor >> evaluate_responses >> keep_columns >> push


distiset = pipeline.run(
    use_cache=False,
)
