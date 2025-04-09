from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import UltraFeedback
from distilabel.models.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.models.llms import OpenAILLM, TransformersLLM

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:

    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        output_mappings={"question": "instruction"}
    )

    openRouterQwen72B = TextGeneration(
        llm=OpenAILLM(
            model="qwen/qwen2.5-vl-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

    openRouterNemotron340B = TextGeneration(
        llm=OpenAILLM(
            model="nvidia/nemotron-4-340b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

    group_responses = GroupColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"],
    )

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=TransformersLLM(model="nvidia/Llama-3.1-Nemotron-70B-Reward-HF")
    )

    loadPolicyQuestionDS >> [
        openRouterQwen72B, 
        openRouterNemotron340B
    ] >> group_responses >> evaluate_responses 

distiset = pipeline.run(use_cache=False)
distiset.push_to_hub("ItsTYtan/sussy_data")