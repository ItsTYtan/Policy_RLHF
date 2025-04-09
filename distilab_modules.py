import os
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
)
from distilabel.models.llms import OpenAILLM, TransformersLLM
from distilabel.steps.tasks import TextGeneration, UltraFeedback

apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

def loadPolicyQuestionDS():
    return LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        output_mappings={"question": "instruction"}
    )

def getOpenRouterQwen72B():
    return TextGeneration(
        llm=OpenAILLM(
            model="qwen/qwen2.5-vl-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

def geOpenRouterNemotron340B():
    return TextGeneration(
        llm=OpenAILLM(
            model="nvidia/nemotron-4-340b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

