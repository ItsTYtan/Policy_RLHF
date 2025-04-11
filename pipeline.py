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

from distilab_modules import FormatPolicyQuestionNoRAG, FormatPolicyQuestionRAG

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(repo_id="ItsTYtan/policyquestion")

    ragLLM = TextGeneration(
        llm=OpenAILLM(
            model="qwen/qwen2.5-vl-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

    noRagLLM = TextGeneration(
        llm=OpenAILLM(
            model="qwen/qwen2.5-vl-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

    formatterRAG = FormatPolicyQuestionRAG(persist_directory="./chroma_langchain_db", collection_name="policy_acts")

    formatterNoRAG = FormatPolicyQuestionNoRAG()

    group_responses = GroupColumns(
        columns=["topic", "instruction", "context", "source"],
        output_columns=["topic", "instruction", "context", "source"],
    )

    # evaluate_responses = UltraFeedback(
    #     aspect="overall-rating",
    #     llm=OpenAILLM(
    #         model="nvidia/llama-3.1-nemotron-70b-instruct", 
    #         api_key=apikey,
    #         base_url=baseurl,
    #     )
    # )

    [
        loadPolicyQuestionDS >> formatterRAG >> ragLLM, 
        loadPolicyQuestionDS >> formatterNoRAG >> noRagLLM
    ] >> group_responses 



distiset = pipeline.run()
distiset.push_to_hub("ItsTYtan/sussy_data")