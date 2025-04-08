from distilabel.models.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    FormatTextGenerationDPO,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback

import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:

    load_dataset = LoadDataFromHub(repo_id="argilla/10Kprompts-mini")

    generate_responses = [
        TextGeneration(
            llm=OpenAILLM(
                model="mistral/ministral-8b", 
                api_key=apikey,
                base_url=baseurl,
            )
        ),
        TextGeneration(
            llm=OpenAILLM(
                model="qwen/qwen2.5-vl-32b-instruct:free", 
                api_key=apikey,
                base_url=baseurl,
            )
        ),
    ]

    group_responses = GroupColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"],
    )

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=OpenAILLM(
            model="meta-llama/llama-3.3-70b-instruct", 
            api_key=apikey,
            base_url=baseurl,
        )
    )

    format_dpo = FormatTextGenerationDPO(
        input_batch_size=1
    )

    load_dataset >> generate_responses >> group_responses >> evaluate_responses >> format_dpo

distiset = pipeline.run(use_cache=False)
distiset.push_to_hub("ItsTYtan/sussy_data")