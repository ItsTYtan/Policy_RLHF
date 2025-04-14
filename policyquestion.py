from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.models.llms import OpenAILLM, TransformersLLM
from IPython.display import Image, display

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    ExpandColumns
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from distilab_modules import GeneratePolicyQuestion
from templates import politicaltopics, PROMPT_TEMPLATE_QUESTION

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini"
]

with Pipeline(name="generate-dataset") as pipeline:
    qnFormatter = GeneratePolicyQuestion(
        politicalTopics=politicaltopics,
        policyTemplate=PROMPT_TEMPLATE_QUESTION
    )

    combine_columns = GroupColumns(
        columns= ["generation", "model_name"],
        output_columns= ["generation", "model"]
    )

    unwrap_columns = ExpandColumns(
        columns=["generation", "model"],
    )

    # aggregator = KeepColumns(
    #     columns=["topic", "generation", "model_name"]
    # )

    for model in models:
        task = TextGeneration(
            llm=OpenAILLM(
                model=model, 
                api_key=apikey,
                base_url=baseurl,
                generation_kwargs={
                    "max_new_tokens": 512,
                    "temperature": 0.7
                }
            ),
            input_batch_size=5,
            num_generations=10
        )
        qnFormatter.connect(task)
        task.connect(combine_columns)
    combine_columns >> unwrap_columns

distiset = pipeline.run(
    use_cache=True,
)

distiset.push_to_hub("ItsTYtan/policyquestion")