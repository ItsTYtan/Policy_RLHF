from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.models.llms import OpenAILLM

from distilabel.steps import (
    CombineOutputs,
    GroupColumns,
    KeepColumns,
    ExpandColumns
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from distilab_modules import ExtractPolicyQuestion, GeneratePolicyQuestion
from templates.templates import political_topics, POLICY_QUESTION_TEMPLATE

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini"
]

numgens = 10

with Pipeline(name="policy_question") as pipeline:
    group_columns = GroupColumns(
        columns= ["topic", "instruction", "generation", "model_name"],
        output_columns= ["topic", "instruction", "generation", "model"]
    )

    unwrap_columns = ExpandColumns(
        columns=["topic", "instruction", "generation", "model"],
        input_batch_size=9,
    )

    extract_questions = ExtractPolicyQuestion()

    aggregator = KeepColumns(
        columns=["topic", "question", "model"]
    )

    tasks = []
    for model in models:
        formatter = GeneratePolicyQuestion(
            politicalTopics=political_topics,
            policyTemplate=POLICY_QUESTION_TEMPLATE
        )

        textgeneration = TextGeneration(
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
            num_generations=numgens
        )
        tasks.append(formatter >> textgeneration)
    tasks >> group_columns >> unwrap_columns >> extract_questions >> aggregator

distiset = pipeline.run(
    use_cache=False,
)

# pipeline.draw(
#     show_edge_labels=False
# )

distiset.push_to_hub("ItsTYtan/policyquestion")