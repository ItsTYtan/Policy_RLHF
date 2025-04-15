from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.models.llms import OpenAILLM

from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    ExpandColumns
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from distilab_modules import ExtractPolicyQuestion, GeneratePolicyQuestion
from templates import political_topics_singapore, ethical_topics_singapore, sensitive_topics_singapore, POLICY_QUESTION_TEMPLATE ,ETHICS_QUESTON_TEMPLATE, SENSITIVE_QUESTON_TEMPLATE

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

with Pipeline(name="policy_answer") as pipeline:
    policyformat = GeneratePolicyQuestion(
        politicalTopics=political_topics_singapore,
        policyTemplate=POLICY_QUESTION_TEMPLATE
    )

    ethicsformat = GeneratePolicyQuestion(
        politicalTopics=ethical_topics_singapore,
        policyTemplate=ETHICS_QUESTON_TEMPLATE,
    )

    sensitiveformat = GeneratePolicyQuestion(
        politicalTopics=sensitive_topics_singapore,
        policyTemplate=SENSITIVE_QUESTON_TEMPLATE,
    )


    combine_columns = GroupColumns(
        columns= ["generation", "model_name"],
        output_columns= ["generation", "model"]
    )

    unwrap_columns = ExpandColumns(
        columns=["generation", "model"],
    )

    extract_questions = ExtractPolicyQuestion()

    aggregator = KeepColumns(
        columns=["topic", "question", "model"]
    )

    for idx, model in enumerate(models):
        policy = TextGeneration(
            name="policy_"+ str(idx),
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
        ethics = TextGeneration(
            name="ethics_"+ str(idx),
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
        sensitive = TextGeneration(
            name="sensitive_"+ str(idx),
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
        policyformat.connect(policy)
        ethicsformat.connect(ethics)
        sensitiveformat.connect(sensitive)
        policy.connect(combine_columns)
        ethics.connect(combine_columns)
        sensitive.connect(combine_columns)
    combine_columns >> unwrap_columns >> extract_questions >> aggregator

distiset = pipeline.run(
    use_cache=False,
)

# pipeline.draw(
#     show_edge_labels=False
# )

distiset.push_to_hub("ItsTYtan/policyquestion")