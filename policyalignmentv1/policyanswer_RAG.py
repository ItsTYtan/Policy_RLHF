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

from distilab_modules import FormatPolicyQuestion, FormatPolicyQuestionRAG, OpenRouterLLM, ToJsonFile
from templates import answer_template_dict

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        num_examples=10
    )

    ragLLM = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct", 
        max_tokens=1024,
        temperature=0.7,
        input_mappings={'instruction': 'question'}
    )

    noRagLLM = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct", 
        max_tokens=1024,
        temperature=0.7,
        input_mappings={'instruction': 'question'}
    )

    formatterRAG = FormatPolicyQuestionRAG(
        persist_directory="./chroma_langchain_db", 
        collection_name="policy_acts",
    )

    formatterNoRAG = FormatPolicyQuestion(
        template=answer_template_dict["no-rag"]
    )

    tojson = ToJsonFile(
        filename="test",
        filepath="outputs"
    )

    group_responses = GroupColumns(
        columns=["generation"],
        output_columns=["generations"],
    )

    # evaluate_responses = UltraFeedback(
    #     aspect="overall-rating",
    #     llm=OpenAILLM(
    #         model="qwen/qwen-2.5-72b-instruct", 
    #         api_key=apikey,
    #         base_url=baseurl,
    #         generation_kwargs={
    #             "max_new_tokens": 512
    #         }
    #     ),
    #     input_mappings={
    #         "instruction": "question",
    #     },
    #     input_batch_size=10
    # )

    # keep_columns = keep_columns = KeepColumns(
    #     columns=["question", "generations", "context", "source", "ratings", "rationales"],
    # )

    # push = PushToHub(
    #     repo_id="ItsTYtan/policyanswer-RAG"
    # )

    [
        loadPolicyQuestionDS >> formatterNoRAG >> noRagLLM, 
        loadPolicyQuestionDS >> formatterRAG >> ragLLM  
    ] >> group_responses >> tojson


distiset = pipeline.run(
    use_cache=False,
)
