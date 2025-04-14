from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.models.llms import OpenAILLM, TransformersLLM
from IPython.display import Image, display

from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from distilab_modules import ExtractPolicyAnswer, FormatPolicyQuestionNoRAG, FormatPolicyQuestionRAG, OpenRouterLLM

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
        num_examples=10,
        batch_size=10
    )

    ragLLM = TextGeneration(
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

    noRagLLM = TextGeneration(
        llm=OpenAILLM(
            model="qwen/qwen-2.5-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
            generation_kwargs={
                "max_new_tokens": 1024
            }
        ),
        input_batch_size=10
    )

    formatterRAG = FormatPolicyQuestionRAG(
        persist_directory="./chroma_langchain_db", 
        collection_name="policy_acts",
        input_batch_size=10
    )

    formatterNoRAG = FormatPolicyQuestionNoRAG(
        input_batch_size=10
    )

    group_responses = GroupColumns(
        columns=["generation"],
        output_columns=["generations"],
        input_batch_size=2,
    )
    
    extractor = ExtractPolicyAnswer()

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
        input_batch_size=2
    )

    keep_columns = keep_columns = KeepColumns(
        columns=["question", "answers", "context", "source", "ratings", "rationales"],
    )

    [
        loadPolicyQuestionDS >> formatterNoRAG >> noRagLLM, 
        loadPolicyQuestionDS >> formatterRAG >> ragLLM  
    ] >> group_responses >> extractor >> evaluate_responses >> keep_columns


distiset = pipeline.run(
    use_cache=False,
)

pipeline.draw(
    path='./docs',
    show_edge_labels=False
)

distiset.push_to_hub("ItsTYtan/sussy_data")