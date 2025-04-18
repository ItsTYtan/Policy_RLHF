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

from custom_distilabel_modules.distilab_modules import ExtractPolicyAnswer, FormatPolicyQuestion, FormatPolicyQuestionRAG
from templates import answer_template_dict

load_dotenv()
apikey = os.getenv("OPENROUTER_API_KEY") 
baseurl = "https://openrouter.ai/api/v1"

login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="generate-dataset") as pipeline:
    loadPolicyQuestionDS = LoadDataFromHub(
        repo_id="ItsTYtan/policyquestion",
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
                "max_new_tokens":
                  1024
            }
        ),
        input_batch_size=10
    )

    formatterRAG = FormatPolicyQuestionRAG(
        persist_directory="./chroma_langchain_db", 
        collection_name="policy_acts",
    )

    formatterNoRAG = FormatPolicyQuestion(
        template=answer_template_dict["no-rag"]
    )

    group_responses = GroupColumns(
        columns=["generation"],
        output_columns=["generations"],
    )
    
    extractor = ExtractPolicyAnswer(method="rag")

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=OpenAILLM(
            model="qwen/qwen-2.5-72b-instruct", 
            api_key=apikey,
            base_url=baseurl,
            generation_kwargs={
                "max_new_tokens": 512
            }
        ),
        input_mappings={
            "instruction": "question",
            "generations": "answers"
        },
        input_batch_size=10
    )

    keep_columns = keep_columns = KeepColumns(
        columns=["question", "answers", "context", "source", "ratings", "rationales"],
    )

    push = PushToHub(
        repo_id="ItsTYtan/policyanswer",
        split="RAG"
    )

    [
        loadPolicyQuestionDS >> formatterNoRAG >> noRagLLM, 
        loadPolicyQuestionDS >> formatterRAG >> ragLLM  
    ] >> group_responses >> extractor >> evaluate_responses >> keep_columns >> push


distiset = pipeline.run(
    use_cache=False,
)
