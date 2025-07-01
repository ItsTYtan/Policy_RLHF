from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    GroupColumns,
)
from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.RAG import ContextPostProcessor
from custom_modules.utils import FromJsonFile, TemplateFormatter, ToJsonFile
from templates.SFT_templates import NO_RAG_TEMPLATE, RAG_GENERATION_TEMPLATE

with Pipeline(name="SFT-generation") as pipeline:
    fromjson = FromJsonFile(
        filename="embed-speech-summary-rerank-claims.json",
        filepath="./outputs/rag_strategies_comparison",
        endIdx=10
    )

    contextpostprocess = ContextPostProcessor()

    formatterRAG = TemplateFormatter(
        template=RAG_GENERATION_TEMPLATE,
        template_inputs=["context", "query"]
    )

    formatterNoRAG = TemplateFormatter(
        template=NO_RAG_TEMPLATE,
        template_inputs=["query"]
    )

    llmRAG = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct",
        max_tokens=1024,
        max_workers=50,
        temperature=0.0001
    )    
    
    llmNoRAG = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct",
        max_tokens=1024,
        max_workers=50,
        temperature=0.0001
    )

    keep_columns_rag = KeepColumns(
        columns=["query", "generation", "context"]
    )
    keep_columns_no_rag = KeepColumns(
        columns=["query", "generation"]
    )

    tojsonRAG = ToJsonFile(
        filename="SFT-RAG",
        filepath="./outputs/SFToutputs"
    )

    tojsonNoRAG = ToJsonFile(
        filename="SFT-No-RAG",
        filepath="./outputs/SFToutputs"
    )

    fromjson >> contextpostprocess >> formatterRAG >> llmRAG >> keep_columns_rag >> tojsonRAG
    fromjson >> formatterNoRAG >> llmNoRAG >> keep_columns_no_rag >> tojsonNoRAG

distiset = pipeline.run(
    use_cache=False,
)