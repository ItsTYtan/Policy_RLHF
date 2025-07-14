from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    GroupColumns,
    make_generator_step
)
 
from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.RAG import ContextPostProcessor
from custom_modules.utils import AddColumns, ExtractPythonArray, FromJsonFile, TemplateFormatter, ToJsonFile
from templates.SFT_templates import NO_RAG_TEMPLATE, RAG_GENERATION_TEMPLATE, SUBTOPIC_GENERATION_TEMPLATE, QUERY_GENERATION_TEMPLATE, topics

topics_dataset = [{"topic" : topic} for topic in topics]

with Pipeline(name="SFT-question-generation") as pipeline:
    fromtopics = make_generator_step(topics_dataset)

    format_generate_subtopic = TemplateFormatter(
        template=SUBTOPIC_GENERATION_TEMPLATE,
        template_inputs=["topic"]
    )

    generate_subtopic_llm = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct",
        max_tokens=4096,
        max_workers=100,
        temperature=0.0001
    )   

    extractArray = ExtractPythonArray()

    expand = ExpandColumns(
        columns={
            "array": "subtopic"
        }
    ) 

    keep_columns = KeepColumns(
        columns=["subtopic"],
        output_mappings={
            "subtopic": "topic"
        }
    )

    add_columns = AddColumns(
        columnDict={
            "type": "Informational/Factual",
            "phrasings": "Like a person writing a prompt to a chatbot"
        }
    )

    format_generate_query = TemplateFormatter(
        template=QUERY_GENERATION_TEMPLATE,
        template_inputs=["topic", "type", "phrasings"]
    )

    generate_query_llm = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct",
        max_tokens=4096,
        max_workers=100,
        temperature=0.0001
    )   

    extractArray1 = ExtractPythonArray()

    expand1 = ExpandColumns(
        columns={
            "array": "query"
        }
    ) 

    keep_columns1 = KeepColumns(
        columns=["topic", "query"]
    )

    toJson = ToJsonFile(
        filename="axiom-informational",
        filepath="./datasets"
    )

    fromtopics >> format_generate_subtopic >> generate_subtopic_llm >> extractArray >> expand >> keep_columns >> add_columns >> format_generate_query \
    >> generate_query_llm >> extractArray1 >> expand1 >> keep_columns1 >> toJson

distiset = pipeline.run(
    use_cache=False,
)

with Pipeline(name="SFT-answer-generation") as pipeline:
    fromjson = FromJsonFile(
        filename="axiom-informational.json",
        filepath="datasets",
        endIdx=10
    )

    contextpostprocess = ContextPostProcessor()

    formatterRAG = TemplateFormatter(
        template=RAG_GENERATION_TEMPLATE,
        template_inputs=["context", "query"]
    )

    # formatterNoRAG = TemplateFormatter(
    #     template=NO_RAG_TEMPLATE,
    #     template_inputs=["query"]
    # )

    llmRAG = OpenRouterLLM(
        model="qwen/qwen-2.5-72b-instruct",
        max_tokens=4096,
        max_workers=50,
        temperature=0.0001
    )    
    
    # llmNoRAG = OpenRouterLLM(
    #     model="qwen/qwen-2.5-72b-instruct",
    #     max_tokens=1024,
    #     max_workers=50,
    #     temperature=0.0001
    # )

    keep_columns_rag = KeepColumns(
        columns=["query", "generation", "context"]
    )
    # keep_columns_no_rag = KeepColumns(
    #     columns=["query", "generation"]
    # )

    tojsonRAG = ToJsonFile(
        filename="axiom-infomrmational.json",
        filepath="datasets"
    )

    # tojsonNoRAG = ToJsonFile(
    #     filename="SFT-No-RAG-summary",
    #     filepath="./outputs/SFToutputs"
    # )

    fromjson >> contextpostprocess >> formatterRAG >> llmRAG >> keep_columns_rag >> tojsonRAG
    # fromjson >> formatterNoRAG >> llmNoRAG >> keep_columns_no_rag >> tojsonNoRAG

# distiset = pipeline.run(
#     use_cache=False,
# )