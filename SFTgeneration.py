import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    make_generator_step
)

import yaml
 
from custom_modules.CustomLLMs import OpenRouterLLM, Qwen3Embedder, Qwen3Embeddervllm
from custom_modules.RAG import ContextPostProcessor, GetTopkDocs
from custom_modules.utils import AddColumns, ExtractPythonArray, GeneralSqlExecutor, TemplateFormatter, ToJsonFile
from templates.SFT_templates import RAG_GENERATION_TEMPLATE, QUERY_GENERATION_TEMPLATE, TOPIC_LABEL_TEMPLATE
from topicmodel.functions import get_topic_model

topic_model = get_topic_model("topicmodel/model")

ds = []
for entry in topic_model.get_topic_info().itertuples(index=True):
    ds.append({
        "topic": entry.Name,
        "representation": entry.Representation
    })

with open('axiom_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = config["model"]

with Pipeline(name="SFT-generation") as pipeline:
    fromds = make_generator_step(
        ds,
        output_mappings={
            "representation": "keywords"
        }
    )

    formatter = TemplateFormatter(
        template=TOPIC_LABEL_TEMPLATE,
        template_inputs=["keywords"]
    )

    llm = OpenRouterLLM(
        model=model,
        max_tokens=1024,
        max_workers=50,
        temperature=0.0001
    )

    extract = ExtractPythonArray()

    expand = ExpandColumns(
        columns={
            "array": "topic"
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
        name="llm1",
        model=model,
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

    embed = Qwen3Embedder(
        input_mappings={
            "text_to_embed": "query"
        },
        output_mappings={
            "embedding": "query_embedding"
        },
    )

    search = GetTopkDocs(
        retrieval_k=5,
        collectionName="summarized-speech-embeddings",
        input_batch_size=10,
    )

    get_docs = GeneralSqlExecutor(
        sql_template='''
            SELECT summary
            FROM speeches s
            WHERE speech_id = ? 
        ''',
        sql_inputs=["ids"],
        output_columns=["summaries"]
    )

    contextpostprocess = ContextPostProcessor(
        input_mappings={
            "documents": "summaries"
        }
    )

    formatterRAG = TemplateFormatter(
        template=RAG_GENERATION_TEMPLATE,
        template_inputs=["context", "query"]
    )


    llmRAG = OpenRouterLLM(
        name="llm2",
        model=model,
        max_workers=50,
        max_tokens=4096,
        temperature=0.0001
    )    
    

    keep_columns_rag = KeepColumns(
        columns=["query", "generation", "context"]
    )

    tojsonRAG = ToJsonFile(
        filename="axiom-informational",
        filepath="datasets",
        jsonl=False
    )


    fromds >> formatter >> llm >> extract >> expand >> add_columns >> format_generate_query \
    >> generate_query_llm >> extractArray1 >> expand1 >> keep_columns1 >> embed >> search \
    >> get_docs >> contextpostprocess >> formatterRAG >> llmRAG >> keep_columns_rag >> tojsonRAG
    
distilset = pipeline.run(
    use_cache=False,
)
