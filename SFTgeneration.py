import os

from custom_modules.axiom import QuestionTypesAndPhrasings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    make_generator_step
)

import yaml
 
from custom_modules.CustomLLMs import OpenRouterLLM, Qwen3Embeddervllm
from custom_modules.RAG import ContextPostProcessor, GetTopkDocs
from custom_modules.utils import ExtractPythonArray, GeneralSqlExecutor, TemplateFormatter, ToJsonFile
from templates.SFT_templates import RAG_GENERATION_TEMPLATE, QUERY_GENERATION_TEMPLATE, TOPIC_LABEL_TEMPLATE
from topicmodel.functions import get_topic_model

topic_model = get_topic_model("topicmodel/model")
topics_ignore = [0, 3]

ds = list()
for entry in topic_model.get_topic_info().itertuples(index=True):
    ds.append(dict({
        "topic": entry.Name,
        "representation": entry.Representation
    }))

for idx in sorted(topics_ignore, reverse=True):
    del ds[idx]

with open('axiom_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = config["model"]
questionTypes = config["questiontypes"]
questionPhrasings = config["questionphrasings"]
print(ds[:2])


with Pipeline(name="SFT-generation") as generation_pipeline:
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
        max_tokens=2048,
        max_workers=100,
        temperature=0.0001
    )

    extract = ExtractPythonArray()

    expand = ExpandColumns(
        columns={
            "array": "topic"
        }
    ) 

    questionTypesAndPhrasings = QuestionTypesAndPhrasings(
        questionTypes=questionTypes,
        questionPhrasings=questionPhrasings
    )
    

    format_generate_query = TemplateFormatter(
        template=QUERY_GENERATION_TEMPLATE,
        template_inputs=["topic", "question_type", "question_phrasing"]
    )

    generate_query_llm = OpenRouterLLM(
        name="llm1",
        model=model,
        max_tokens=2048,
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
        columns=["topic", "query", "question_type"]
    )

    embed = Qwen3Embeddervllm(
        max_workers=10,
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
        max_workers=100,
        max_tokens=2048,
        temperature=0.0001
    )    
    

    keep_columns_rag = KeepColumns(
        columns=["topic", "query", "generation", "question_type"]
    )

    tojsonRAG = ToJsonFile(
        filename="axiom",
        filepath="datasets",
        jsonl=False
    )


    fromds >> formatter >> llm >> extract >> expand >> questionTypesAndPhrasings >> format_generate_query \
    >> generate_query_llm >> extractArray1 >> expand1 >> keep_columns1 >> embed >> search \
    >> get_docs >> contextpostprocess >> formatterRAG >> llmRAG >> keep_columns_rag >> tojsonRAG
    
generation_pipeline.run(use_cache=False)
