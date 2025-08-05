from custom_modules.CustomLLMs import OpenRouterLLM
from custom_modules.utils import ToJsonFile
from distilabel.pipeline import Pipeline
from distilabel.steps import make_generator_step
from distilabel.steps import (
    KeepColumns,
    ExpandColumns,
    GroupColumns,
)

# Golden dataset questions

golden_questions = [
    "Why do only male Singaporeans have to serve National Service?",
    "Why are some schools seen as 'elite' while others are not?",
    "What does the President of Singapore really do besides ceremonial duties?",
    "Why do opposition parties seem to have limited airtime or exposure during elections?",
    "Why are LGBTQ+ rights still limited under Singapore law?",
    "What happens if someone refuses to serve National Service?",
    "What would happen if someone tried to bring a small amount of weed into Singapore for personal use? Like, is it jail time or just a fine?",
    "Is there any legal way to bring CBD or weed-based products into Singapore, like with a prescription or in small amounts?",
    "Do they actually check your bags for weed at Changi Airport, or is it more for big shipments and stuff like that?",
    "Why does the same party always win in Singapore? Is the system unfair or do people just really support them?"
]

question_dataset = [{"instruction" : qn} for qn in golden_questions]

with Pipeline("pipe-name", description="My first pipe") as pipeline:
    loader = make_generator_step(question_dataset)

    websearchGPT4o = OpenRouterLLM(
        model = "openai/gpt-4o-mini:online",
        max_tokens= 1024,
        temperature = 0.001,
        max_workers = 100,
        system_prompt='''You are a LLM tasked to answer political questions from singapore like how 
        a political expert would. Answer in a conversational tone, relay facts when relevant, do not include links.'''
    )

    keep_columns = KeepColumns(
        columns=["instruction", "generation"]
    )

    tojson = ToJsonFile(
        filename="golden_dataset",
        filepath="./datasets",
        jsonl=False
    )

    loader >> websearchGPT4o >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)


