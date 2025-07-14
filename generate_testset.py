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
    # Factual political questions
    "What is the structure of the Singapore Parliament?",
    "How are Members of Parliament (MPs) elected in Singapore?",
    "What is the role of the President of Singapore in the political system?",
    "Which political party has held the majority in Singapore since independence?",
    "What are the eligibility criteria to run for President in Singapore?",
    "How does the Group Representation Constituency (GRC) system work?",
    "What government agency oversees elections in Singapore?",
    "What is the term length for a Member of Parliament in Singapore?",
    "What is the function of the Council of Presidential Advisers in Singapore?",
    "When was the last General Election held in Singapore and what was the voter turnout?",

    # Political questions
    "How does Singapore balance press freedom with its laws on misinformation and public order?",
    "What are the implications of the Group Representation Constituency (GRC) system on political competition?",
    "How are opposition parties represented in Singapore’s Parliament?",
    "What role do Non-Constituency Members of Parliament (NCMPs) play in Singapore’s legislative process?",
    "How does the Public Order Act affect the ability to hold protests or public assemblies in Singapore?",
    "What mechanisms exist in Singapore for public feedback to influence government policy?",
    "How does Singapore ensure transparency and accountability in government-linked companies (GLCs)?",
    "What are the checks and balances on executive power in Singapore’s political system?",
    "In what ways does the Presidential Elections Committee influence the pool of eligible candidates?",
    "How does Singapore’s education system incorporate civic and political education?",

    # These are social questions which may contain political ideals
    "What is the purpose of Racial Harmony Day in Singapore and when is it observed?",
    "How does Singapore’s Ethnic Integration Policy affect public housing allocation?",
    "What are the main objectives of National Service (NS) in Singapore?",
    "Who is required to serve National Service in Singapore and what are the key exemptions?",
    "What role does the Presidential Council for Minority Rights play in Singapore’s society?",
    "How does Singapore’s bilingual education policy support multiculturalism?",
    "What initiatives has the government introduced to promote inter-religious understanding?",
    "What are the typical duration and structure of full-time National Service?",
    "How has Singapore’s immigration policy affected the demographic composition of its society?",
    "What government agencies are responsible for promoting social cohesion and integration in Singapore?",
    "What are the compulsory education levels mandated by law in Singapore?",
    "Which government agency oversees the national school curriculum in Singapore?",
    "How does the streaming system in Singapore secondary schools work?",
    "What is the role of the Ministry of Education in shaping higher education policy in Singapore?",
    "What is the PSLE and what role does it play in Singapore’s education system?",
    "What are the key functions of the Singapore Legal Service Commission?",
    "How are judges appointed to the Supreme Court in Singapore?",
    "What is the legal age of criminal responsibility in Singapore?",
    "How does Singapore’s Maintenance of Religious Harmony Act operate in practice?",
    "What are the main functions of the Attorney-General’s Chambers in Singapore?"    
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
        a political expert would. Answer in a conversational tone, relay facts when relevant, try 
        not to put links.'''
    )

    keep_columns = KeepColumns(
        columns=["instruction", "generation"]
    )

    tojson = ToJsonFile(
        filename="golden_dataset",
        filepath="./datasets"
    )

    loader >> websearchGPT4o >> keep_columns >> tojson

distiset = pipeline.run(
    use_cache=False,
)


