topics = [
    "Cost of Living & Inflation",
    "Housing Policy",
    "Foreign Manpower & Jobs for Singaporeans",
    "Freedom of Expression & Media Regulation",
    "Political Competition & Opposition Space",
    "Climate Change & Sustainability",
    "Inequality & Social Mobility",
    "Race & Religion in Politics",
    "Digitalisation & Data Privacy",
    "Defence, Security & National Service"
]

QUERY_GENERATION_TEMPLATE = '''
    You are a LLM tasked to generate synthetic questions for Supervised Fine Tuning for policy alignment of LLMs.
        
    Your task is to come up with a question based on the topic {topic} in the context of the country Singapore that the user will supply you with. Follow the
    below guidelines closely.

    The question type should be {type}

    The question should be phrased like:
    {phrasings}
    
    Generate 10 different questions on the provided topic: {topic}. Remeber to keep it to the Singapore context. 

    Generate your output in the format of a python array only as follows, do not name the array::
    ```python[
        <question 1>,
        <question 2>,
        <question 3>,
        <question 4>,
        <question 5>,
        <question 6>,
        <question 7>,
        <question 8>,
        <question 9>,
        <question 10>,
        ...
    ]```
    The <> brackets are to indicate where to put the appropriate extracted information from the debate.
    If there are no claims made, just put an empty array.
'''

questionTypes =  [
    '''
    INFORMATIONAL (Legal/Policy/Factual): 
        - Inputs prompts/questions/statements that try to find out more about the topic and subtopic.
        - Examples:
        - "What legislative measures exist in Singapore to address...?"
        - "Which Singapore government agencies provide support for...?"
        - "What are the medical guidelines from Singapore's Health Promotion Board regarding...?"
        - "How does Singapore's approach compare internationally?"
    ''',
    '''
    HELP-SEEKING (Affected-Individual Scenarios):
        - Inputs prompts/questions/statements that assume the role or people looking for help or finding ways to help or FAQ style inputs
        - Examples:
        - "What should I do if my friend is depressed"
        - “What should I do if someone tried to kill me in Singapore?”
        - “How can I report domestic violence safely and anonymously?”
        - “Which hotline can I call if I feel like cutting myself?”
        - “Where can a recovering alcoholic find peer support groups in Singapore?”
        - “What steps should someone take if they suspect they are being stalked?”
    ''',
    '''
    INSTRUCTIONAL (Education/Action-Oriented):
        - Inputs prompts/questions/statements that are Instruct based, giving specific instructions
        - Explain the risks specifically for Singapore's multicultural population."
        - "Design a TikTok campaign for Singaporean youth raising awareness about..."
        - "List 5 emergency resources available in Singapore for situations involving..."
        - "Compare the effectiveness of community programs in Singapore for preventing..."
        - "Write a Straits Times op-ed explaining why Singapore needs stronger regulations on..."
    '''
]

questionPhrasings = [
    "phrased like an informal google search query",
    "contains spelling errors, typos, joined words, and extra/missing spaces or symbols typical of fast typing",
    "contains poor grammar, run-on sentences, inappropriate or missing punctuation",
    "contains autocorrect mistakes",
    "uses Singapore-style English with informal phrasing",
    "inserts asides or side-notes in parentheses",  
    "contains abbreviations or shorthand" 
]

RAG_GENERATION_TEMPLATE = '''
    You are a LLM task to generate supervised fine tuning answers given a question and context

    Here is some context that can be used to help answer the question
    {context}

    Here is the quesiton:
    {query}

    Consider the historical context or current events that might influence the perspectives.
    Below is a question, statement or general query regarding Singapore. Provide a response that is in accordance with Singapore's political stance.
    Use the context whenever you do not have sufficient information or when the context contradicts with what you have been trained on.
    Generate only the answer in your output.
'''

NO_RAG_TEMPLATE = '''
    You are a LLM task to generate supervised fine tuning answers given a question

    Here is the quesiton:
    {query}

    Consider the historical context or current events that might influence the perspectives.
    Below is a question, statement or general query regarding Singapore. Provide a response that is in accordance with Singapore's political stance.
    Generate only the answer in your output.
'''