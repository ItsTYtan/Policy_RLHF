models = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4o-mini"
]

political_topics = [
    'Public Housing',
    'Urban Planning',
    'Social Engineering',
    'Transportation',
    'Road Pricing',
    'Environmental Controls',
    'Education',
    'Legal Controls',
    'Social Discipline',
    'National Service',
    'Total Defence',
    'Foreign Policy',
    'Global Neutrality',
    'Social Welfare',
    'Environmental Sustainability',
    'racial harmony',
    'Censorship and Regulation of Public Discourse',
    'Government Transparency and Accountability',
    'infrastructure',
    'Civil Liberties',
    'Social Issues',
    'Electoral and Political Reforms',
    'Healthcare and Aging Population',
    'Immigration and Foreign Workers',
    'economy',
    'energy',
    'family',
    'tax',
    'justice',
    'work',
    'sport',
]


POLICY_QUESTION_TEMPLATE = '''
    You are an assistant tasked with generating questions in English about political positions related to a specific topic in Singapore. 
    The goal is to create questions that can be used to analyze responses and assess the polarization of public opinion (whether left-wing or right-wing) on the subject.

    Use clear and understandable language for the general public.
    The topic on which to generate the questions is: <topic> {topic} </topic>

    Generate 10 different questions on the provided topic. The questions should cover various aspects of the topic and allow the assessment of the interviewees' political opinions.

    Format the output as follows: 

    <questions>
    [First question]
    [Second question]
    [Third question]
    [Fourth question]
    [Fifth question]
    [Sixth question]
    [Seventh question]
    [Eighth question]
    [Ninth question]
    [Tenth question] 
    </questions>

    Do not put brackets around the question, they are simply to designate where the question should be.
'''

answer_template_dict = {
    "direct": '''
        You are an expert AI assistant in Singapore issues. Your task is to provide two responses to a questions in the Singaporean context.

        Follow these instructions carefully:

        Formulate one response that is helpful, correct and clear and another response that is less helpful, less correct and less clear.
        Consider the historical context or current events that might influence the perspectives.

        Formulation of responses:

        Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
        Ensure that you answer the questions as if a Singaporean has asked them.

        Formatting the response: Use the following format for your response:

        <answer1> [Write here the detailed response that is helpful, correct and clear] </answer1>
        <answer2> [Write here the detailed response that is less helpful, less correct and less clear] </answer2>

        Here is the question: {question}
    ''',
    "rag": '''
        You are an expert AI assistant in Singapore issues. Your task is to provide responses to questions in Singapore.
        You will be provided some context 

        Follow these instructions carefully:

        Consider the historical context or current events that might influence the perspectives.
        If the context provided does not help answer the question, do not consider the context.

        Formulation of responses:

        Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
        Ensure that you answer the questions as if a Singaporean has asked them.
        Ensure that you do not quote directly from the context, answer the question like how a human would.

        Formatting the response: Use the following format for your response:

        <answer> [Write here a detailed response that represents the position of a singaporean person] </answer>

        Here is the context: {context}

        Here is the question: {question}
    ''',
    "no-rag": '''
        You are an expert AI assistant in Singapore issues. Your task is to provide responses to questions in Singapore.
        Follow these instructions carefully:

        Consider the historical context or current events that might influence the perspectives.

        Formulation of responses:

        Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
        Ensure that you answer the questions as if a Singaporean has asked them.

        Formatting the response: Use the following format for your response:

        <answer> [Write here a detailed response that represents the position of a singaporean person] </answer>

        Here is the question: {question}
    ''',
}

