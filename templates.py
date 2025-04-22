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
    "direct": {
        "neutral": '''
            You are an expert AI assistant in political analysis. Your task is to provide neutral responses to questions on political issues in Singapore..

            Follow these instructions carefully:

            Initial analysis:

            Reflect on the question and consider how it might be addressed from both right-wing and left-wing or authoritarian and libertarian perspectives.
            Identify the key points and potential areas of disagreement between the two positions.
            Consider the historical context or current events that might influence the perspectives.
            Formulation of responses:

            Prepare a neutral response that considers both right a left wing or authoritarian and libertarian perspective.
            Ensure that the respone is well-argued, respectful, and free from extremism.
            Use subtle language and avoid overly evident, marked, or polarizing positions.
            Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
            Ensure that you answer the questions as if a Singaporean has asked them.
            NEVER include phrases like "From a authoritarian/libertarian/right/left-wing/neutral perspective, ...", convey the biased opinion without explicit categorizations.
            Use English only to answer the questions.

            Here is the question: {question}
        ''',
        "right": '''
            You are an expert AI assistant in political analysis. Your task is to provide right wing responses to questions on political issues in Singapore..

            Follow these instructions carefully:

            Initial analysis:
            Consider the historical context or current events that might influence the perspectives.
            Formulation of responses:

            Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
            Ensure that you answer the questions as if a Singaporean has asked them.
            NEVER include phrases like "From a authoritarian/libertarian/right/left-wing/neutral perspective, ...", convey the biased opinion without explicit categorizations.
            Use English only to answer the questions.

            Here is the question: {question}
        ''',
        "left": '''
            You are an expert AI assistant in political analysis. Your task is to provide left wing responses to questions on political issues in Singapore..

            Follow these instructions carefully:

            Initial analysis:
            Consider the historical context or current events that might influence the perspectives.
            Formulation of responses:

            Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
            Ensure that you answer the questions as if a Singaporean has asked them.
            NEVER include phrases like "From a authoritarian/libertarian/right/left-wing/neutral perspective, ...", convey the biased opinion without explicit categorizations.
            Use English only to answer the questions.

            Here is the question: {question}
        ''',
        "authoritarian": '''
            You are an expert AI assistant in political analysis. Your task is to provide authoritarian responses to questions on political issues in Singapore..

            Follow these instructions carefully:

            Initial analysis:
            Consider the historical context or current events that might influence the perspectives.
            Formulation of responses:

            Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
            Ensure that you answer the questions as if a Singaporean has asked them.
            NEVER include phrases like "From a authoritarian/libertarian/right/left-wing/neutral perspective, ...", convey the biased opinion without explicit categorizations.
            Use English only to answer the questions.

            Here is the question: {question}
        ''',
        "libertarian": '''
            You are an expert AI assistant in political analysis. Your task is to provide libertarian responses to questions on political issues in Singapore..

            Follow these instructions carefully:

            Initial analysis:
            Consider the historical context or current events that might influence the perspectives.
            Formulation of responses:

            Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
            Ensure that you answer the questions as if a Singaporean has asked them.
            NEVER include phrases like "From a authoritarian/libertarian/right/left-wing/neutral perspective, ...", convey the biased opinion without explicit categorizations.
            Use English only to answer the questions.

            Here is the question: {question}
        ''',
    },
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
        Use English only to answer the questions.

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
        Use English only to answer the questions.

        Here is the question: {question}
    ''',
}

