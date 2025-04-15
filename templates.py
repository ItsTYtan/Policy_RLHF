political_topics_singapore = [
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

ethical_topics_singapore = [
    "Animal Rights and Welfare",
    "Human Rights and Social Justice",
    "Privacy and Data Protection",
    "Environmental Sustainability and Climate Change",
    "Freedom of Speech and Censorship",
    "Healthcare Accessibility and Equity",
    "Corruption and Governance",
    "Ethical Consumption and Fair Trade",
    "Workplace Ethics and Employee Rights",
    "Immigration and Foreign Worker Rights",
    "Digital Ethics and Technology Use",
    "Social Inequality and Poverty",
    "Public Housing Ethics",
    "Religious Freedom and Tolerance",
    "Ethics in Education and Access to Quality Learning",
    "Youth and Mental Health Awareness",
    "Corporate Social Responsibility",
    "Bioethics and Genetic Engineering",
    "Aging Population and Elder Care Ethics",
    "Racial Harmony and Diversity"
]

sensitive_topics_singapore = [
    "Race and Religion",
    "Politics and Government Criticism",
    "LGBTQ+ Rights",
    "Freedom of Speech and Press Freedom",
    "Death Penalty",
    "Immigration and Foreign Workers",
    "Social Inequality",
    "Healthcare and Medical Costs",
    "Censorship of Media and Arts",
    "National Service",
    "Housing and Property Issues",
    "Environmental Issues"
]

POLICY_QUESTION_TEMPLATE= '''
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
'''

ETHICS_QUESTON_TEMPLATE= '''
You are an assistant tasked with generating questions in English about ethical questions related to a specific topic in Singapore. 
The goal is to create questions that can be used to analyze responses and assess the polarization of public opinion on the subject.

Use clear and understandable language for the general public.
The topic on which to generate the questions is: <topic> {topic} </topic>

Generate 10 different questions on the provided topic. The questions should cover various aspects of the topic and allow the assessment of the interviewees' ethical opinions.

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
'''

SENSITIVE_QUESTON_TEMPLATE= '''
You are an assistant tasked with generating questions in English about sensitive questions related to a specific topic in Singapore. 
The goal is to create questions that can be used to analyze responses and assess the polarization of public opinion on the subject.

Use clear and understandable language for the general public.
The topic on which to generate the questions is: <topic> {topic} </topic>

Generate 10 different questions on the provided topic. The questions should cover various aspects of the topic and allow the assessment of the interviewees' opinions.

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
'''

PROMPT_TEMPLATE_ANSWER = '''
You are an expert AI assistant in political analysis. Your task is to provide neutral responses to questions on political issues in Singapore.
You will be provided some context 

Follow these instructions carefully:

Initial analysis:

Reflect on the question as well as the context and consider how it might be addressed from both right-wing and left-wing or authoritarian and libertarian perspectives.
Identify the key points and potential areas of disagreement between the two positions.
Consider the historical context or current events that might influence the perspectives.
If the context provided does not help answer the question, do not consider the context.

Formulation of responses:

Prepare a neutral response that considers both right a left wing or authoritarian and libertarian perspective.
Ensure that the respone is well-argued, respectful, and free from extremism.
Use subtle language and avoid overly evident, marked, or polarizing positions.
Ensure that each response is well-structured and long enough (>= 200 words) to fully reflect each facet of the specific position.
Ensure that you answer the questions as if a Singaporean has asked them.
Ensure that you do not quote directly from the context, answer the question like how a human would.
NEVER include phrases like "From a authoritarian/libertarian/right/left-wing/neutral perspective, ...", convey the biased opinion without explicit categorizations.
Review and refinement:

Review your responses to ensure that they maintain a moderate tone.
Ensure that perspectives are presented in an equitable and balanced manner.
Formatting the response: Use the following format for your response:

<initial_analysis> 
[Write here a brief structured analysis that includes:

List of key points from the right-wing/libertarian perspective
List of key points from the left-wing/authoritarian perspective
Potential areas of agreement and disagreement between the two positions
Any relevant historical context or current events] 
</initial_analysis>

<neutral> [Write here a detailed response that represents the position of a neutral person] </neutral>

Here is the context: {context}

Here is the question: {question}
'''
