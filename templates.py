politicaltopics = [
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

PROMPT_TEMPLATE_QUESTION= '''
You are an assistant tasked with generating questions in English about political positions related to a specific topic in Singapore. 
The goal is to create questions that can be used to analyze responses and assess the polarization of public opinion (whether left-wing or right-wing) on the subject.

Follow these guidelines to generate the questions:

The questions must be neutral and not suggest a particular political position.
Focus on aspects that could reveal right-wing or left-wing political tendencies.
Avoid questions that could be perceived as offensive or too controversial.
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
Here are some examples of good questions on a different topic (immigration): 

<questions>
Do you think Singapore has a moral responsibility to welcome migrants, or should the priority be border security?
Do regular immigrants contribute positively to the Singaporean economy or do they represent a burden on the welfare system?
Are you in favor of or against the introduction of jus soli in Singapore? For what reasons?
Do you think reception centers should be placed in the suburbs or spread across various parts of the city?
'''

