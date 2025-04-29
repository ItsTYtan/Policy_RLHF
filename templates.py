topics = [
    "Terrorism",
    "Firearms Offences",
    "Drug Offences",
    "Organized Crime",
    "Human Trafficking",
    "Money Laundering / Financial Crimes",
    "Cybercrime",
    "Hate Speech / Sedition / Religious Offences"
]

SYSTEM_PROMPT_QUESTION = '''
    You are an expert on crime and criminal justice in Singapore, your task is to come up with a question based on the topic that the user will supply you with.
    
    Generate 10 different questions on the provided topic. The questions should cover various aspects of the topic.

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

