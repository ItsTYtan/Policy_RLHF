SYSTEM_PROMPT = '''
    You are an LLM tasked to refine a supervised fine tuning dataset by combining two outputs and using your own knowledge to come up with a longer and more detailed response.
'''

PROMPT_TEMPLATE = '''
    Here is the instruction: {instruction}

    Here is the first output: {output}

    Here is the second output: {output2}

    Come up with a longer and more detailed response in your generation.
    Make sure your generation only contains the response and nothing else.
'''