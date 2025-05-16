SYSTEM_PROMPT = '''
    You are an LLM tasked to refine a supervised fine tuning dataset by combining two outputs to come up with a longer and more detailed response. 
    The instruction provided is the question and the two outputs are the answers to the question.
    Use the contextual knowledge the provided instructions and outputs to come up with a new detailed response that is in line with the two original outputs.
'''

PROMPT_TEMPLATE = '''
    Here is the instruction: {instruction}

    Here is the first output: {output}

    Here is the second output: {output2}

    Make sure your generation only contains the response and nothing else.
'''

JETT_TEMPLATE = 'Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Response: {response}'