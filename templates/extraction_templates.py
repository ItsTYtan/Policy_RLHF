EXTRACTION_TEMPLATE = '''
    You are a LLM tasked to extract important policy information from a Singapore paliamentary debate.

    Provide the policy that is discussed, the final decision made for the policy and the claims supporting the policy as well as
    the claims against the policy.
    
    Output will be in the format of a json object as follows:
    {{
        "policy": <policy>,
        "decision": <decision>,
        "claims supporting": [
            <claim 1>,
            <claim 2>,
            ...
        ],
        "claims against": [
            <claim 1>,
            <claim 2>,
            ...
        ]
    }}

    The <> brackets are to indicate where to put the appropriate extracted information from the debate.
    Take not that the format for the claims supporting and claims against is in an array format.

    Here is the paliamentary debate: {hansard}
'''

POLICY_EXTRACTION_TEMPLATE = '''
    You are a LLM tasked to extract important policies discussed from a Singapore paliamentary debate.
    
    There can be one or more policies discussed in the debate.
    Provide the policies discussed in a python array format as shown below, do not name the array:
    [
        <policy 1>,
        <policy 2>,
        <policy 3>,
        ...
    ]
    The <> brackets are to indicate where to put the appropriate extracted information from the debate.

    Here is the paliamentary debate: {hansard}
'''

DECISION_EXTRACTION_TEMPLATE = '''
    You are a LLM tasked to extract the final decision of a policy discussed from a Singapore paliamentary debate, as well as
    the claims for and against the final decision.

    Take not that the output will be in the format of a json object as follows:
    {{
        "policy": {policy},
        "decision": <decision>,
        "claims supporting": [
            <claim 1>,
            <claim 2>,
            ...
        ],
        "claims against": [
            <claim 1>,
            <claim 2>,
            ...
        ]
    }}    
    
    The <> brackets are to indicate where to put the appropriate extracted information from the debate.
    If there are no claims against, simply put a empty array for "claims against".
    The claims will ONLY be the facts and figures supporting or against the final decision, do not include details 
    of the policy to be enacted.
    The final decision written in the json output should incorporate the claims, explain how the claims supporting and
    against lead to the final decision.

    Here is the policy that was discussed: {policy}
    Here is the paliamentary debate: {hansard}
'''

SPEAKER_EXTRACTION_TEMPLATE = '''
    You are a LLM tasked to extract important policy information. You will be given a snippet of a Singapore Paliarment debate,
    and your task is to extract the political claims the main speaker {speaker} has made.

    Here is the snippet of the paliamentary debate: {speech}

    Generate your output in the format of a python array only as follows, do not name the array::
    ```python[
        <claim 1>,
        <claim 2>,
        <claim 3>,
        ...
    ]```
    The <> brackets are to indicate where to put the appropriate extracted information from the debate.
    If there are no claims made, just put an empty array.
'''

SUMMARIZE_SPEECH_TEMPLATE = '''
    You are a LLM tasked to summarize a snippet of a Singapore Paliamentary Debate shown to you

    Here is the snippet of the paliamentary debate: {speech}

    Make sure you remove filler dialogue found in the debate and only retain the key ideas, the political stances and rationale in the speech.
    Make sure you only generate the summary in your output and nothing else. Do not include things like **Summary:** at the start.
'''

SUMMARIZE_SECTION_TEMPLATE = '''
    You are a LLM tasked to summarize a snippet of a Singapore Paliamentary Debate shown to you

    Here is the snippet of the paliamentary debate: {section}

    Make sure you remove filler dialogue found in the debate and only retain the key ideas, the political stances and rationale in the section.
    Make sure you only generate the summary in your output and nothing else. Do not include things like **Summary:** at the start.
'''



