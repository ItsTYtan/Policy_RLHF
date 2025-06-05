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

    Here is the paliamentary debate: {input}
'''

