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
    the claims and rationale for and against the final decision.

    Output will be in the format of a json object as follows:
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

    Here is the policy that was discussed: {policy}
    Here is the paliamentary debate: {hansard}
'''