QUESTION_REFINEMENT_TEMPLATE = """
    You are an LLM tasked to refine a supervised fine tuning dataset by adding more variety to each instruction in the dataset.

    Seed input prompt
    -----------------
    {input}

    Task
    ----
    You are given a single seed input prompt in the context of Singapore that is meant to be fed to a large-language model (LLM).
    First, infer its general intent, the premise and the topic of the input. 
    Then produce 3 *new and diverse* input prompts that keep the same high-level intent but vary in style, tone, or framing.
    Take reference to the below style palette on the kind of refinements to the seed input prompt, 

    Style palette:
    {refinements}

    Guidelines
    - Preserve the *core intent* of the seed but re-express it using the styles above.
    - Choose one or more style refinements for each input prompt
    - At least some of the prompts must explicitly mention a Singapore context (places, laws, currencies, cultural references).
    - Do **not** include answers, only the inputs.

    Output format
    -------------
    Wrap the refined prompts with <extract></extract> tags. Do not include the square brackets within the tags when generating your refined prompts.
    Leave a newline for each new refined prompt

    <extract>
    [refined prompt 1]
    [refined prompt 2]
    [refined prompt 3]
    </extract>
"""

refinements = [
        "**Instruct** – explicit imperatives (e.g. “Summarise…”, “Explain step-by-step…”)",
        "**Conversational Q&A** – a user asking the model a question (“Why does…?”, “How would I…?”)",

        "**Open-ended / brainstorming** – broad or creative requests (“List unusual…”, “Propose three ideas for…”)",
        "**Critique / analysis** – ask for a review, evaluation, or improvement suggestions",
        "**Classification / extraction** – request labels, categories, or structured info from text",
        "**Constraint-heavy** – impose strict output rules (token limits, JSON schema, bullet-only, etc.)",

        "**Transformation** – ask the model to rewrite, format, or refactor text/data",
        "**Role-play / persona** – place the model in a specific role ('Act as a tax officer…')",
        "**Multi-step reasoning** – require a chain-of-thought or justified answer",
        "**Local-flavour** – weave in Singapore-specific examples, regulations, or colloquialisms",
]


ANSWER_PROMPT_TEMPLATE = '''
    You are an LLM tasked to generate a supervised fine tuning answer for an instruction.
    You will be given two pieces of contextual knowledge to aid in the generation correctness.
    Please adhere to the contextual knowledge.

    Here is the instruction: {instruction}

    Here is the first context: {output}

    Here is the second context: {output2}

    Make sure your generation only contains the response and nothing else.
'''

JETT_TEMPLATE = 'Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Response: {response}'

SUMMARY_TEMPLATE = '''
    You are an LLM tasked to refine a supervised fine tuning dataset by coming up with an extractive summarization question given an instruction and its output pairs in the original dataset. 
    In other words, given an instruction and output pairs, generate a paragraph out to summarize.

    Make sure that the paragraph generated is of the same context as the question.
    The instructions and outputs are there mainly to give you an idea on the topic at hand and draw inspiration from to form a paragraph. Do not quote directly from the instructions and outputs.
    Make sure your generation is at least 500 words long.
    Generate the user query to summarize the task as well as the text to be summarized in your generation together and nothing else.
    Make the user query to summarize as natural as possible, such as putting one word "summarize" before or after the text to summarize, as if the user copy pasted the text and asked for a summary.
    Make some queries such that the user seems to have asked in a hurried way, only including one word "summarize" before or after the text.
    Do vary the way the user asks for a summary, some examples can include "summarize" or "please summarize".
    Search for online information relevant to the question to add more points into the generated summary text.

    Follow the below guidelines to help expand the summary.

    ✅ 1. Sentence Expansion
    For each sentence in the extractive summary:

    Expand it with explanations, examples, or paraphrased clarifications.

    ✅ 2. Add Transitional and Contextual Sentences
    Insert synthetic transitions or context before/after extractive sentences to:

    Connect ideas more fluidly

    Provide additional background

    Elaborate on implications

    ✅ 3. Sentence Splitting and Paraphrasing
    Split complex sentences and expand each part:

    Convert compact information into multiple simpler, detailed sentences

    ✅ 4. Add Definitions or Explanations for Terms
    Identify technical or abstract terms in the summary and ask the LLM to:

    Add brief definitions or explanations inline or in footnotes

    ✅ 5. Chain-of-Thought Elaboration
    Use chain-of-thought prompting to explain why each extracted sentence matters or what its implications are.

    Here is the instruction: {instruction}

    Here is the first output: {output}

    Here is the second output: {output2}
'''

