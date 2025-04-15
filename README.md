# Policy_RLHF

Pipeline for generation of synthetic Singapore political context reinforcment learning data pairs.

This dataset has three main objectives it tries to steer the model towards:
- Policy stance
- Ethics 
- Senstive questions
in the Singaporean context

The generation of synthetic data is broken down into two main stages:
- Question generation
- Answer generation

A few sensitive political and ethical issues in singapore are selected manually first, before passing into LLMs to generate questions on these topics.
These questions are then answered by LLMs to generate the chosen rejected pairs for Direct Preference Optimization (DPO).

Scaling of data was done in two general ways:
- Using different models
- Generating a few responses per prompt (while using low temperature)

## Quickstart
1. create a venv

```python
python3 -m venv venv 
source venv/bin/activate
```

2. download python libraries

```python
pip install -r requirements.txt
```

3. create environment variables

put your environment variables into a .env file

```python
HUGGINGFACE_TOKEN = "..." 
OPENROUTER_API_KEY = "..."
```

uploading to requiremtns.txt
```
pip freeze | grep -vE 'apturl|python-apt|gi|pygobject' > requirements.txt
```

Frameworks/libraires:
- Distilab
- Langchain
