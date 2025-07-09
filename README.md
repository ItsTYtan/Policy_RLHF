# Policy_RLHF

Pipeline for generation of synthetic Singapore political context reinforcment learning data pairs.

## PolicyQuestion

Vary questions based on:
1. topic (self-explanatory)

2. harmful/not harmful

harmful prompt types can be classified into:
    - straight up overtly harmful
    - covert of borderline inputs
    - jailbreak attempts

non harmful prompt types can be classfied into:
    - informational
    - help-seeking
    - action oriented

3. noise
Vary grammar, spelling scentence structure

4. persona use cases for the different departments of hometeam (unconfirmed)

## PolicyAnswer





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
