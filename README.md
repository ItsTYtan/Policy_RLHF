# Policy_RLHF

Pipeline for generation of synthetic Singapore political context reinforcment learning data pairs.

## Quickstart
1. create a venv

```python
python -m venv venv 
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



Frameworks/libraires:
- Distilab
- Langchain
