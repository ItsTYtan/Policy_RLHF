# Architecture of pipelines and databases

Current SQL database schema:
![alt text](./images/sql_architecture.png)


Current vector database collections:

| Collection name               | Embedding size | Embedding model         |
|-------------------------------|----------------|-------------------------|
| hansard_speeches              | 4096           | Qwen/Qwen3-Embedding-8B |
| summarized-speech-embeddings  | 4096           | Qwen/Qwen3-Embedding-8B |
| summarized-section-embeddings | 4096           | Qwen/Qwen3-Embedding-8B |