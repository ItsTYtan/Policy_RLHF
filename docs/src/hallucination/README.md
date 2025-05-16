# Hallucination evaluation

This page details my thinking process for developing a hallucination filter for synthetic data generation. 

## Rationale
In refining the htllama dataset, the output was prompted to be more detailed and of greater length. This, by intuition, poses a bigger threat of hallucination, 
since Singapore context data may not be trained into the models used to refine these datasets.

## First steps
Due to the black box/grey box nature of llms, and the fact that the synthetic data is generated from llm providers using API calls, log probablities are perhaps
the most suitable way to assess the confidence of an llm and therefore detect hallucinations. 

### Sources
1. [Look Before You Leap: An Exploratory Study of Uncertainty Analysis for Large Language Models](https://arxiv.org/html/2307.10236v4)
2. [Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach](https://arxiv.org/html/2404.15993v4)