# Policyanswer

Main ideas to generate DPO synthetic data from questions

1. Direct chosen reject pair prompting
Prompt the model to give a good answer and a bad answer.

2. Repeated prompting
Prompt the model to give an acceptable answer, then prompt it again to improve on the answer.

3. UltraFeedback
Get two different models to output a good answer to the question, use another reward LLM to grade the models' answers, choose the model with better score.
More information on UltraFeedback can be found [here](https://arxiv.org/abs/2310.01377)

4. RAG augment on one model
Augment a model with RAG to generate better responses than the other model without RAG

Method 1 can be used to generate contrasting pairs, while methods 2, 3 and 4 can be used to generate closer DPO pairs.

## Scaling the data
Once again using multiple models or increasing the number of generations may help to scale the dataset.