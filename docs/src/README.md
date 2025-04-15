# Introduction - Singapore RL

Pipeline for generation of synthetic Singapore political context reinforcment learning data pairs.

This dataset has three main objectives it tries to steer the model towards:
- Policy stance
- Ethics 
- Preventing hallucinations
in the Singaporean context

The generation of synthetic data is broken down into two main stages:
- Question generation
- Answer generation

A few sensitive political and ethical issues in singapore are selected manually first, before passing into LLMs to generate questions on these topics.
These questions are then answered by LLMs to generate the chosen rejected pairs for Direct Preference Optimization (DPO).

Scaling of data was done in two general ways:
- Using different models
- Generating a few responses per prompt (while using low temperature)
