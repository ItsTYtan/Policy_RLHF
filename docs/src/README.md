# Introduction - AXIOM
AXIOM is an automated pipeline for generation of synthetic SFT data on Singapore policies.

## Rationale
Policies change and models should reflect the latest updates in policy changes. Thus, AXIOM automates this process of updating a model's political alignment

## Methodology
This section outlines the thought process in creating AXIOM

The whole process of automating the generation of a dataset can be roughly broken down into 3 stages:
1. Extraction of new policies from the web
2. Creating a database that can handle updates of new policies that were extracted
3. Generation of the SFT dataset from the database.

# Meeting Notes and 

## (06/06/2025)
Meeting today gave an overview of how to go about creating axiom, and how to start on the first part which is the extraction of new policies fom the web.
We chose Hansard as the source of policy information from the web. Hansard provides publicly available comprehensive transcripts of paliamentary debates in Singapore.

The information deemed relevant from paliamentary debates are the policies discussed, the final decision made for each policy, and the claims supporting the final decision as well as the claims against the final decision. In short, the information extracted can be visualized as many json objects, each one with the format below:

```json
  {
    "policy": "[the policy discussed in the debate]",
    "paliamentary debate": "[containing the details of which debate the policy was discussed in]",
    "final decision": "[the final decision of the policy discussed]",
    "claims for": [
      "claim 1 supporting the final decision",
      "claim 2 supporting the final decision",
      "..."
    ],
    "claims against": [
      "claim 1 against the final decision",
      "claim 2 against the final decision",
      "..."
    ],
  }
```

As multiple policies may be discussed in a paliamentary debate, a LLM is first tasked to extract the policies discussed in the debate. For each policy extracted, another LLM is tasked to extract
the final decision, the claims for and against as well as the ministries involved.

## (09/06/2025)
A few issues surfaced, below describes the issues and the proposed workarounds

### 1. Hansard paliamentary debates do not fit into the context length of Sagemaker hosted models.
A few chunking strategies were proposed, and 2 are to be tried together.
- Chunking by topic: within each sitting, there are clear separations between different topics discussed
- Chunking by speaker: extract the relevant content of each speaker through regex

Chunking by speaker presents the problem of not being able to extract the final decision of the policy discussed. Currently am unsure if the final decision is neccessary information for generating the SFT data.

### 2. Schema modifications 
After taking a look at the generation of claims, final decision and policy, it seemed that the claims also contained policy information and could be used to generate the final SFT data instead. Some paliamentary debates were also clarifying and not an actual debate about a policy. Hence, some modifications to the schema were suggested:

#### Including the speaker into each json object
This allows us to determine the credibility of claims. For simplicity's sake, it was decided that anything said by the current ruling party is the truth.

#### Merging of claims for and against:
Not much point in separating the claims for and against if whether the deciding factor whether each claim is to be used is if the speaker making the claim is in the current ruling party or not.

new json format:
```json
  {
    "policy": "[the policy discussed in the debate]",
    "date": "[originally 'paliamentary debate', but I think only the data of the debate is important anyway]",
    "speaker": "[speaker of the claims below]"
    "claims": [
      "claim 1",
      "claim 2",
      "..."
    ],
  }
```

Arka also showed the new Qwen3 embedding model [link](https://qwenlm.github.io/blog/qwen3-embedding/) we could use in the future to validate our data

## (11/06/2025)
Identified 2 main starting patterns for a speaker, using Ms Rahayu Mahzam as an example:
1. The Minister of State for Health (Ms Rahayu Mahzam) (for the Minister for Health)\n:
2. Ms Rahayu Mahzam\n: 

The end of a speech ends in 3 main ways:
1. Ending with Mr speaker: 
  \nMr Speaker\n:
2. Ending with another Mp start of speech: 
  \nMr Alvin Tan\n:
3. Ending due to end of file.

Thus, 2 main regex patterns were used to extract speeches
```python
  rf"{mp}\n:.*\n{nextspeaker}\n:"
```

```python
  rf"({mp})\s(.*)\n:.*\n{nextspeaker}\n:"
```

{mp} denotes a placeholder for a mp name
{nextspeaker} denotes the next speaker, which can be another mp, Mr Speaker or nobody (will be represented as an empty string)

Thus, speeches are extracted through generating regex patterns by the cartesian product of mps with nextspeakers. 
The EOF end of speech need to be treated differently. After every regex pattern is tested, if no match is found, the EOF regex, which is essentialy rf"{mp}\n:.*" or rf"({mp})\s(.*)\n:.*"
will then be used to check for match.