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

## Meeting Notes and thought process

### (06/06/2025)
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

### (09/06/2025)
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

### (11/06/2025)
Identified some starting patterns for a speaker, using Ms Rahayu Mahzam as an example:
- The Minister of State for Health (Ms Rahayu Mahzam) (for the Minister for Health)\n:
- Ms Rahayu Mahzam\n: 
- (Ms Rahayu Mahzam)\n:

Speech ends with another person speaking or due to end of string

Thus, 2 main regex patterns were used to extract speeches:
```python
  rf"{mp}\n:.*?\n[^\d]*?\n:"
```

```python
  rf"\({mp}\)[^\d]*?\n:.*?\n[^\d]*?\n:"
```

{mp} denotes a placeholder for a mp name

Speech ending due to end of string is checked if the 2 above regex patterns do not return matches. Slightly modified version of the regex above are used, without the "?\n[^\d]*?\n:" at the end
of each regex pattern.

### (12/06/2025)
Sucessfully extracted speeches from individual speakers and their corresponding claims.

One example:
```json
{
  "file": "2025-04-08.json",
  "section_title": "Proposal to Reduce Levy for Hiring of First Migrant Domestic Workers 
  to $60 for All Households with One Singapore Citizen",
  "speaker": "Ms Gan Siow Huang",
  "speech": "Ms Gan Siow Huang\n: I thank the Member for raising the two supplementary 
  questions. I think our policies have to be taken in our local context – looking at 
  the lifespan of seniors in Singapore and also to calibrate all our policies according 
  to our local needs. We will, however, continue to review our policies so that they 
  are kept relevant and also support households that are in need.\nTo the question of 
  reducing or providing concessionary levies for all households in Singapore hiring 
  MDWs, I would like to reiterate the point that the purpose of the levy as a pricing 
  mechanism is to regulate the number of MDWs in Singapore. Today, we already have a 
  growing and quite a large number of MDWs. We need to have some lever to be able to 
  regulate the overpopulation of MDWs to keep it sustainable.\nIf there are households 
  that the Member is aware of who are in financial need and require domestic help, 
  please highlight to MOM. We will look at the case.\n1.01 pm\nMr Speaker\n:",
  "claims": [
    "Policies must be tailored to the local context, considering the lifespan of seniors 
    in Singapore.",
    "The government will continue to review policies to ensure they remain relevant and 
    supportive of households in need.",
    "The purpose of the levy on foreign domestic workers (MDWs) is to regulate their 
    numbers and maintain sustainability.",
    "Households in financial need requiring domestic help should be highlighted to 
    the Ministry of Manpower (MOM)."
  ]
}
```

Some speeches do not contain useful information, and the claims array is made to be empty
```json 
{
  "file": "2025-04-08.json",
  "section_title": "Increase in Water Seepage Issues in HDB Flats and Adequacy of 
  Staff Assigned to Rectify These Issues",
  "speaker": "Ms Sim Ann",
  "speech": "(Ms Sim Ann) (for the Minister for National Development)\n: Mr Speaker, 
  Sir, may I have your permission to give a combined reply to Question Nos 3 through 
  6 in today’s Order Paper?\nMr Speaker\n:",
  "claims": []
}
```

However, some speeches contain useful info, but claims array is still empty
```json
{
  "file": "2025-04-08.json",
  "section_title": "Target Date to Revise Penalties for Animal Cruelty and 
  Introduce Failure in Duty of Care Provisions",
  "speaker": "Mr Tan Kiat How",
  "speech": "Mr Tan Kiat How\n: Sir, on the two questions that Mr Chua has raised, 
  let me take them in turn.\nOn the first one around how we ensure compliance with 
  the\nCode of Animal Welfare (for the Pet Industry), t\nhese are guidelines that 
  we put forward. And if there are members of the public, industry players or operators 
  who want to report any non-compliance, please let us know. NParks, as part of its 
  broader licensing framework and regulatory ambit, will do spot checks and take a 
  look at some of these places.\nOn the second point on the disqualification order 
  (DO), just to confirm with Mr Chua that he was asking about DO? No? I could not 
  hear the question.\nMr Chua Kheng Wee Louis\n:",
  "claims": []
}
```

Suspect is due to API limits on rate of incoming requests.