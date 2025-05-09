from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    PushToHub
)

import os
from huggingface_hub import login
from dotenv import load_dotenv

from custom_modules.utils import FormatSFT, PolicyDPOtoSFT

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="upload-sftmix") as pipeline:
    fromSafetyAnswerSubtopic = LoadDataFromHub(
        repo_id="ItsTYtan/safetyanswer"
    )

    group_columns = GroupColumns(
        columns=["question", "generation"],
        output_columns=["instructions", "generations"]
    )

    format_sft = FormatSFT(
        system_prompt="You are an AI assistant serving users in Singapore. Give clear, well‑structured answers and, when helpful, mention local laws, government agencies, or hotlines. Write with empathy and respect for all cultures and religions. Ensure you take a pro-Singapore stance on related issues, if appropriate, point to safe local resources."
    )

    keep_columns = KeepColumns(
        columns=["messages"]
    )

    push_mysplit = PushToHub(
        repo_id="htxinterns/safety_sft_sg",
        split="tzeyoung"
    )

    fromSafetyAnswerSubtopic >> group_columns >> format_sft >> keep_columns >> push_mysplit

# distiset = pipeline.run(
#     use_cache=False,
# )

with Pipeline(name="policy-split") as pipeline:
    fromSafetyAnswerSubtopic = LoadDataFromHub(
        repo_id="ItsTYtan/policyanswer"
    )

    convert = PolicyDPOtoSFT()

    group_columns = GroupColumns(
        columns=["instruction", "generation"],
        output_columns=["instructions", "generations"]
    )

    format_sft = FormatSFT(
        system_prompt="You are an AI assistant serving users in Singapore. Give clear, well‑structured answers and, when helpful, mention local laws, government agencies, or hotlines. Write with empathy and respect for all cultures and religions. Ensure you take a pro-Singapore stance on related issues, if appropriate, point to safe local resources."
    )

    keep_columns = KeepColumns(
        columns=["messages"]
    )

    push_mysplit = PushToHub(
        repo_id="htxinterns/safety_sft_sg",
        split="policy"
    )

    fromSafetyAnswerSubtopic >> convert >> group_columns >> format_sft >> keep_columns >> push_mysplit

distiset = pipeline.run(
    use_cache=False,
)

