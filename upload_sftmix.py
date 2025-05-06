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

from custom_modules.utils import FormatSFT
from templates import SYSTEM_PROMPT_ANSWER

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False)

with Pipeline(name="upload-sftmix") as pipeline:
    fromSafetyAnswerSubtopic = LoadDataFromHub(
        repo_id="ItsTYtan/safetyanswer-subtopic"
    )

    group_columns = GroupColumns(
        columns=["question", "generation"],
        output_columns=["instructions", "generations"]
    )

    format_sft = FormatSFT(
        system_prompt=SYSTEM_PROMPT_ANSWER
    )

    keep_columns = KeepColumns(
        columns=["messages"]
    )

    push_mysplit = PushToHub(
        repo_id="ItsTYtan/safety_sft_sg",
        split="tzeyoung"
    )

    fromSafetyAnswerSubtopic >> group_columns >> format_sft >> keep_columns >> push_mysplit

distiset = pipeline.run(
    use_cache=False,
)