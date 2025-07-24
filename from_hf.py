from huggingface_hub import snapshot_download, login
import os
from dotenv import load_dotenv

from distilabel.pipeline import Pipeline
from distilabel.steps import make_generator_step
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    KeepColumns,
    ExpandColumns,
    PushToHub
)

load_dotenv()
login(os.getenv("HUGGINGFACE_TOKEN"))

# snapshot_download(
#     repo_id="ItsTYtan/Qwen2.5_1.5b_policytuned", 
#     repo_type="model",
#     revision="checkpoint-40000",
#     local_dir="./models/Qwen2.5_1.5b_policytuned/checkpoint-40000"
# )

with Pipeline("pipe-name", description="My first pipe") as pipeline:
    fromhf = LoadDataFromHub(
        repo_id="htxinterns/axiom",
    )

distiset = pipeline.run()
print(distiset)
splits = distiset["default"]["train"].train_test_split(train_size=0.8)

splits.push_to_hub(
    "ItsTYtan/axiom",
    commit_message="Initial commit",
    private=True,
    token=os.getenv("HF_TOKEN"),
)