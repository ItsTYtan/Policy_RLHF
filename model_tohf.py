from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

from huggingface_hub import create_branch, create_tag

directory = '/home/tytan216/volume/tzeyoung/Policy_RLHF/models/Qwen/Qwen2.5-1.5B-Instruct-SFT' 

for f in sorted(os.listdir(directory), reverse=True):
    print(f)
    create_branch("ItsTYtan/Qwen2.5_1.5b_policytuned", repo_type="model", branch=f, exist_ok=True)

    api.upload_folder(
        folder_path=directory + "/" + f,
        repo_id="ItsTYtan/Qwen2.5_1.5b_policytuned",
        repo_type="model",
        revision=f,
    )