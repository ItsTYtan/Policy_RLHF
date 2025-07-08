from huggingface_hub import snapshot_download, login
import os
from dotenv import load_dotenv

login(os.getenv("HUGGINGFACE_TOKEN"))

snapshot_download(
    repo_id="ItsTYtan/Qwen2.5_1.5b_policytuned", 
    repo_type="model",
    revision="checkpoint-20000",
    local_dir="./models/Qwen2.5_1.5b_policytuned/checkpoint-20000"
)