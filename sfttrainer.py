import json
import subprocess
import sys
from dotenv import load_dotenv
import huggingface_hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import os
import wandb
load_dotenv()
wandb.login()
huggingface_hub.login()

###########################################
# ABLATION PARAMETERS
###########################################
params = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "learning-rate": 1e-9,
    "comments": "50/50 mix of general qna and policy dataset"
}

###########################################
# TRAINING PARAMETERS
###########################################
gpu = "0"
gpu_utilization = 0.5
epochs = 0.6
save_steps = 10000
save_total_limit = 6
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
logging_steps = 10

learning_rate = params["learning-rate"]

experiment_no = len(os.listdir("./models"))
output_dir = "models/" + "experiemnt-" + str(experiment_no)

###########################################
# DATASET CONFIGURATION
###########################################
dataset = load_dataset("htxinterns/axiom", split="train")


###########################################
# WANDB PARAMETERS
###########################################
project_name = "sft_ablation"


if __name__ == "__main__":
    # Write ablation params to config file
    with open(output_dir + "ablation_params.json", "w") as f:
        json.dump(params, f, ensure_ascii=False)


    # Set up environment variables and hardware
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.cuda.set_per_process_memory_fraction(gpu_utilization, 0)


    model = AutoModelForCausalLM.from_pretrained(params["model"])
    tokenizer = AutoTokenizer.from_pretrained(params["model"])

    training_args = SFTConfig(
        report_to="wandb",
        run_name=params["model"],
        logging_steps=logging_steps,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,  
        gradient_accumulation_steps=gradient_accumulation_steps, 
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        learning_rate=learning_rate,
        num_train_epochs=epochs
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    # Convert to gguf format
    for checkpoint in os.listdir(output_dir):
        subprocess.run([sys.executable, "./llama.cpp/convert_hf_to_gguf.py", output_dir + "/" + checkpoint])
