import json
import subprocess
import sys
from dotenv import load_dotenv
from datasets import concatenate_datasets, load_dataset
import huggingface_hub
import os
import wandb
load_dotenv()
wandb.login()
huggingface_hub.login(os.getenv("HUGGINGFACE_TOKEN"))

###########################################
# ABLATION PARAMETERS
###########################################
params = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "learning-rate": 1e-9,
    "comments": "50/50 mix of general qna alpaca dataset and policy dataset. 40000 examples from each dataset"
}

###########################################
# TRAINING PARAMETERS
###########################################
gpu = "7"
gpu_utilization = 0.5
epochs = 0.6
save_steps = 10000
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
logging_steps = 10

learning_rate = params["learning-rate"]

experiment_no = len(os.listdir("./models"))
output_dir = "models/" + "ablation-" + str(experiment_no)

###########################################
# DATASET CONFIGURATION
###########################################
policy_num_examples = 40000
alpaca_num_examples = 40000

policy_dataset = load_dataset("htxinterns/axiom", split="train").shuffle(seed=42).select(range(policy_num_examples))
alpaca_dataset = load_dataset("json", data_files="./datasets/alpaca.jsonl", split="train").shuffle(seed=42).select(range(alpaca_num_examples))

dataset = concatenate_datasets([policy_dataset, alpaca_dataset]).shuffle(seed=42)

###########################################
# WANDB PARAMETERS
###########################################
project_name = "sft_ablation",
run_name = os.path.basename(output_dir)

wandb.init(
    project_name=project_name,
    name=run_name,
    tags=[params["model"], params["learning-rate"]],
    notes=params["comments"]
)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.cuda.set_per_process_memory_fraction(gpu_utilization, 0)

    # Write ablation params to config file
    os.makedirs(os.path.dirname(output_dir + "/" + "ablation_params.json"), exist_ok=True)
    with open(output_dir + "/" + "ablation_params.json", "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


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
