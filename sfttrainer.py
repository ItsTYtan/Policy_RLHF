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
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "learning-rate": 1e-9,
    "peft-config" : {
        "rank": 8,
        "alpha": 32,
        "dropout": 0.1,
    },
    "comments": "100% policy but with peft",
}

###########################################
# TRAINING PARAMETERS
###########################################
gpu = "4,5,6,7"
# gpu_utilization = 1
epochs = 1
save_steps = 10000
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
logging_steps = 10

learning_rate = params["learning-rate"]

experiment_no = len(os.listdir("./models"))
output_dir = "models/" + "ablation-" + str(experiment_no)

############### PEFT CONFIG ###############
use_peft = True



###########################################
# DATASET CONFIGURATION
###########################################
policy_num_examples = 80000
alpaca_num_examples = 0

policy_dataset = load_dataset("htxinterns/axiom", split="train").shuffle(seed=42).select(range(policy_num_examples))
alpaca_dataset = load_dataset("json", data_files="./datasets/alpaca.jsonl", split="train").shuffle(seed=42).select(range(alpaca_num_examples))

dataset = concatenate_datasets([policy_dataset, alpaca_dataset]).shuffle(seed=42)

###########################################
# WANDB PARAMETERS
###########################################
run_name = os.path.basename(output_dir)

wandb.init(
    project="sft_ablation",
    name=run_name,
    tags=[params["model"], str(params["learning-rate"]), "peft"],
    notes=params["comments"]
)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    # torch.cuda.set_per_process_memory_fraction(gpu_utilization, 0)
    model = AutoModelForCausalLM.from_pretrained(
        params["model"],
        device_map='auto'
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,     
        r=params["peft-config"]["rank"], 
        lora_alpha=params["peft-config"]["alpha"], 
        lora_dropout=params["peft-config"]["dropout"]
    )
    
    if (use_peft):
        model = get_peft_model(model, peft_config)

    if isinstance(model, PeftModel):
        print("✅ Model is using PEFT (e.g., LoRA)")
    else:
        print("❌ Model is NOT using PEFT")
    
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

    try:
        trainer.train()

    except Exception as e:
        print(e)
        sys.exit(1)

    # Convert to gguf format
    for checkpoint in os.listdir(output_dir):
        subprocess.run([sys.executable, "./llama.cpp/convert_hf_to_gguf.py", output_dir + "/" + checkpoint])
    
    # Write ablation params to config file
    os.makedirs(os.path.dirname(output_dir + "/" + "ablation_params.json"), exist_ok=True)
    with open(output_dir + "/" + "ablation_params.json", "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)