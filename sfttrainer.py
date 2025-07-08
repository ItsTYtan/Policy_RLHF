import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import wandb

load_dotenv()
wandb.login()

models = [
    # "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-1.5B-Instruct"
]

os.environ["WANDB_PROJECT"]="sft_ablation"
torch.cuda.set_per_process_memory_fraction(0.5, 0)

dataset = load_dataset("htxinterns/axiom", split="train")

for model_name in models:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = SFTConfig(
        report_to="wandb",
        run_name=model_name,
        logging_steps=10,
        output_dir="models/" + model_name + "-SFT",
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=1, 
        save_steps=10000,
        save_total_limit=6,
        learning_rate=1e-9,
        num_train_epochs=0.75
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()