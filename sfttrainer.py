import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import wandb

load_dotenv()
wandb.login()

models = [
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
]

os.environ["WANDB_PROJECT"]="sft_ablation"

dataset = load_dataset("htxinterns/safety_sft_sg", split="tzeyoung")

for model in models:
    model = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    training_args = SFTConfig(
        output_dir="/models",
        report_to="wandb",
        run_name=model,
        logging_steps=10,
        output_dir="models/" + model + "-SFT",
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=8,
        padding_value=tokenizer.eos_token_id,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=training_args
    )

    trainer.train()