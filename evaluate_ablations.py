import json
import os

########## SET GPUS ###########
gpu = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
from accelerate.test_utils.testing import get_backend
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(os.getenv("HUGGINGFACE_TOKEN"))

device, _, _ = get_backend()

def evaluate_perplexity(modelpath, max_length, stride, queries, generations):
    with open(modelpath + "/../ablation_params.json", "r") as ablation_file:
        data = json.load(ablation_file)
        isPeft = "peft-config" in data
        baseModelName = data["model"]

    if isPeft:
        baseModel = AutoModelForCausalLM.from_pretrained(
            baseModelName,
            device_map='auto'
        ).to(device)
        model = PeftModel.from_pretrained(baseModel, modelpath)
        tokenizer = AutoTokenizer.from_pretrained(baseModelName)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            device_map='auto'
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(modelpath)

    nll_sum = 0.0
    n_tokens = 0

    for query, generation in tqdm(list(zip(queries, generations))):
        query_encodings = tokenizer(query, return_tensors="pt")
        encodings = tokenizer(query + generation, return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        prev_end_loc = query_encodings.input_ids.size(1)

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            nll_sum += neg_log_likelihood * num_valid_tokens
            n_tokens += num_valid_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    return torch.exp(avg_nll).item()


if __name__ == "__main__":

    ####################################################
    #                EVALUATE ABLATIONS                #
    ####################################################

    directory = "models"
    dataset = load_dataset("ItsTYtan/axiom")
    queries = list(map(lambda msg: msg[1]["content"], dataset["test"]["messages"]))
    generations = list(map(lambda msg: msg[2]["content"], dataset["test"]["messages"]))

    with open("evaluation_results/abalation_axiom_report.json", "w") as new_report_file:
        new_report = {}
        for ablation in sorted(os.listdir(directory)):
            ablation_log = {}
            for chkpt in sorted(os.listdir(directory + "/" + ablation)):
                if not chkpt.startswith("checkpoint-"):
                    continue 
                fullpath = directory + "/" + ablation + "/" + chkpt
                ablation_log[chkpt] = evaluate_perplexity(
                    modelpath=fullpath,
                    max_length=8192,
                    stride=1,
                    queries=queries,
                    generations=generations
                )
            new_report[ablation] = ablation_log
        json.dump(new_report, new_report_file, indent=2)

    ####################################################
    #               EVALUATE BASE MODELS               #
    ####################################################    
        
    # directory = "base_models"
    # dataset = load_dataset("ItsTYtan/axiom")
    # queries = list(map(lambda msg: msg[1]["content"], dataset["test"]["messages"]))
    # generations = list(map(lambda msg: msg[2]["content"], dataset["test"]["messages"]))

    # with open("evaluation_results/base_model_axiom_report.json", "w") as new_report_file:
    #     new_report = {}
    #     for base_model in os.listdir(directory):
    #         fullpath = directory + "/" + base_model
    #         ppl = evaluate_perplexity(
    #             modelpath=fullpath,
    #             max_length=8192,
    #             stride=1,
    #             queries=queries,
    #             generations=generations
    #         )
    #         new_report[base_model] = ppl
    #     json.dump(new_report, new_report_file, indent=2)