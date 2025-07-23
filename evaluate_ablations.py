import json
import os
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend()

directory = "models"

with open("abalation_report.json", "r") as f:
    old_report = json.load(f)

os.remove("abalation_report.json")

with open("abalation_report.json", "w") as new_report:
    for ablation in os.listdir(directory):
        ablation_log = {}
        for chkpt in os.listdir(directory + chkpt):

            if old_report["ablation"]["chkpt"]:
                continue

            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to(device)
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

            dataset = load_dataset("htxinterns/axiom")

            queries = list(map(lambda msg: msg[1]["content"], dataset["train"]["messages"]))

            generations = list(map(lambda msg: msg[2]["content"], dataset["train"]["messages"]))

            max_length = 8192
            stride = 1

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
            ppl = torch.exp(avg_nll)
            ablation_log[chkpt] = ppl

        new_report[ablation] = ablation_log
        