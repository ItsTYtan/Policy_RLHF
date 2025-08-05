from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base
from os import listdir

directory = "../models"

models = []
ablations = sorted(listdir(directory))
for ablation in ablations:
    checkpoints = sorted(listdir(directory + "/" + ablation))
    for checkpoint in checkpoints:
        fullpath = directory + "/" + ablation + "/" + checkpoint

        models.append(
            dict(
                type=HuggingFaceCausalLM,
                # Parameters for `HuggingFaceCausalLM` initialization.
                path=fullpath,
                tokenizer_path=fullpath,
                tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
                max_seq_len=2048,
                batch_padding=False,
                # Common parameters shared by various models, not specific to `HuggingFaceCausalLM` initialization.
                abbr=ablation + "_" + checkpoint,            # Model abbreviation used for result display.
                max_out_len=100,            # Maximum number of generated tokens.
                batch_size=16,              # The size of a batch during inference.
                run_cfg=dict(num_gpus=1),   # Run configuration to specify resource requirements.
            )
        )

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets

datasets = mmlu_datasets