import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import os
import random
import pickle
import itertools

import mypkg.whitebox_infra.attribution as attribution
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.data_utils as data_utils
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.interp_utils as interp_utils
import mypkg.pipeline.setup.dataset as dataset_setup
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
from mypkg.eval_config import EvalConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

model_names = [
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
]

for model_name in model_names:
    if "mistral" in model_name:
        trainer_ids = [0, 1, 2, 3]
    else:
        trainer_ids = [131]

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    gradient_checkpointing = False

    if model_name == "google/gemma-2-27b-it":
        gradient_checkpointing = True
        batch_size = 1
    elif model_name == "mistralai/Mistral-Small-24B-Instruct-2501":
        batch_size = 2
    else:
        batch_size = 4

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    chosen_layer_percentages = [25, 50, 75]

    acts_folder = "max_acts"
    os.makedirs(acts_folder, exist_ok=True)

    for trainer_id, chosen_layer_percentage in itertools.product(
        trainer_ids, chosen_layer_percentages
    ):
        chosen_layer = model_utils.MODEL_CONFIGS[model_name]["layer_mappings"][
            chosen_layer_percentage
        ]["layer"]

        sae = model_utils.load_model_sae(
            model_name, device, dtype, chosen_layer_percentage, trainer_id=trainer_id
        )

        submodules = [model_utils.get_submodule(model, chosen_layer)]

        acts_filename = os.path.join(
            acts_folder,
            f"acts_{model_name}_layer_{chosen_layer}_trainer_{trainer_id}_layer_percent_{chosen_layer_percentage}.pt".replace(
                "/", "_"
            ),
        )

        if not os.path.exists(acts_filename):
            max_tokens, max_acts = interp_utils.get_interp_prompts(
                model,
                submodules[0],
                sae,
                torch.tensor(list(range(sae.W_dec.shape[0]))),
                context_length=128,
                tokenizer=tokenizer,
                batch_size=batch_size * 8,
                num_tokens=30_000_000,
            )
            acts_data = {
                "max_tokens": max_tokens,
                "max_acts": max_acts,
            }
            torch.save(acts_data, acts_filename)
