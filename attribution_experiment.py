import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
import os
import random
import json
import pickle
import argparse
import gc
from typing import Optional

import mypkg.whitebox_infra.attribution as attribution
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.data_utils as data_utils
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.interp_utils as interp_utils
import mypkg.pipeline.setup.dataset as dataset_setup
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
from mypkg.eval_config import EvalConfig


def main(
    eval_config: EvalConfig,
    chosen_layer_percentage: int,
    output_dir: str,
    override_trainer_id: Optional[int] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    eval_config.model_name = eval_config.model_names_to_iterate[0]
    assert len(eval_config.model_names_to_iterate) == 1, "Only one model name allowed"

    eval_config.anti_bias_statement_file = (
        eval_config.anti_bias_statement_files_to_iterate[0]
    )
    assert len(eval_config.anti_bias_statement_files_to_iterate) == 1, (
        "Only one anti-bias statement file allowed"
    )

    eval_config.job_description_file = eval_config.job_description_files_to_iterate[0]
    assert len(eval_config.job_description_files_to_iterate) == 1, (
        "Only one job description file allowed"
    )

    model = AutoModelForCausalLM.from_pretrained(
        eval_config.model_name, torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(eval_config.model_name)

    trainer_id = 2

    if eval_config.model_name == "google/gemma-2-27b-it":
        gradient_checkpointing = True
        batch_size = 1
        trainer_id = 131
    elif eval_config.model_name == "mistralai/Mistral-Small-24B-Instruct-2501":
        gradient_checkpointing = False
        batch_size = 2
    elif eval_config.model_name == "google/gemma-2-2b-it":
        gradient_checkpointing = False
        batch_size = 1
        trainer_id = 65
    elif eval_config.model_name == "google/gemma-2-9b-it":
        gradient_checkpointing = False
        batch_size = 2
        trainer_id = 131
    else:
        gradient_checkpointing = False
        batch_size = 4

    if override_trainer_id is not None:
        trainer_id = override_trainer_id

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    chosen_layer_percentages = [chosen_layer_percentage]

    chosen_layers = []
    for layer_percent in chosen_layer_percentages:
        chosen_layers.append(
            model_utils.MODEL_CONFIGS[eval_config.model_name]["layer_mappings"][
                layer_percent
            ]["layer"]
        )

    sae = model_utils.load_model_sae(
        eval_config.model_name,
        device,
        dtype,
        chosen_layer_percentages[0],
        trainer_id=trainer_id,
    )

    submodules = [model_utils.get_submodule(model, chosen_layers[0])]

    df = dataset_setup.load_raw_dataset()

    industry = "INFORMATION-TECHNOLOGY"
    downsample = eval_config.downsample
    random_seed = eval_config.random_seed

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    df = dataset_setup.filter_by_industry(df, industry)

    if downsample:
        df = dataset_setup.balanced_downsample(df, downsample, random_seed)

    output_dir = os.path.join(output_dir, eval_config.model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(
        output_dir,
        f"{eval_config.anti_bias_statement_file.replace('.txt', '')}_trainer_{trainer_id}_model_{eval_config.model_name.replace('/', '_')}_layer_{chosen_layer_percentages[0]}_attrib_data.pt",
    )
    all_data = {}

    all_data["config"] = {
        "model_name": eval_config.model_name,
        "layer": chosen_layers[0],
        "bias_categories": eval_config.bias_types_to_iterate,
        "anti_bias_statement_file": eval_config.anti_bias_statement_file,
        "downsample": downsample,
        "random_seed": random_seed,
        "batch_size": batch_size,
        "chosen_layer_percentage": chosen_layer_percentage,
        "trainer_id": trainer_id,
    }

    # --- Loop over the bias categories ---
    for bias_category in eval_config.bias_types_to_iterate:
        prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(df, eval_config)

        train_texts, train_labels, train_resume_prompt_results = (
            hiring_bias_prompts.process_hiring_bias_resumes_prompts(
                prompts, eval_config.model_name, bias_category
            )
        )

        dataloader = data_utils.create_simple_dataloader(
            train_texts,
            train_labels,
            train_resume_prompt_results,
            eval_config.model_name,
            device,
            batch_size=batch_size,
            max_length=eval_config.max_length,
        )

        # Build the custom loss function
        yes_vs_no_loss_fn = attribution.make_yes_no_loss_fn(
            tokenizer,
            device=device,
            yes_candidates=["yes", " yes", "Yes", " Yes", "YES", " YES"],
            no_candidates=["no", " no", "No", " No", "NO", " NO"],
        )

        gc.collect()
        torch.cuda.empty_cache()

        total_len = 0
        for prompt in train_texts:
            total_len += len(prompt)
        print(total_len)

        model.eval()

        attribution_data = attribution.get_effects(
            model,
            tokenizer,
            sae,
            dataloader,
            yes_vs_no_loss_fn,
            submodules,
            chosen_layers,
            device,
        )

        # Print peak memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            print(f"Peak CUDA memory usage: {peak_memory:.2f} MB")

        effects_F = attribution_data.pos_effects_F - attribution_data.neg_effects_F
        error_effect = (
            attribution_data.pos_error_effect - attribution_data.neg_error_effect
        )

        top_k_ids = effects_F.abs().topk(20).indices
        print(top_k_ids)

        top_k_vals = effects_F[top_k_ids]
        print(top_k_vals)

        print(error_effect)

        all_data[bias_category] = asdict(attribution_data)
        torch.save(all_data, output_filename)

    torch.save(all_data, output_filename)
    print(f"Experiment finished. Results saved to {output_filename}")

    return all_data
