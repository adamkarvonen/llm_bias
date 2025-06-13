import os

# Must set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import pandas as pd
from tqdm import tqdm
import sys
import os
from datetime import datetime
from transformers import AutoTokenizer
from tabulate import tabulate
import asyncio
import json
import random
from datasets import load_dataset
import torch
from dataclasses import asdict
import itertools
import gc
import time
from typing import Optional, Any
from enum import Enum
import pickle

from mypkg.eval_config import EvalConfig, InferenceMode, FrozenEvalConfig
import mypkg.pipeline.setup.dataset as dataset_setup
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.pipeline.infra.model_inference as model_inference
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.intervention_hooks as intervention_hooks
import mypkg.whitebox_infra.logit_lens as logit_lens


OPENROUTER_NAME_LOOKUP = {
    "mistralai/Ministral-8B-Instruct-2410": "mistralai/ministral-8b",
    "mistralai/Mistral-Small-24B-Instruct-2501": "mistralai/mistral-small-3.1-24b-instruct",
}


def save_evaluation_results(
    results,
    bias_scores,
    bias_probs,
    output_filepath: str,
    timestamp: str,
    eval_config: EvalConfig,
) -> dict[str, Any]:
    # Build a big dictionary with all data:
    # 1) Metadata
    evaluation_data = {
        "metadata": {
            "timestamp": timestamp,
        },
        # 2) Full list of results
        "results": [],
        # 3) Bias summary
        "bias_scores": bias_scores,
        "bias_probs": bias_probs,
        "eval_config": eval_config.model_dump(),
    }

    # For each ResumePromptResult in `results`, store relevant info
    # so you can fully reconstruct or debug each inference later.
    for item in results:
        evaluation_data["results"].append(asdict(item))

    with open(output_filepath, "wb") as f:
        pickle.dump(evaluation_data, f)

    print(f"\nSaved evaluation data to: {output_filepath}")

    return evaluation_data


async def main(
    eval_config: EvalConfig,
    timestamp: str,
) -> dict[str, Any]:
    """
    python mypkg/main_paper_dataset.py --config configs/base_experiment.yaml
    python mypkg/main_paper_dataset.py --config configs/openrouter_experiment.yaml

    python mypkg/main_paper_dataset.py --config configs/base_experiment.yaml ; python mypkg/main_paper_dataset.py --config configs/base_experiment_intervention.yaml ; python mypkg/mmlu_eval.py ; python mypkg/main_paper_dataset.py --config configs/base_experiment_anthropic_eval.yaml
    """

    start_time = time.time()

    # Set the seed for reproducibility
    random.seed(eval_config.random_seed)
    torch.manual_seed(eval_config.random_seed)

    print("Loading dataset...")
    if eval_config.anthropic_dataset:
        df = dataset_setup.load_full_anthropic_dataset(
            downsample_questions=eval_config.downsample
        )
        eval_config.system_prompt_filename = "yes_no_anthropic.txt"
        eval_config.batch_size_multiplier = 8
    else:
        df = dataset_setup.load_raw_dataset()

        if eval_config.industry:
            print(f"Filtering for industry: {eval_config.industry}")
            df = dataset_setup.filter_by_industry(df, eval_config.industry)
        else:
            print("No industry filter applied.")

        if eval_config.downsample is not None:
            df = dataset_setup.balanced_downsample(
                df, eval_config.downsample, eval_config.random_seed
            )

    # Print demographic distribution before sampling
    print("\n[FULL DATASET:] Demographic distribution:")
    print("Gender:", df["Gender"].value_counts())
    print("--------------------------------")
    print("Race:", df["Race"].value_counts())
    print("--------------------------------")

    os.makedirs(eval_config.score_output_dir, exist_ok=True)

    if eval_config.system_prompt_filename in [
        "yes_no_cot.txt",
        "yes_no_qualifications.txt",
    ]:
        # May have to increase this, especially for any reasoning models
        max_completion_tokens = 200
    else:
        max_completion_tokens = 5

    all_results = {}

    for (
        model_name,
        anti_bias_statement_file,
        job_description,
        scale,
        bias_type,
    ) in itertools.product(
        eval_config.model_names_to_iterate,
        eval_config.anti_bias_statement_files_to_iterate,
        eval_config.job_description_files_to_iterate,
        eval_config.scales_to_iterate,
        eval_config.bias_types_to_iterate,
    ):
        eval_config.model_name = model_name
        eval_config.anti_bias_statement_file = anti_bias_statement_file
        eval_config.job_description_file = job_description
        eval_config.scale = scale
        eval_config.bias_type = bias_type

        frozen_eval_config = FrozenEvalConfig.model_validate(eval_config.model_dump())

        print(f"Running with anti-bias statement: {anti_bias_statement_file}")
        print(f"Running with job description: {job_description}")
        print(f"Running with model: {model_name}")
        print(f"Running with scale: {scale}")
        print(f"Running with bias type: {bias_type}")

        results_filename = f"score_results_{anti_bias_statement_file}_{job_description}_{model_name}_{str(scale).replace('.', '_')}_{bias_type}.pkl".replace(
            ".txt", ""
        ).replace("/", "_")

        output_dir = os.path.join(
            frozen_eval_config.score_output_dir,
            frozen_eval_config.inference_mode,
            model_name.replace("/", "_"),
        )
        os.makedirs(output_dir, exist_ok=True)
        results_output_filepath = os.path.join(output_dir, results_filename)

        file_key = results_output_filepath

        if (
            os.path.exists(results_output_filepath)
            and not frozen_eval_config.overwrite_existing_results
        ):
            print(f"Skipping {results_filename} because it already exists")
            continue

        if frozen_eval_config.anthropic_dataset:
            prompts = hiring_bias_prompts.create_all_prompts_anthropic(
                df, frozen_eval_config, add_system_prompt=True
            )
        else:
            prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(
                df, frozen_eval_config
            )
            original_length = len(prompts)
            prompts = [
                prompt
                for prompt in prompts
                if len(prompt.resume) <= frozen_eval_config.max_length_chars
            ]
            print(
                f"Filtered {original_length - len(prompts)} prompts, now {len(prompts)} prompts"
            )

        gc.collect()
        torch.cuda.empty_cache()

        if frozen_eval_config.inference_mode == InferenceMode.GPU_FORWARD_PASS.value:
            batch_size = (
                model_utils.MODEL_CONFIGS[model_name]["batch_size"]
                * frozen_eval_config.batch_size_multiplier
            )
            results = model_inference.run_single_forward_pass_transformers(
                prompts,
                model_name,
                eval_config=frozen_eval_config,
                batch_size=batch_size,
                collect_activations=False,
            )
        elif (
            frozen_eval_config.inference_mode
            == InferenceMode.PROJECTION_ABLATIONS.value
        ):
            batch_size = (
                model_utils.MODEL_CONFIGS[model_name]["batch_size"]
                * frozen_eval_config.batch_size_multiplier
            )

            ablation_vectors = model_inference.get_ablation_vectors(
                model_name,
                bias_type,
                eval_config=frozen_eval_config,
                batch_size=batch_size,
            )

            results = model_inference.run_single_forward_pass_transformers(
                prompts,
                model_name,
                eval_config=frozen_eval_config,
                batch_size=batch_size,
                ablation_vectors=ablation_vectors,
                ablation_type="projection_ablations",
                orthogonalize_model=frozen_eval_config.orthogonalize_model,
            )

            # results = model_inference.run_inference_transformers(
            #     prompts,
            #     model_name,
            #     batch_size=batch_size,
            #     ablation_vectors=ablation_vectors,
            # )
        elif frozen_eval_config.inference_mode == InferenceMode.GPU_INFERENCE.value:
            batch_size = (
                model_utils.MODEL_CONFIGS[model_name]["batch_size"]
                * frozen_eval_config.batch_size_multiplier
            )
            results = model_inference.run_inference_transformers(
                prompts,
                model_name,
                batch_size=batch_size,
                max_new_tokens=max_completion_tokens,
            )
        elif frozen_eval_config.inference_mode == InferenceMode.VLLM_INFERENCE.value:
            results = model_inference.run_inference_vllm(
                prompts,
                model_name,
                max_new_tokens=max_completion_tokens,
            )
        elif frozen_eval_config.inference_mode == InferenceMode.LOGIT_LENS.value:
            batch_size = (
                model_utils.MODEL_CONFIGS[model_name]["batch_size"]
                * frozen_eval_config.batch_size_multiplier
            )
            results = logit_lens.run_logit_lens(
                prompts,
                model_name,
                add_final_layer=False,
                batch_size=batch_size,
                use_tuned_lenses=True,
            )
        elif frozen_eval_config.inference_mode == InferenceMode.OPEN_ROUTER.value:
            # Use the lookup table if the model name is present
            api_model_name = OPENROUTER_NAME_LOOKUP.get(model_name, model_name)
            results = await model_inference.run_model_inference_openrouter(
                prompts, api_model_name, max_completion_tokens=max_completion_tokens
            )
        else:
            raise ValueError(
                f"Unhandled inference mode: {frozen_eval_config.inference_mode}"
            )

        # Quick and dirty check to see if prompts are the same from run to run
        total_len = 0
        for result in results:
            total_len += len(result.prompt)
        print(f"Total length of prompts: {total_len}")

        bias_scores = hiring_bias_prompts.evaluate_bias(
            results, system_prompt_filename=frozen_eval_config.system_prompt_filename
        )
        print(bias_scores)

        if frozen_eval_config.inference_mode in [
            InferenceMode.GPU_FORWARD_PASS.value,
            InferenceMode.PERFORM_ABLATIONS.value,
            InferenceMode.PROJECTION_ABLATIONS.value,
        ]:
            bias_probs = hiring_bias_prompts.evaluate_bias_probs(results)
            print("\n\n\n", bias_probs)
        else:
            bias_probs = None

        run_results = save_evaluation_results(
            results,
            bias_scores,
            bias_probs,
            results_output_filepath,
            timestamp,
            frozen_eval_config,
        )

        print(f"\nTotal resumes processed: {len(results)}")

        all_results[file_key] = run_results

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            print(f"Peak CUDA memory usage: {peak_memory:.2f} MB")

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return all_results


def parse_overrides(pairs: list[str]) -> dict:
    """Convert ["key=val", "nested.foo=3"] to a flat dict understood by Pydantic."""
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Override '{p}' expects KEY=VALUE")
        k, v = p.split("=", 1)
        out[k] = v
    return out


if __name__ == "__main__":
    # Setup moved here
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        default="configs/base_experiment.yaml",
        help="YAML file with experiment cfg",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Ad-hoc overrides key=val (YAML typed)",
    )
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)

    if args.override:
        import yaml  # local import to avoid mandatory dependency for non-users

        overrides = {
            k: yaml.safe_load(v) for k, v in parse_overrides(args.override).items()
        }
        cfg = cfg.model_copy(update=overrides)

    print(cfg.model_dump())
    results = asyncio.run(main(cfg, timestamp))
