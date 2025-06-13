import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
import pickle
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import gc

from mypkg.eval_config import EvalConfig
from mypkg.pipeline.infra import model_inference
from mypkg.whitebox_infra import model_utils
from mypkg.whitebox_infra import intervention_hooks

if __name__ == "__main__":
    config_file = "configs/base_experiment.yaml"
    cfg = EvalConfig.from_yaml(config_file)
    dtype = torch.bfloat16

    # tasks = [
    #     "mmlu_high_school_statistics",
    #     "mmlu_high_school_computer_science",
    #     "mmlu_high_school_mathematics",
    #     "mmlu_high_school_physics",
    # ]

    tasks = ["mmlu"]

    output_folder = "mmlu_ablation_results"
    os.makedirs(output_folder, exist_ok=True)

    for model_name in cfg.model_names_to_iterate:
        filename = f"{output_folder}/{model_name.replace('/', '_')}.pkl"
        # if os.path.exists(filename):
        #     print(f"Skipping {model_name} because it already exists")
        #     continue

        start_time = time.time()
        print(f"Evaluating {model_name}...")

        batch_size = (
            model_utils.MODEL_CONFIGS[model_name]["batch_size"]
            * cfg.batch_size_multiplier
        )

        ablation_vectors = model_inference.get_ablation_vectors(
            model_name,
            bias_type="all",
            eval_config=cfg,
            batch_size=batch_size,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

        lm_obj = HFLM(model)

        task_manager = lm_eval.tasks.TaskManager()

        base_results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            num_fewshot=0,
            task_manager=task_manager,
        )
        # base_results = {}

        handles = []

        for layer_idx, vec_dict in ablation_vectors.items():
            ablation_hook = intervention_hooks.get_ablation_hook(
                "projection_ablations",
                vec_dict["diff_acts_D"],
                None,
                None,
                vec_dict["mu"],
            )

            submodule = model_utils.get_submodule(model, layer_idx)
            handle = submodule.register_forward_hook(ablation_hook)
            handles.append(handle)

        try:
            intervention_results = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=tasks,
                num_fewshot=0,
                task_manager=task_manager,
            )
        finally:
            for handle in handles:
                handle.remove()

        filename = f"{output_folder}/{model_name.replace('/', '_')}.pkl"
        results = {
            "base_results": base_results,
            "intervention_results": intervention_results,
            "eval_config": cfg.model_dump(),
        }

        with open(filename, "wb") as f:
            pickle.dump(results, f)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        del model
        del lm_obj

        torch.cuda.empty_cache()
        gc.collect()
