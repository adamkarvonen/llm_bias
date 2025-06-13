from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import contextlib

import mypkg.whitebox_infra.dictionaries.base_sae as base_sae
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.dictionaries.jumprelu_sae as jumprelu_sae


# Model configuration mapping
MODEL_CONFIGS = {
    "google/gemma-2-2b-it": {
        "batch_size": 1,
    },
    "google/gemma-2-9b-it": {
        "batch_size": 4,
    },
    "google/gemma-2-27b-it": {
        "batch_size": 1,
    },
    "google/gemma-3-12b-it": {
        "batch_size": 4,
    },
    "google/gemma-3-27b-it": {
        "batch_size": 1,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "batch_size": 8,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "batch_size": 4,
    },
}

MODEL_SAE_CONFIGS = {
    "google/gemma-2-2b-it": {
        "total_layers": 26,  # Adding for reference
        "layer_mappings": {
            25: {"layer": 5},
            50: {"layer": 12},
            75: {"layer": 19},
        },
        "trainer_id": 65,
    },
    "google/gemma-2-9b-it": {
        "total_layers": 40,  # Adding for reference
        "layer_mappings": {
            25: {"layer": 9},
            50: {"layer": 20},
            75: {"layer": 31},
        },
        "trainer_id": 131,
    },
    "google/gemma-2-27b-it": {
        "total_layers": 44,  # Adding for reference
        "layer_mappings": {
            25: {"layer": 10},
            50: {"layer": 22},
            75: {"layer": 34},
        },
        "trainer_id": 131,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "total_layers": 36,
        "layer_mappings": {
            25: {"layer": 9},
            50: {"layer": 18},
            75: {"layer": 27},
        },
        "trainer_id": 2,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "total_layers": 40,
        "layer_mappings": {
            25: {"layer": 10},
            50: {"layer": 20},
            75: {"layer": 30},
        },
        "trainer_id": 2,
    },
}

# Gemma-specific width information
GEMMA_WIDTH_INFO = {
    "google/gemma-2-2b-it": {
        16: {
            25: "width_16k/average_l0_143",
            50: "width_16k/average_l0_82",
            75: "width_16k/average_l0_137",
        },
        65: {
            25: "width_65k/average_l0_105",
            50: "width_65k/average_l0_141",
            75: "width_65k/average_l0_115",
        },
    },
    "google/gemma-2-9b-it": {
        16: {
            25: "width_16k/average_l0_88",
            50: "width_16k/average_l0_91",
            75: "width_16k/average_l0_142",
        },
        131: {
            25: "width_131k/average_l0_121",
            50: "width_131k/average_l0_81",
            75: "width_131k/average_l0_109",
        },
    },
    "google/gemma-2-27b-it": {
        131: {
            25: "width_131k/average_l0_106",
            50: "width_131k/average_l0_82",
            75: "width_131k/average_l0_155",
        },
    },
}


def get_layer_info(model_name: str, layer_percent: int) -> int:
    """Get layer number for a given model and percentage."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported")

    layer_mappings = MODEL_CONFIGS[model_name]["layer_mappings"]
    if layer_percent not in layer_mappings:
        raise ValueError(f"Layer percent must be 25, 50, or 75, got {layer_percent}")

    mapping = layer_mappings[layer_percent]
    return mapping["layer"]


def load_gemma_2_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
    trainer_id: int,
):
    """NOTE: trainer_id means the width"""
    assert trainer_id in [16, 65, 131]
    if model_name == "google/gemma-2-9b-it":
        repo_id = "google/gemma-scope-9b-it-res"
    elif model_name == "google/gemma-2-27b-it":
        repo_id = "google/gemma-scope-27b-pt-res"
    elif model_name == "google/gemma-2-2b-it":
        repo_id = "google/gemma-scope-2b-pt-res"
    else:
        raise ValueError(f"Model {model_name} not supported")

    layer = get_layer_info(model_name, layer_percent)

    # Retrieve width_info from the dedicated dictionary
    if (
        model_name not in GEMMA_WIDTH_INFO
        or trainer_id not in GEMMA_WIDTH_INFO[model_name]
        or layer_percent not in GEMMA_WIDTH_INFO[model_name][trainer_id]
    ):
        raise ValueError(
            f"Width info not available for {model_name} at {layer_percent}% layer with trainer {trainer_id}."
        )
    width_info = GEMMA_WIDTH_INFO[model_name][trainer_id][layer_percent]

    filename = f"layer_{layer}/{width_info}/params.npz"

    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=repo_id,
        filename=filename,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
        local_dir="downloaded_saes",
    )
    return sae


def load_mistral_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
    trainer_id: int = 1,
):
    layer = get_layer_info(model_name, layer_percent)
    if model_name == "mistralai/Ministral-8B-Instruct-2410":
        sae_repo = "adamkarvonen/ministral_saes"
        sae_path = f"mistralai_Ministral-8B-Instruct-2410_batch_top_k/resid_post_layer_{layer}/trainer_{trainer_id}/ae.pt"
    elif model_name == "mistralai/Mistral-Small-24B-Instruct-2501":
        sae_repo = "adamkarvonen/mistral_24b_saes"
        sae_path = f"mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k/resid_post_layer_{layer}/trainer_{trainer_id}/ae.pt"
    else:
        raise ValueError(f"Model {model_name} not supported")

    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=device,
        dtype=dtype,
        layer=layer,
        local_dir="downloaded_saes",
    )
    return sae


def load_model_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
    trainer_id: int,
) -> base_sae.BaseSAE:
    if layer_percent not in (25, 50, 75):
        raise ValueError(f"Layer percent must be 25, 50, or 75, got {layer_percent}")

    if (
        model_name == "google/gemma-2-9b-it"
        or model_name == "google/gemma-2-27b-it"
        or model_name == "google/gemma-2-2b-it"
    ):
        return load_gemma_2_sae(model_name, device, dtype, layer_percent, trainer_id)
    elif (
        model_name == "mistralai/Ministral-8B-Instruct-2410"
        or model_name == "mistralai/Mistral-Small-24B-Instruct-2501"
    ):
        return load_mistral_sae(model_name, device, dtype, layer_percent, trainer_id)
    else:
        raise ValueError(f"Model {model_name} not supported")


def add_chat_template(prompts: list[str], model_name: str, task_prompt: str | None = None) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatted_prompts = []
    for prompt in prompts:
        messages = []

        if "gemma-2" not in model_name and task_prompt is not None:
            # gemma-2 doesn't support system prompts
            messages.append({"role": "system", "content": task_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts


def collect_token_ids(tokenizer: AutoTokenizer, candidates: list[str]) -> list[int]:
    """
    Collect the set of token IDs that appear when tokenizing
    any candidate string in `candidates`.
    """
    token_ids = set()
    for c in candidates:
        # add_special_tokens=False so we only get raw subwords
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) > 1:
            ids = ids[:1]
        assert len(ids) == 1, f"Expected 1 token ID, got {len(ids)} for {c}"
        token_ids.update(ids)

    return list(token_ids)


def get_yes_no_ids(
    tokenizer: AutoTokenizer,
    device: torch.device,
    yes_candidates: Optional[list[str]] = None,
    no_candidates: Optional[list[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if yes_candidates is None:
        yes_candidates = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
    if no_candidates is None:
        no_candidates = ["no", " no", "No", " No", "NO", " NO"]

    yes_ids = collect_token_ids(tokenizer, yes_candidates)
    no_ids = collect_token_ids(tokenizer, no_candidates)

    yes_ids_t = torch.tensor(yes_ids).to(device)  # for indexing
    no_ids_t = torch.tensor(no_ids).to(device)

    return yes_ids_t, no_ids_t


def get_yes_no_probs(
    tokenizer: AutoTokenizer,
    logits_BV: torch.Tensor,
    yes_candidates: Optional[list[str]] = None,
    no_candidates: Optional[list[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = logits_BV.device
    yes_ids_t, no_ids_t = get_yes_no_ids(
        tokenizer,
        yes_candidates=yes_candidates,
        no_candidates=no_candidates,
        device=device,
    )

    probs_BV = torch.nn.functional.softmax(logits_BV, dim=-1)
    yes_probs = probs_BV.index_select(1, yes_ids_t)
    no_probs = probs_BV.index_select(1, no_ids_t)

    yes_probs_B = yes_probs.sum(dim=1)
    no_probs_B = no_probs.sum(dim=1)

    return yes_probs_B, no_probs_B


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""

    if model.config.architectures[0] == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer]
    elif (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
        or model.config.architectures[0] == "MistralForCausalLM"
    ):
        return model.model.layers[layer]
    elif model.config.architectures[0] == "Gemma3ForConditionalGeneration":
        return model.language_model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model.config._name_or_path}")


def get_num_layers(model: AutoModelForCausalLM):
    """Gets the number of layers"""

    if model.config.architectures[0] == "GPTNeoXForCausalLM":
        return len(model.gpt_neox.layers)
    elif (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
        or model.config.architectures[0] == "MistralForCausalLM"
    ):
        return len(model.model.layers)
    elif model.config.architectures[0] == "Gemma3ForConditionalGeneration":
        return len(model.language_model.layers)
    else:
        raise ValueError(f"Please add submodule for model {model.config._name_or_path}")


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    if activations_BLD is None:
        # This should ideally not happen if the hook worked and EarlyStopException was raised,
        # but handle it just in case.
        raise RuntimeError(
            "Failed to collect activations. The hook might not have run correctly."
        )

    return activations_BLD


@torch.no_grad()
def get_activations_per_layer(
    model: AutoModelForCausalLM,
    submodules: list[torch.nn.Module],
    tokens_batch: dict[str, torch.Tensor],
    get_final_token_only: bool = False,
) -> tuple[dict[torch.nn.Module, torch.Tensor], torch.Tensor]:
    activations_BLD = {}

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        assert isinstance(outputs, tuple)
        if get_final_token_only:
            activations_BLD[module] = outputs[0][:, -1:, :]
        else:
            activations_BLD[module] = outputs[0]

    all_handles = []

    try:
        for submodule in submodules:
            handle = submodule.register_forward_hook(gather_target_act_hook)
            all_handles.append(handle)

        logits_BLV = model(**tokens_batch).logits

        if get_final_token_only:
            logits_BLV = logits_BLV[:, -1:, :]

    except Exception as e:
        print(e)
        raise e
    finally:
        for handle in all_handles:
            handle.remove()

    return activations_BLD, logits_BLV


def truncate_model(model: AutoModelForCausalLM, layer: int):
    """From tilde-research/activault
    https://github.com/tilde-research/activault/blob/db6d1e4e36c2d3eb4fdce79e72be94f387eccee1/pipeline/setup.py#L74
    This provides significant memory savings by deleting all layers that aren't needed for the given layer.
    You should probably test this before using it"""
    import gc

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Model parameters before truncation: {total_params_before:,}")

    if (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
    ):
        removed_layers = model.model.layers[layer + 1 :]

        model.model.layers = model.model.layers[: layer + 1]

        del removed_layers
        del model.lm_head

        model.lm_head = torch.nn.Identity()

    elif model.config.architectures[0] == "GPTNeoXForCausalLM":
        removed_layers = model.gpt_neox.layers[layer + 1 :]

        model.gpt_neox.layers = model.gpt_neox.layers[: layer + 1]

        del removed_layers
        del model.embed_out

        model.embed_out = torch.nn.Identity()

    else:
        raise ValueError(f"Please add truncation for model {model.name_or_path}")

    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"Model parameters after truncation: {total_params_after:,}")

    gc.collect()
    torch.cuda.empty_cache()

    return model
