from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import asyncio
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Callable, Literal
import torch
from torch import Tensor
import einops
import torch.nn.functional as F
from dataclasses import asdict
import wandb
import os
from contextlib import contextmanager, nullcontext
from torch import autocast
import math

import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.data_utils as data_utils
import mypkg.whitebox_infra.intervention_hooks as intervention_hooks
import mypkg.whitebox_infra.interp_utils as interp_utils


def run_final_block(
    acts_BLD: torch.Tensor,
    model: AutoModelForCausalLM,
) -> torch.Tensor:
    """
    Feeds `acts_BLD` through the last Gemma2DecoderLayer (attention+MLP)
    and returns the updated hidden states.
    """
    B, L, _ = acts_BLD.shape
    device = acts_BLD.device

    pos_ids_BL = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    position_embeddings = model.model.rotary_emb(acts_BLD, pos_ids_BL)

    hidden_BLD = model.model.layers[-1](
        hidden_states=acts_BLD,
        position_embeddings=position_embeddings,
    )
    return hidden_BLD[0]


def apply_logit_lens(
    acts_BLD: torch.Tensor,
    model: AutoModelForCausalLM,
    final_token_only: bool = True,
    add_final_layer: bool = True,
    tuned_lens: torch.nn.Linear | None = None,
) -> torch.Tensor:
    if final_token_only:
        acts_BLD = acts_BLD[:, -1:, :]

    if add_final_layer:
        assert tuned_lens is None, "Cannot add final layer if tuned_lens is provided"
        acts_BLD += run_final_block(acts_BLD, model)

    if tuned_lens is not None:
        orig_dtype = acts_BLD.dtype
        assert add_final_layer is False, "Cannot add final layer if tuned_lens is provided"
        with autocast("cuda", dtype=torch.bfloat16):
            acts_BLD = tuned_lens(acts_BLD)  # weight stays fp32; activations & GEMM run bf16
        acts_BLD = acts_BLD.to(orig_dtype)


    logits_BLV = model.lm_head(model.model.norm(acts_BLD))
    return logits_BLV


def kl_between_logits(
    logits_p_BLV: torch.Tensor,  # “teacher” / reference
    logits_q_BLV: torch.Tensor,  # “student” / lens
    reduction: Literal["batchmean", "mean", "none"] = "none",
) -> torch.Tensor:
    """
    KL‖(p || q) for two logit tensors of shape [B, L, V].

    * `p` is obtained with softmax on `logits_p_BLV`
    * `q` is obtained with softmax on `logits_q_BLV`
    * Uses `torch.kl_div(log_q, p)` so we pass log-probabilities for q
    """

    B, L, V = logits_p_BLV.shape

    # Convert logits → probabilities / log-probabilities
    p_BLV = torch.softmax(logits_p_BLV.float(), dim=-1)
    log_q_BLV = torch.log_softmax(logits_q_BLV.float(), dim=-1)

    # KL divergence per token; `reduction` handles batching
    kl = F.kl_div(log_q_BLV, p_BLV, reduction=reduction).sum(dim=-1)

    return kl  # scalar if reduction ≠ "none", else [B, L]


def get_yes_no_statistics(
    logits_BLV: torch.Tensor,
    yes_ids_t: torch.Tensor,
    no_ids_t: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 10,
):
    logits_BV = logits_BLV[:, -1, :]
    probs_BV = torch.softmax(logits_BV, dim=-1)
    yes_logits_BV = logits_BV.index_select(1, yes_ids_t)
    no_logits_BV = logits_BV.index_select(1, no_ids_t)

    yes_probs_BV = probs_BV.index_select(1, yes_ids_t)
    no_probs_BV = probs_BV.index_select(1, no_ids_t)

    yes_logits_B = yes_logits_BV.sum(dim=1)
    no_logits_B = no_logits_BV.sum(dim=1)

    yes_probs_B = yes_probs_BV.sum(dim=1)
    no_probs_B = no_probs_BV.sum(dim=1)

    results = []

    for i in range(logits_BV.shape[0]):
        top_k_vals, top_k_idxs = torch.topk(probs_BV[i], k=k, dim=-1)
        top_tokens = tokenizer.convert_ids_to_tokens(top_k_idxs.tolist())

        results.append(
            {
                "yes_logits": yes_logits_B[i].item(),
                "no_logits": no_logits_B[i].item(),
                "yes_probs": yes_probs_B[i].item(),
                "no_probs": no_probs_B[i].item(),
                "top_k_tokens": top_tokens,
                "top_k_vals": top_k_vals.tolist(),
                "top_k_idxs": top_k_idxs.tolist(),
            }
        )

    return results


@torch.inference_mode()
def run_logit_lens(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    model_name: str,
    add_final_layer: bool = False,
    batch_size: int = 64,
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
    use_tuned_lenses: bool = False,
) -> dict:
    assert padding_side in ["left", "right"]

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            # attn_implementation="flash_attention_2",  # Currently having install issues with flash attention 2
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_prompts = [p.prompt for p in prompt_dicts]

    formatted_prompts = model_utils.add_chat_template(original_prompts, model_name)
    dataloader = data_utils.create_simple_dataloader(
        formatted_prompts,
        [0] * len(formatted_prompts),
        prompt_dicts,
        model_name,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    num_layers = len(list(model.model.layers))
    print(f"Number of layers: {num_layers} for {model_name}")

    all_submodules = [model_utils.get_submodule(model, i) for i in range(num_layers)]

    get_final_token_only = True

    if use_tuned_lenses:
        assert add_final_layer is False, "Cannot add final layer if tuned_lenses are used"
        tuned_lenses = load_lenses(model, model_name, device, dtype=dtype)
    else:
        tuned_lenses = [None] * num_layers

    yes_ids_t, no_ids_t = model_utils.get_yes_no_ids(tokenizer, device)

    per_layer_results = {}

    for i, layer in enumerate(all_submodules):
        per_layer_results[i] = []
    per_layer_results["output"] = []

    for batch in tqdm(dataloader, desc="Processing prompts"):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
            model,
            all_submodules,
            model_inputs,
            get_final_token_only=get_final_token_only,
        )

        output_stats = get_yes_no_statistics(logits_BLV, yes_ids_t, no_ids_t, tokenizer)
        for i in range(len(output_stats)):
            output_stats[i]["label"] = labels[i].item()
            output_stats[i]["kl"] = 0.0
            output_stats[i]["resume_prompt_result"] = asdict(
                resume_prompt_results_batch[i]
            )

        per_layer_results["output"].extend(output_stats)

        for i, layer in enumerate(all_submodules):
            logit_lens_BLV = apply_logit_lens(
                activations_BLD[layer],
                model,
                final_token_only=get_final_token_only,
                add_final_layer=add_final_layer,
                tuned_lens=tuned_lenses[i],
            )
            kl_BL = kl_between_logits(logits_BLV, logit_lens_BLV)
            kl_B = kl_BL.mean(dim=1)
            layer_stats = get_yes_no_statistics(
                logit_lens_BLV, yes_ids_t, no_ids_t, tokenizer
            )
            for j in range(len(layer_stats)):
                layer_stats[j]["kl"] = kl_B[j].item()
            per_layer_results[i].extend(layer_stats)

    return per_layer_results

@torch.inference_mode()
def run_logit_lens_with_intervention(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    model_name: str,
    ablation_features: torch.Tensor,
    ablation_type: str,
    scale: float,
    add_final_layer: bool = False,
    batch_size: int = 64,
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
    use_tuned_lenses: bool = False,
) -> dict:
    assert padding_side in ["left", "right"]

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            # attn_implementation="flash_attention_2",  # Currently having install issues with flash attention 2
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_prompts = [p.prompt for p in prompt_dicts]

    formatted_prompts = model_utils.add_chat_template(original_prompts, model_name)
    dataloader = data_utils.create_simple_dataloader(
        formatted_prompts,
        [0] * len(formatted_prompts),
        prompt_dicts,
        model_name,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    num_layers = len(list(model.model.layers))
    print(f"Number of layers: {num_layers} for {model_name}")

    all_submodules = [model_utils.get_submodule(model, i) for i in range(num_layers)]

    get_final_token_only = True

    if use_tuned_lenses:
        assert add_final_layer is False, "Cannot add final layer if tuned_lenses are used"
        tuned_lenses = load_lenses(model, model_name, device, dtype=dtype)
    else:
        tuned_lenses = [None] * num_layers

    yes_ids_t, no_ids_t = model_utils.get_yes_no_ids(tokenizer, device)

    per_layer_results = {}

    for i, layer in enumerate(all_submodules):
        per_layer_results[i] = []
    per_layer_results["output"] = []

    trainer_id = model_utils.MODEL_CONFIGS[model_name]["trainer_id"]
    sae = model_utils.load_model_sae(
        model_name, device, dtype, 25, trainer_id=trainer_id
    )
    scales = [scale] * len(ablation_features)
    encoder_vectors, decoder_vectors, encoder_biases = (
        intervention_hooks.get_sae_vectors(ablation_features, sae)
    )
    sae = sae.to("cpu")
    hook_layer = sae.hook_layer
    del sae
    torch.cuda.empty_cache()

    ablation_hook = intervention_hooks.get_ablation_hook(
        ablation_type,
        encoder_vectors,
        decoder_vectors,
        scales,
        encoder_biases,
    )

    submodule = model_utils.get_submodule(model, hook_layer)
    intervention_handle = submodule.register_forward_hook(ablation_hook)

    for batch in tqdm(dataloader, desc="Processing prompts"):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
            model,
            all_submodules,
            model_inputs,
            get_final_token_only=get_final_token_only,
        )

        output_stats = get_yes_no_statistics(logits_BLV, yes_ids_t, no_ids_t, tokenizer)
        for i in range(len(output_stats)):
            output_stats[i]["label"] = labels[i].item()
            output_stats[i]["kl"] = 0.0
            output_stats[i]["resume_prompt_result"] = asdict(
                resume_prompt_results_batch[i]
            )

        per_layer_results["output"].extend(output_stats)

        for i, layer in enumerate(all_submodules):
            logit_lens_BLV = apply_logit_lens(
                activations_BLD[layer],
                model,
                final_token_only=get_final_token_only,
                add_final_layer=add_final_layer,
                tuned_lens=tuned_lenses[i],
            )
            kl_BL = kl_between_logits(logits_BLV, logit_lens_BLV)
            kl_B = kl_BL.mean(dim=1)
            layer_stats = get_yes_no_statistics(
                logit_lens_BLV, yes_ids_t, no_ids_t, tokenizer
            )
            for j in range(len(layer_stats)):
                layer_stats[j]["kl"] = kl_B[j].item()
            per_layer_results[i].extend(layer_stats)

    intervention_handle.remove()

    return per_layer_results

@torch.no_grad()
def logit_lens_on_batched_tokens(
    model: AutoModelForCausalLM,
    submodules: list[torch.nn.Module],
    batched_tokens: dict[str, torch.Tensor],
    get_final_token_only: bool = False,
    add_final_layer: bool = False,
    all_tuned_lenses: torch.nn.ModuleList | None = None,
) -> dict[int, float]:
    mean_kl_per_layer = {}

    for i, layer in enumerate(submodules):
        mean_kl_per_layer[i] = 0.0

    for batch in tqdm(batched_tokens):
        activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
            model, submodules, batch, get_final_token_only=get_final_token_only
        )

        for i, layer in enumerate(submodules):
            if all_tuned_lenses is not None:
                tuned_lens = all_tuned_lenses[i]
            else:
                tuned_lens = None

            logit_lens_BLV = apply_logit_lens(
                activations_BLD[layer], model, final_token_only=get_final_token_only, add_final_layer=add_final_layer, tuned_lens=tuned_lens
            )
            kl = kl_between_logits(logits_BLV, logit_lens_BLV)
            mean_kl_per_layer[i] += kl.mean().item()

    for i, layer in enumerate(submodules):
        mean_kl_per_layer[i] /= len(batched_tokens)

    return mean_kl_per_layer

def build_tuned_lenses(num_layers: int, d_model: int, device, dtype):
    """
    One learnable linear map per layer, all identity-initialised.
    """
    lenses = torch.nn.ModuleList([
        torch.nn.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)
        for _ in range(num_layers)
    ])
    for lens in lenses:
        torch.nn.init.eye_(lens.weight)      # start as plain logit lens
        torch.nn.init.zeros_(lens.bias)
    return lenses

def train_tuned_lens(
    model: AutoModelForCausalLM,
    train_batched_tokens: list[dict[str, torch.Tensor]],
    num_epochs: int = 1,
    lr: float = 5e-4,
    grad_accum_steps: int = 1,
    get_final_token_only: bool = False,
    lens_dtype: torch.dtype = torch.float32,
    log_wandb: bool = False,
    wandb_project: str = "tuned-lens",
    wandb_run_name: str | None = None,
):
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    num_layers = len(list(model.model.layers))

    submodules = [model_utils.get_submodule(model, i) for i in range(num_layers)]
    lenses = build_tuned_lenses(num_layers, hidden_size, device, dtype=lens_dtype)

    optimizer = torch.optim.AdamW(lenses.parameters(), lr=lr, weight_decay=0.01)

    total_optimizer_steps = math.ceil(
        len(train_batched_tokens) * num_epochs / grad_accum_steps
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=total_optimizer_steps
    )

    # ----- WANDB run init (only if logging is requested) -----
    if log_wandb:
        if wandb.run is None:        # avoid creating nested runs
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=dict(
                    lr=lr,
                    num_epochs=num_epochs,
                    model_name=model.config._name_or_path,
                    hidden_size=hidden_size,
                    layers=num_layers,
                    final_token_only=get_final_token_only,
                ),
            )

    # ---- freeze base model ----
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    step          = 0          # optimiser steps (i.e. after accumulation)
    micro_step    = 0          # raw batch counter
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for batch in tqdm(train_batched_tokens, desc=f"epoch {epoch+1}/{num_epochs}"):
            micro_step += 1
            # --- base forward (no grad) ---
            with torch.no_grad():
                acts_dict, teacher_logits = model_utils.get_activations_per_layer(
                    model,
                    submodules,
                    batch,
                    get_final_token_only=get_final_token_only,
                )
                teacher_logits = teacher_logits.detach().float()

            # --- tuned-lens forward & per-layer KL ---
            layer_losses = []
            logging_loss = 0.0
            for i, layer in enumerate(submodules):
                acts = acts_dict[layer].detach()
                logits_q = apply_logit_lens(acts, model, final_token_only=get_final_token_only, add_final_layer=False, tuned_lens=lenses[i])

                layer_kl = kl_between_logits(
                    teacher_logits,
                    logits_q
                ).mean()
                layer_losses.append(layer_kl)

                loss = layer_kl
                loss = loss / num_layers
                loss = loss / grad_accum_steps                # scale for accumulation
                loss.backward()
                logging_loss += loss.item()

            # ---------- step optimiser every N micro-batches ----------
            if micro_step % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            # --- logging ---
            # if step % 10 == 0:
            #     lr_now = scheduler.get_last_lr()[0]
            #     print(f"step {step:>6}/{total_steps}  loss {loss.item():.4f}  lr {lr_now:.2e}")

            if log_wandb and micro_step % 10 == 0:
                log_dict = {
                    "loss/mean": logging_loss * grad_accum_steps,
                    "lr": scheduler.get_last_lr()[0],
                }
                for i, l_val in enumerate(layer_losses):
                    log_dict[f"loss/layer_{i}"] = l_val.item()
                wandb.log(log_dict, step=micro_step)

    return lenses


def save_lenses(lenses: torch.nn.ModuleList, model_name: str, out_dir: str = "tuned_lenses"):
    """
    Saves all layers’ tuned-lens weights to a single .pt file.
    """
    filename = f"{model_name.replace('/', '_')}_tuned_lens.pt"
    output_path = os.path.join(out_dir, filename)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(lenses.state_dict(), output_path)

def load_lenses(model: AutoModelForCausalLM,
                model_name: str,
                device: torch.device,
                dtype=torch.float32,
                ckpt_dir: str = "tuned_lenses") -> torch.nn.ModuleList:
    """
    Rebuilds an empty lens stack and loads weights from disk.
    """
    num_layers = len(list(model.model.layers))
    hidden_size = model.config.hidden_size
    lenses = build_tuned_lenses(num_layers, hidden_size, device, dtype)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name.replace('/', '_')}_tuned_lens.pt")
    lenses.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    lenses.eval()
    return lenses

@contextmanager
def wandb_session(project: str,
                name: str | None = None,
                **init_kwargs):
    """
    Starts a run if none exists; always `wandb.finish()` on exit.
    Safe to nest (won’t close an outer run you didn’t create).
    """
    import wandb
    created_run = wandb.run is None          # detect outer scope
    if created_run:
        run = wandb.init(project=project, name=name, **init_kwargs)
    else:
        run = wandb.run                      # reuse existing one
    try:
        yield run                            # code inside the “with” block
    finally:
        if created_run:
            wandb.finish()                   # ensures clean shutdown

if __name__ == "__main__":
    # model_names = ["google/gemma-2-27b-it"]
    # model_names = ["mistralai/Mistral-Small-24B-Instruct-2501"]
    model_names = ["google/gemma-2-2b-it", "google/gemma-2-9b-it", "mistralai/Ministral-8B-Instruct-2410"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    lens_dtype = torch.float32

    use_wandb = True

    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = model_utils.MODEL_CONFIGS[model_name]["batch_size"] * 2
        context_length = 512

        num_layers = len(list(model.model.layers))
        submodules = [model_utils.get_submodule(model, i) for i in range(num_layers)]

        dataset_name = "togethercomputer/RedPajama-Data-V2"
        num_tokens = 1_000_000
        # num_tokens = 200_000

        batched_tokens = interp_utils.get_batched_tokens(
            tokenizer=tokenizer,
            model_name=model_name,
            dataset_name=dataset_name,
            num_tokens=num_tokens,
            batch_size=batch_size,
            device=device,
            context_length=context_length,
            force_rebuild_tokens=False,
        )

        val_batched_tokens = batched_tokens[:10]
        train_batched_tokens = batched_tokens[10:]

        val_loss = logit_lens_on_batched_tokens(
            model,
            submodules,
            val_batched_tokens,
            get_final_token_only=False,
        )

        print(f"Initial val loss: {val_loss}")

        torch.set_grad_enabled(True)

        if use_wandb:
            logging_context = wandb_session(project="tuned_lenses",
                        name=f"{model_name}_lr5e-4_epoch1")
        else:
            logging_context = nullcontext()

        with logging_context:
            lenses = train_tuned_lens(
                model,
                train_batched_tokens,
                num_epochs=1,
                lr=5e-4,
                get_final_token_only=False,
                lens_dtype=lens_dtype,
                log_wandb=use_wandb                    # training loop will just log
            )

        save_lenses(lenses, model_name)

        # inference in bfloat16
        lenses = load_lenses(model, model_name, device, dtype=dtype)

        final_val_loss = logit_lens_on_batched_tokens(
            model,
            submodules,
            val_batched_tokens,
            get_final_token_only=False,
            add_final_layer=False,
            all_tuned_lenses=lenses,
        )
        print(f"Final val loss: {final_val_loss}")