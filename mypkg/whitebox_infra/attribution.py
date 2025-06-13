import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, Callable, Any
import torch
import einops
from dataclasses import dataclass, field, fields

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from mypkg.whitebox_infra.dictionaries import topk_sae, base_sae
import mypkg.whitebox_infra.model_utils as model_utils


def get_results_per_label(
    vals_BLF: torch.Tensor, labels_B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_mask_B = labels_B == 1
    neg_mask_B = labels_B == 0

    pos_vals_KLF = vals_BLF[pos_mask_B]
    neg_vals_KLF = vals_BLF[neg_mask_B]
    feat_dim = vals_BLF.size(-1)
    device = vals_BLF.device

    if pos_vals_KLF.numel():
        pos_vals_KF = einops.reduce(
            pos_vals_KLF.to(torch.float32), "K L F -> K F", "mean"
        )
        mean_pos_F = einops.reduce(pos_vals_KF, "K F -> F", "sum")
    else:
        mean_pos_F = torch.zeros(feat_dim, dtype=torch.float32, device=device)

    if neg_vals_KLF.numel():
        neg_vals_KF = einops.reduce(
            neg_vals_KLF.to(torch.float32), "K L F -> K F", "mean"
        )
        mean_neg_F = einops.reduce(neg_vals_KF, "K F -> F", "sum")
    else:
        mean_neg_F = torch.zeros(feat_dim, dtype=torch.float32, device=device)

    return mean_pos_F, mean_neg_F


@dataclass
class AttributionData:
    D: int
    F: int
    device: torch.device | str = "cpu"

    # tensors initialised in __post_init__
    pos_acts_D: torch.Tensor = field(init=False)
    neg_acts_D: torch.Tensor = field(init=False)
    pos_sae_acts_F: torch.Tensor = field(init=False)
    neg_sae_acts_F: torch.Tensor = field(init=False)
    pos_sae_grads_F: torch.Tensor = field(init=False)
    neg_sae_grads_F: torch.Tensor = field(init=False)
    pos_effects_F: torch.Tensor = field(init=False)
    neg_effects_F: torch.Tensor = field(init=False)

    # running scalars
    pos_error_effect: float = 0.0
    neg_error_effect: float = 0.0
    pos_yes_probs: float = 0.0
    pos_no_probs: float = 0.0
    neg_yes_probs: float = 0.0
    neg_no_probs: float = 0.0
    pos_count: int = 0
    neg_count: int = 0

    # token logs
    pos_predicted_tokens: list[str] = field(default_factory=list)
    neg_predicted_tokens: list[str] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────
    def __post_init__(self):
        zD = torch.zeros(self.D, device=self.device)
        zF = torch.zeros(self.F, device=self.device)

        self.pos_acts_D = zD.clone()
        self.neg_acts_D = zD.clone()
        self.pos_sae_acts_F = zF.clone()
        self.neg_sae_acts_F = zF.clone()
        self.pos_sae_grads_F = zF.clone()
        self.neg_sae_grads_F = zF.clone()
        self.pos_effects_F = zF.clone()
        self.neg_effects_F = zF.clone()

    def to(self, device: torch.device | str):
        """Move all tensors to `device` (similar to nn.Module.to)."""
        for f in fields(self):
            v: Any = getattr(self, f.name)
            if torch.is_tensor(v):
                setattr(self, f.name, v.to(device))
        self.device = device
        return self

    @classmethod
    def from_dict(
        cls,
        src: dict[str, Any],
        *,
        device: str | torch.device | None = None,
    ) -> "AttributionData":
        """
        Rebuild an AttributionData from a dict produced by `asdict()`.
        If `device` is given it overrides the value stored in the dict.
        """
        D = src["D"]
        F = src["F"]
        dev = device if device is not None else src.get("device", "cpu")

        obj = cls(D=D, F=F, device=dev)  # runs __post_init__

        for f in fields(obj):
            name = f.name
            if name in {"D", "F", "device"}:  # already set
                continue
            if name not in src:  # missing → keep default
                continue
            val = src[name]
            if torch.is_tensor(val):
                val = val.to(dev)
            obj.__setattr__(name, val)
        return obj

    @torch.no_grad()
    def update_from_batch(
        self,
        batch_results: dict[str, torch.Tensor | list[str]],
    ):
        """
        Accumulate attribution stats from one minibatch already on `self.device`.
        All variable names keep the *_suffix convention for shapes.
        """
        # ===== unpack =====
        labels_B = batch_results["labels"]
        predicted_tokens = batch_results["predicted_tokens"]  # list[str]

        sae_acts_BLF = batch_results["encoded_acts_BLF"]
        effects_BLF = batch_results["effects_BLF"]
        grad_x_dot_decoder_BLF = batch_results["grad_x_dot_decoder_BLF"]
        acts_BLD = batch_results["acts_BLD"]
        error_effects_BLD = batch_results["error_effects_BLD"]

        yes_log_probs_B = batch_results["yes_log_probs_B"]
        no_log_probs_B = batch_results["no_log_probs_B"]

        pos_mask_B = labels_B == 1
        neg_mask_B = labels_B == 0

        # ===== F-dim reductions =====
        pos_sae_acts_F, neg_sae_acts_F = get_results_per_label(sae_acts_BLF, labels_B)
        pos_grads_F, neg_grads_F = get_results_per_label(
            grad_x_dot_decoder_BLF, labels_B
        )
        pos_effects_F, neg_effects_F = get_results_per_label(effects_BLF, labels_B)

        # ===== D-dim reductions =====
        pos_acts_D, neg_acts_D = get_results_per_label(acts_BLD, labels_B)

        # ===== error-effect scalar =====
        if pos_mask_B.any():
            pos_error_effect = (
                error_effects_BLD[pos_mask_B].mean(dim=(1, 2)).sum().item()
            )
            pos_yes_probs = yes_log_probs_B[pos_mask_B].sum().item()
            pos_no_probs = no_log_probs_B[pos_mask_B].sum().item()
        else:
            pos_error_effect = 0.0
            pos_yes_probs = 0.0
            pos_no_probs = 0.0

        if neg_mask_B.any():
            neg_error_effect = (
                error_effects_BLD[neg_mask_B].mean(dim=(1, 2)).sum().item()
            )
            neg_yes_probs = yes_log_probs_B[neg_mask_B].sum().item()
            neg_no_probs = no_log_probs_B[neg_mask_B].sum().item()
        else:
            neg_error_effect = 0.0
            neg_yes_probs = 0.0
            neg_no_probs = 0.0

        # ===== accumulate =====
        self.pos_acts_D += pos_acts_D
        self.neg_acts_D += neg_acts_D
        self.pos_sae_acts_F += pos_sae_acts_F
        self.neg_sae_acts_F += neg_sae_acts_F
        self.pos_sae_grads_F += pos_grads_F
        self.neg_sae_grads_F += neg_grads_F
        self.pos_effects_F += pos_effects_F
        self.neg_effects_F += neg_effects_F
        self.pos_error_effect += pos_error_effect
        self.neg_error_effect += neg_error_effect
        self.pos_yes_probs += pos_yes_probs
        self.pos_no_probs += pos_no_probs
        self.neg_yes_probs += neg_yes_probs
        self.neg_no_probs += neg_no_probs
        self.pos_count += int(pos_mask_B.sum())
        self.neg_count += int(neg_mask_B.sum())

        for label_i, tok in zip(labels_B.tolist(), predicted_tokens):
            if label_i == 1:
                self.pos_predicted_tokens.append(tok)
            else:
                self.neg_predicted_tokens.append(tok)


def make_yes_no_loss_fn(
    tokenizer,
    device: torch.device,
    yes_candidates: list[str] = ["yes", " yes", "Yes", " Yes", "YES", " YES"],
    no_candidates: list[str] = ["no", " no", "No", " No", "NO", " NO"],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Collect sets of IDs
    yes_ids_t, no_ids_t = model_utils.get_yes_no_ids(
        tokenizer,
        device,
        yes_candidates,
        no_candidates,
    )

    def yes_no_loss_fn(next_token_logits_BLV: torch.Tensor, labels_B: torch.Tensor):
        next_token_logits_BV = next_token_logits_BLV[:, -1, :]
        # Gather the logits for yes_ids and no_ids
        # .index_select(1, ...) means we select columns by ID
        yes_logits = next_token_logits_BV.index_select(1, yes_ids_t)
        no_logits = next_token_logits_BV.index_select(1, no_ids_t)

        # Sum over the sets
        yes_sum = yes_logits.sum(dim=1)  # shape [batch_size]
        no_sum = no_logits.sum(dim=1)

        # difference per example
        diff_B = yes_sum - no_sum  # shape [batch_size]

        # Apply the label-dependent logic:
        # For label=0: use diff as is
        # For label=1: flip the sign of diff
        # label_factor_B = 1 - 2 * labels_B.float()  # Convert 0->1, 1->-1
        # adjusted_diff_B = diff_B * label_factor_B

        return diff_B.mean()

    return yes_no_loss_fn


def make_max_logprob_loss_fn() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def max_logprob_loss_fn(
        next_token_logits_BLV: torch.Tensor, labels_B: torch.Tensor
    ):
        next_token_logits_BV = next_token_logits_BLV[:, -1, :]
        # convert logits to log‑probs
        log_probs = torch.nn.functional.log_softmax(next_token_logits_BV, dim=-1)
        # highest‑probability token per example
        max_log_probs_B = log_probs.max(dim=1).values  # shape [batch_size]

        # Apply the label-dependent logic:
        # For label=0: use diff as is
        # For label=1: flip the sign of diff
        # label_factor = 1 - 2 * labels_B.float()  # Convert 0->1, 1->-1
        # max_log_probs_B = max_log_probs_B * label_factor

        return -max_log_probs_B.mean()  # negate so lower = better

    return max_logprob_loss_fn


def entropy_loss_fn(
    next_token_logits_BLV: torch.Tensor, labels_B: torch.Tensor
) -> torch.Tensor:
    """
    Entropy‑based loss with optional label flipping.

    next_token_logits_BLV : [B, L, |V|]
        Batched logits over the vocabulary at every sequence position.
    labels_B             : [B]  (0 or 1)
        If label == 1 the sign of the entropy term is flipped.

    Returns
    -------
    Scalar tensor – negative mean signed entropy (lower = better
    under the same convention as the max‑log‑prob version).
    """
    # logits for the last token in each sequence  →  [B, |V|]
    next_token_logits_BV = next_token_logits_BLV[:, -1, :]

    # log‑probs and probs
    log_probs_BV = torch.nn.functional.log_softmax(next_token_logits_BV, dim=-1)
    probs_BV = log_probs_BV.exp()

    # entropy per example  (‑Σ p log p)  →  [B]
    entropy_B = -(probs_BV * log_probs_BV).sum(dim=-1)

    # label‑dependent sign flip: 0 ↦ +1, 1 ↦ −1
    # label_factor = 1.0 - 2.0 * labels_B.float().to(entropy_B.device)
    # signed_entropy_B = entropy_B * label_factor

    # negate so "lower loss" means "better" under this convention
    return -entropy_B.mean()


def activation_loss_fn(
    activations_BLD: torch.Tensor,  # [B, L, D]
    labels_B: torch.Tensor,  # [B]  (0 or 1)
    seq_slice: slice = slice(-1, None),
) -> torch.Tensor:
    # 1. pick the desired sequence positions
    h = activations_BLD[:, seq_slice, :]  # [B, L_sel, D]

    # 2. self‑energy per example
    energy_B = 0.5 * (h**2).sum(dim=(-1, -2))  # sum over D and L_sel  → [B]

    # 3. label‑dependent sign flip  (0 -> +1, 1 -> –1)
    # label_factor = 1.0 - 2.0 * labels_B.float().to(energy_B.device)
    # signed_energy_B = energy_B * label_factor

    # 4. final loss: negate so "lower = better"
    return -energy_B.mean()


def apply_logit_lens(
    acts_BLD: torch.Tensor, model: AutoModelForCausalLM
) -> torch.Tensor:
    acts_BLD = acts_BLD[:, -1:, :]
    logits_BLV = model.lm_head(model.model.norm(acts_BLD))
    return logits_BLV


def view_outputs(
    tokenizer, all_input_ids: torch.Tensor, all_answer_logits: torch.Tensor
):
    for i in range(all_input_ids.shape[0]):
        input_ids = all_input_ids[i]
        answer_logits = all_answer_logits[i : i + 1, :]

        print()
        print(tokenizer.decode(input_ids))
        # Compute log probabilities (numerically stable)
        log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        # Exponentiate log probabilities to get actual probabilities
        probs = torch.exp(log_probs)

        # Set the number of top tokens you want to inspect
        topk = 10

        for batch_idx in range(answer_logits.shape[0]):
            # Get top-k log probabilities and corresponding indices for this batch element
            top_percentages, top_indices = torch.topk(probs[batch_idx], topk)
            # Exponentiate to obtain probabilities and convert to percentages
            top_percentages *= 100

            # Convert token IDs to token strings
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

            print(f"Batch element {batch_idx}:")
            for token, token_id, percentage in zip(
                top_tokens, top_indices.tolist(), top_percentages.tolist(), strict=True
            ):
                print(
                    f"Token: {token:15s} | Token ID: {token_id:4d} | Probability: {percentage:6.2f}%"
                )
            print("-" * 50)


def analyze_prediction_rates(
    labels: list[int], predicted_tokens: list[str], verbose: bool = False
) -> dict[int, dict]:
    # Initialize counters
    stats = {
        1: {"yes": 0, "no": 0, "invalid": 0, "total": 0},
        0: {"yes": 0, "no": 0, "invalid": 0, "total": 0},
    }

    # Count occurrences
    for label, pred in zip(labels, predicted_tokens):
        stats[label]["total"] += 1
        pred = pred.lower()
        if pred == "yes":
            stats[label]["yes"] += 1
        elif pred == "no":
            stats[label]["no"] += 1
        else:
            stats[label]["invalid"] += 1

    # Calculate rates
    results = {}
    for label in stats:
        total = stats[label]["total"]
        results[label] = {
            "yes_rate": stats[label]["yes"] / total * 100,
            "no_rate": stats[label]["no"] / total * 100,
            "invalid_rate": stats[label]["invalid"] / total * 100,
            "total_samples": total,
        }

    # Print results in a readable format
    if verbose:
        for label in stats:
            print(f"\nLabel {label}:")
            print(f"Total samples: {results[label]['total_samples']}")
            print(f"Yes rate: {results[label]['yes_rate']:.1f}%")
            print(f"No rate: {results[label]['no_rate']:.1f}%")
            print(f"Invalid rate: {results[label]['invalid_rate']:.1f}%")

    return results


def compute_attributions(
    transformers_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: base_sae.BaseSAE,
    model_inputs: dict[str, torch.Tensor],
    labels_B: torch.Tensor,
    chosen_layers: list[int],
    submodules: list[torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    use_stop_gradient: bool = False,
    use_activation_loss_fn: bool = False,
    verbose: bool = False,
    ignore_bos: bool = True,
    padding_side: str = "left",
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Runs a forward/backward pass on `transformers_model` using the given `sae`
    for activation compression/decompression. Returns dictionaries keyed by
    layer

    Args:
        transformers_model: A HuggingFace model (e.g., GPT-NeoX).
        sae: An AutoEncoderTopK instance (or similar) that implements
             encode() and decode().
        input_ids: Model inputs, e.g., from tokenizer(..., return_tensors='pt').
        chosen_layers: List of layer indices to register forward hooks on.
        loss_fn: A function that takes model outputs and returns a scalar loss
                 for the backward pass. Defaults to sum of logits.

    Returns:
        A dict of dicts. For each layer index in chosen_layers, we have:
            "grad_x_dot_decoder_BLF": Tensor
            "effects_BLF": Tensor
            "encoded_acts_BLF": Tensor
            "residual_BLD": Tensor
            "x_grad_BLD": Tensor
            "error_effects_BLD": Tensor
    """

    assert len(chosen_layers) == 1, (
        "Only one layer is supported for now. Stop gradients require pass through gradients (See sparse feature circuits appendix)"
    )

    assert padding_side in ["left", "right"]

    if use_activation_loss_fn:
        loss_fn = None

    # Make sure gradients are enabled for model parameters
    for param in transformers_model.parameters():
        param.requires_grad_(False)

    for param in sae.parameters():
        param.requires_grad_(False)

    # Dict to store the re-encoded layer outputs in the forward hook
    activations = {}

    def save_activation_hook(module, input, output):
        # Get the hidden states (first element of output)
        hidden_states = output[0]
        hidden_states.requires_grad_()  # Re-enable gradient tracking on this detached tensor
        hidden_states.retain_grad()  # Ensure gradients are retained for later inspection
        activations[module] = hidden_states

        return (hidden_states,) + output[1:]

    # def save_activation_hook(module, input, output):
    #     # Get the hidden states (first element of output)
    #     hidden_states = (
    #         output[0].detach().clone()
    #     )  # Detach to cut off gradients from earlier operations
    #     hidden_states.requires_grad_()  # Re-enable gradient tracking on this detached tensor
    #     hidden_states.retain_grad()  # Ensure gradients are retained for later inspection
    #     activations[module] = hidden_states

    #     return (hidden_states,) + output[1:]

    def stop_gradient_activation_hook(module, hook_input, hook_output):
        """
        Forward hook that encodes the hidden states x using `sae`,
        then reconstructs them with x_hat. The difference (residual)
        is detached so it doesn't propagate gradient back through that part.
        """
        x_BLD = hook_output[0]  # hidden states

        with torch.no_grad():
            encoded_acts_BLF = sae.encode(x_BLD)
        x_hat = sae.decode(encoded_acts_BLF)
        residual = x_BLD - x_hat
        # Detach residual so we don't accumulate its gradient
        residual = residual.detach()

        x_recon = x_hat + residual
        # Retain grad so we can do backward on it
        x_recon.requires_grad_()
        x_recon.retain_grad()

        activations[module] = x_recon

        # If the module originally returned a tuple, preserve the structure
        return (x_recon,) + hook_output[1:]

    # Register hooks
    handles = []
    for layer_module in submodules:
        if use_stop_gradient:
            h = layer_module.register_forward_hook(stop_gradient_activation_hook)
        else:
            h = layer_module.register_forward_hook(save_activation_hook)
        handles.append(h)

    if use_activation_loss_fn:
        act_submodule = model_utils.get_submodule(transformers_model, 12)
        h = act_submodule.register_forward_hook(save_activation_hook)
        handles.append(h)

    try:
        # Forward pass
        output_logits_BLV = transformers_model(**model_inputs).logits

        if padding_side == "right":
            raise ValueError(
                "Refactor of loss functions required to support right padding"
            )
            seq_lengths_B = model_inputs["attention_mask"].sum(dim=1) - 1
            answer_logits_BV = output_logits_BLV[
                torch.arange(output_logits_BLV.shape[0]),
                seq_lengths_B,
                :,
            ]
        elif padding_side == "left":
            answer_logits_BV = output_logits_BLV[
                :,
                -1,
                :,
            ]

        if use_activation_loss_fn:
            # loss = activation_loss_fn(activations[act_submodule], labels_B)
            logits_BLV = apply_logit_lens(
                activations[act_submodule], transformers_model
            )
            yes_vs_no_loss_fn = make_yes_no_loss_fn(tokenizer, device=sae.W_dec.device)
            # yes_vs_no_loss_fn = make_max_logprob_loss_fn()
            loss = yes_vs_no_loss_fn(logits_BLV, labels_B)
            # loss = entropy_loss_fn(logits_BLV, labels_B)
        else:
            loss = loss_fn(output_logits_BLV, labels_B)
        loss.backward()

        predicted_tokens = tokenizer.batch_decode(answer_logits_BV.argmax(dim=-1))

        yes_probs_B, no_probs_B = model_utils.get_yes_no_probs(
            tokenizer, answer_logits_BV
        )

        if verbose:
            model_name = transformers_model.__class__.__name__
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            view_outputs(tokenizer, model_inputs["input_ids"], answer_logits_BV)

    finally:
        # Remove hooks to avoid side effects if function is called repeatedly
        for h in handles:
            h.remove()

    # Now gather the results
    results = {}
    decoder_weight_FD = sae.W_dec.data

    for layer_idx, layer_module in zip(chosen_layers, submodules, strict=True):
        # The post-hook activation
        x_BLD = activations[layer_module]
        x_grad_BLD = x_BLD.grad

        with torch.no_grad():
            encoded_acts_BLF = sae.encode(x_BLD)
            decoded_acts_BLD = sae.decode(encoded_acts_BLF)

            residual_BLD = x_BLD - decoded_acts_BLD

            grad_x_dot_decoder_BLF = torch.einsum(
                "bld,fd->blf", x_grad_BLD, decoder_weight_FD
            )

            if ignore_bos:
                if padding_side == "right":
                    encoded_acts_BLF[:, 0, :] = 0.0
                    residual_BLD[:, 0, :] = 0.0
                elif padding_side == "left":
                    bos_mask = (
                        model_inputs["attention_mask"].cumsum(dim=1) == 1
                    )  # shape [B, L], True where the first 1 occurs
                    # Now zero out the first attended token in each sequence
                    encoded_acts_BLF[bos_mask] = 0.0
                    residual_BLD[bos_mask] = 0.0

            node_effects_BLF = encoded_acts_BLF * grad_x_dot_decoder_BLF
            error_effects_BLD = residual_BLD * x_grad_BLD

            node_effects_BLF *= model_inputs["attention_mask"][:, :, None]
            grad_x_dot_decoder_BLF *= model_inputs["attention_mask"][:, :, None]
            error_effects_BLD *= model_inputs["attention_mask"][:, :, None]
            encoded_acts_BLF *= model_inputs["attention_mask"][:, :, None]
            x_BLD *= model_inputs["attention_mask"][:, :, None]

            # Store everything
            results[layer_idx] = {
                "effects_BLF": node_effects_BLF.detach(),
                "grad_x_dot_decoder_BLF": grad_x_dot_decoder_BLF.detach(),
                "encoded_acts_BLF": encoded_acts_BLF.detach(),
                "error_effects_BLD": error_effects_BLD.detach(),
                "acts_BLD": x_BLD.detach(),
                "labels": labels_B,
                "model_inputs": model_inputs,
                "predicted_tokens": predicted_tokens,
                "yes_log_probs_B": yes_probs_B,
                "no_log_probs_B": no_probs_B,
            }

    return results


def get_effects(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: base_sae.BaseSAE,
    dataloader: DataLoader,
    yes_vs_no_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    submodules: list[torch.nn.Module],
    chosen_layers: list[int],
    device: torch.device,
    use_activation_loss_fn: bool = False,
    verbose: bool = False,
) -> AttributionData:
    attribution_data = AttributionData(
        D=sae.W_dec.shape[1], F=sae.W_dec.shape[0], device=device
    )
    # TODO: Move to cuda for now, move to cpu at the end
    predicted_tokens = {"labels": [], "predicted_tokens": []}
    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_results = compute_attributions(
            model,
            tokenizer,
            sae,
            model_inputs,
            labels,
            chosen_layers,
            submodules,
            loss_fn=yes_vs_no_loss_fn,
            use_activation_loss_fn=use_activation_loss_fn,
            verbose=False,
            use_stop_gradient=False,
        )

        # Accumulate results from each batch
        with torch.no_grad():
            for layer in chosen_layers:
                attribution_data.update_from_batch(batch_results[layer])
                predicted_tokens["labels"].extend(labels.tolist())
                predicted_tokens["predicted_tokens"].extend(
                    batch_results[layer]["predicted_tokens"]
                )

    _ = analyze_prediction_rates(
        predicted_tokens["labels"],
        predicted_tokens["predicted_tokens"],
        verbose=verbose,
    )

    attribution_data.to("cpu")

    return attribution_data


def adjust_tensor_values(
    input_tensor: torch.Tensor, fallback_value: float = 100.0
) -> torch.Tensor:
    """
    Adjusts tensor values according to the following rules:
    - If x >= 1, keep x.
    - If 0 < x < 1, replace x with 1/x.
    - If x <= 0, replace x with fallback_value.
    """
    # Calculate reciprocal, handle potential division by zero/negative results temporarily
    reciprocal = torch.reciprocal(input_tensor)

    # Apply conditions using torch.where
    # Condition 1: x >= 1 -> keep original value
    # Condition 2: 0 < x < 1 -> use reciprocal
    # Condition 3: x <= 0 -> use fallback_value

    # First, distinguish between >= 1 and < 1
    adjusted = torch.where(input_tensor >= 1, input_tensor, reciprocal)

    # Then, handle the case where input_tensor <= 0, replacing potentially inf/negative reciprocals
    final_adjusted = torch.where(
        input_tensor <= 0,
        torch.tensor(fallback_value, dtype=input_tensor.dtype),
        adjusted,
    )

    return final_adjusted
