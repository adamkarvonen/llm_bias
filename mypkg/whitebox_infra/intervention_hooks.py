import torch
import einops
from jaxtyping import Float
from torch import Tensor
from typing import Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

import mypkg.whitebox_infra.dictionaries.base_sae as base_sae
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.whitebox_infra.attribution as attribution

def get_sae_vectors(
    ablation_features: torch.Tensor, sae: base_sae.BaseSAE
) -> tuple[list[Float[Tensor, "d_model"]], list[Float[Tensor, "d_model"]], list[float]]:
    encoder_vectors = []
    decoder_vectors = []
    encoder_biases = []

    device = sae.W_dec.device

    for i in range(ablation_features.shape[0]):
        feature_idx = ablation_features[i]
        encoder_vector = sae.W_enc[:, feature_idx]
        decoder_vector = sae.W_dec[feature_idx, :]
        encoder_bias = sae.b_enc[feature_idx]

        if hasattr(sae, "threshold") and hasattr(sae, "use_threshold"):
            # batch top k or top k
            encoder_bias += sae.threshold
        elif hasattr(sae, "threshold"):
            encoder_bias += sae.threshold[feature_idx]

        encoder_vector = encoder_vector.to(device)
        decoder_vector = decoder_vector.to(device)
        encoder_bias = encoder_bias.to(device)

        encoder_vectors.append(encoder_vector)
        decoder_vectors.append(decoder_vector)
        encoder_biases.append(encoder_bias)

    return encoder_vectors, decoder_vectors, encoder_biases


def get_conditional_clamping_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally clamps activations.

    Combines conditional intervention with clamping - only clamps activations
    when they exceed the encoder threshold with the decoder intervention.

    Args:
        encoder_vectors: List of vectors defining directions to monitor
        decoder_vectors: List of vectors defining intervention directions
        scales: Target values for clamping
        encoder_thresholds: Threshold values that trigger clamping

    Returns:
        Hook function that conditionally clamps and modifies activations

    Note:
        Most sophisticated intervention type, combining benefits of
        conditional application and activation clamping
    """

    # K = number of encoder/decoder directions
    enc_mat_KD: Float[Tensor, "K d_model"] = torch.stack(
        encoder_vectors
    )  # encoder directions
    dec_mat_KD: Float[Tensor, "K d_model"] = torch.stack(
        decoder_vectors
    )  # decoder directions

    scales_K: Float[Tensor, "K"] = torch.as_tensor(
        scales, dtype=enc_mat_KD.dtype, device=enc_mat_KD.device
    )
    thresholds_K: Float[Tensor, "K"] = torch.as_tensor(
        encoder_thresholds, dtype=enc_mat_KD.dtype, device=enc_mat_KD.device
    )

    def hook_fn(module, _input, output):
        resid_BLD: Float[Tensor, "batch seq_len d_model"] = output[0]

        # Dot product with every encoder dir in one shot -> feats_BLK
        feats_BLK: Float[Tensor, "batch seq_len K"] = torch.einsum(
            "bld,kd->blk", resid_BLD, enc_mat_KD
        )

        # Boolean mask for positions where we clamp
        mask_BLK: torch.BoolTensor = (feats_BLK > thresholds_K) & (feats_BLK > 0)

        # Compute intervention shift Δ_BLKD
        delta_BLKD: Float[Tensor, "batch seq_len K d_model"] = (
            (scales_K - feats_BLK) * mask_BLK
        ).unsqueeze(-1) * dec_mat_KD  # broadcast over d_model

        delta_BLD = einops.reduce(delta_BLKD, "B L K D -> B L D", "sum")

        resid_BLD = resid_BLD + delta_BLD

        return (resid_BLD,) + output[1:]

    return hook_fn

def orthogonalize_vectors(dirs_KD: Float[Tensor, "K d_model"]) -> Float[Tensor, "K d_model"]:
    dirs_KD = torch.nn.functional.normalize(dirs_KD, dim=1)

    # Orthogonalize the directions using QR decomposition
    # Transpose to (D, K) for QR, then transpose back
    dirs_DK = dirs_KD.T
    Q, R = torch.linalg.qr(dirs_DK, mode="reduced")
    # Q is (D, K) with orthonormal columns
    dirs_KD_ortho = Q.T  # Back to (K, D)

    # Note: If K > D or vectors are linearly dependent, some columns of Q might be zero
    # We should filter those out
    norms = torch.norm(dirs_KD_ortho, dim=1)
    valid_mask = norms > 1e-6
    dirs_KD_ortho = dirs_KD_ortho[valid_mask]

    return dirs_KD_ortho



def orthogonalize_matrix_to_directions(
    W: Float[Tensor, "d_out d_in"], 
    dirs_KD: Float[Tensor, "k d_out"]
) -> Float[Tensor, "d_out d_in"]:
    """
    Orthogonalize a weight matrix W so it doesn't write to specified directions.
    
    W_new = W - sum_k (r_k @ r_k.T @ W)
    
    where r_k are normalized direction vectors.
    """
    W_ortho = W.clone().to(dtype=torch.float32)
    
    assert W_ortho.shape[0] == dirs_KD.shape[1], "Number of rows in W must match number of directions"

    dirs_KD_ortho = orthogonalize_vectors(dirs_KD)
    
    # Project out each orthogonalized direction
    for direction in dirs_KD_ortho:
        direction = direction / (direction.norm() + 1e-8)  # Normalize
        # W_new = W - r @ r.T @ W
        projection = torch.outer(direction, direction) @ W_ortho
        W_ortho = W_ortho - projection

    return W_ortho.to(dtype=torch.bfloat16)

def get_model_layers(model):
    """Get the layers attribute regardless of model architecture."""
    if hasattr(model, 'language_model'):
        return model.language_model.layers
    elif hasattr(model, 'model'):
        return model.model.layers
    elif hasattr(model, 'transformer'):
        return model.transformer.h  # For GPT-2 style models
    else:
        raise AttributeError(f"Could not find layers in model of type {type(model)}")

def get_embed_tokens(model):
    """Get the embedding layer regardless of model architecture."""
    if hasattr(model, 'language_model'):
        return model.language_model.embed_tokens
    elif hasattr(model, 'model'):
        return model.model.embed_tokens
    elif hasattr(model, 'transformer'):
        return model.transformer.wte  # For GPT-2 style models
    else:
        raise AttributeError(f"Could not find embed_tokens in model of type {type(model)}")

def orthogonalize_model_weights(
    model: AutoModelForCausalLM,
    layer_directions: dict[int, list[Float[Tensor, "d_model"]]]
):
    """
    Orthogonalize all residual stream writes in the model.
    
    Args:
        model: The transformer model
        layer_directions: Dict mapping layer index to list of directions
    """
    print(f"Orthogonalizing model weights for {len(layer_directions)} layers")

    for layer_idx, layer_dir in layer_directions.items():
        layer_directions[layer_idx] = orthogonalize_vectors(torch.stack(layer_dir).to(dtype=torch.float32))

    layers = get_model_layers(model)
    embed_tokens = get_embed_tokens(model)

    # 1. Orthogonalize embedding matrix (affects all layers)
    if 0 in layer_directions:
        embed_tokens.weight.data = orthogonalize_matrix_to_directions(
            embed_tokens.weight.data.T,  # Transpose to (d_model, vocab_size)
            layer_directions[0]
        ).T  # Transpose back
    
    # 2. Orthogonalize each layer's output projections
    for layer_idx, layer in enumerate(layers):
        if layer_idx not in layer_directions:
            continue
            
        directions = layer_directions[layer_idx]
        
        # Attention output projection
        layer.self_attn.o_proj.weight.data = orthogonalize_matrix_to_directions(
            layer.self_attn.o_proj.weight.data,
            directions
        )
        
        # MLP output projection  
        layer.mlp.down_proj.weight.data = orthogonalize_matrix_to_directions(
            layer.mlp.down_proj.weight.data,
            directions
        )
    
    return model

def assert_orthonormal_vectors(dirs_KD: Float[Tensor, "K d_model"], rtol: float = 1e-3, atol: float = 1e-3) -> None:
    """
    Assert that vectors are unit norm and mutually orthogonal within tolerance.
    
    Args:
        dirs_KD: Tensor of shape (K, d_model) containing K vectors
        rtol: Relative tolerance for comparisons
        atol: Absolute tolerance for comparisons
    """
    K, D = dirs_KD.shape
    
    # Check unit norm for each vector
    norms = torch.norm(dirs_KD, dim=1)
    assert torch.allclose(norms, torch.ones(K, device=dirs_KD.device), rtol=rtol, atol=atol), \
        f"Vectors not unit norm. Norms: {norms.tolist()}"
    
    # Check orthogonality by computing pairwise dot products
    # Should get identity matrix
    gram_matrix = dirs_KD @ dirs_KD.T  # (K, K)
    expected = torch.eye(K, device=dirs_KD.device)
    
    assert torch.allclose(gram_matrix, expected, rtol=rtol, atol=atol), \
        f"Vectors not orthogonal. Max off-diagonal: {(gram_matrix - expected).abs().max().item():.6e}"

def get_projection_ablation_hook(
    ablate_vectors: Float[Tensor, "K d_model"],
    ablate_biases: Float[Tensor, "K"],
) -> Callable:
    """
    Returns a hook that *ablates* (projects out) one or more directions
    from every write to the residual stream.

    Args
    ----
    ablate_vectors : list of direction vectors r_k  (len K)

    Behaviour
    ---------
    For each residual write `resid_BLD`, we compute

        resid_BLD ← resid_BLD  − Σ_k (resid_BLD · r_k) r_k

    so the output tensor is (approximately) orthogonal to every r_k.
    """

    dirs_KD = ablate_vectors.to(dtype=torch.float32)

    # assert_orthonormal_vectors(dirs_KD)

    biases_K = ablate_biases.to(dtype=torch.float32)

    def hook_fn(module, _input, output):
        resid_BLD: Float[Tensor, "batch seq_len d_model"] = output[0]

        # Dot product with all directions → proj_BLK
        proj_BLK: Float[Tensor, "batch seq_len K"] = torch.einsum(
            "bld,kd->blk", resid_BLD.to(dtype=torch.float32), dirs_KD
        ) - biases_K[None, None, :]

        # Expand back to d_model and subtract
        delta_BLKD: Float[Tensor, "batch seq_len K d_model"] = (
            proj_BLK.unsqueeze(-1) * dirs_KD
        )
        delta_BLD: Float[Tensor, "batch seq_len d_model"] = delta_BLKD.sum(dim=2)

        resid_BLD = resid_BLD - delta_BLD.to(dtype=resid_BLD.dtype)

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_conditional_steering_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            max_act_B = feature_acts_BL.max(dim=1).values

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            # decoder_BLD = (
            #     feature_acts_BL[:, :, None] * coeff * decoder_vector_D[None, None, :]
            # )

            decoder_BLD = (
                decoder_vector_D[None, None, :] * max_act_B[:, None, None] * coeff
            )

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = torch.where(
                (intervention_mask_BL[:, :, None]),
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_constant_steering_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            decoder_BLD = coeff * decoder_vector_D[None, None, :]

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = resid_BLD + decoder_BLD

        return (resid_BLD,) + output[1:]

    return hook_fn

def get_ablation_hook(
    ablation_type: str,
    encoder_vectors: Float[Tensor, "k d_model"],
    decoder_vectors: Optional[Float[Tensor, "k d_model"]],
    scales: Optional[Float[Tensor, "k"]],
    encoder_biases: Optional[Float[Tensor, "k"]],
) -> Callable:
    pass

    if ablation_type == "clamping":
        ablation_hook = get_conditional_clamping_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "steering":
        ablation_hook = get_conditional_steering_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "constant":
        ablation_hook = get_constant_steering_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "projection_ablations":
        ablation_hook = get_projection_ablation_hook(encoder_vectors, encoder_biases)
    else:
        raise ValueError(f"Invalid ablation type: {ablation_type}")

    return ablation_hook
