from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn


class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = (
            1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
            return False

    # @torch.no_grad()
    # def test_sae(self, model_name: str):
    #     assert self.W_dec.shape == (self.cfg.d_sae, self.cfg.d_in)
    #     assert self.W_enc.shape == (self.cfg.d_in, self.cfg.d_sae)

    #     # TODO: Refactor to use AutoModelForCausalLM
    #     model = HookedTransformer.from_pretrained(model_name, device=self.device)

    #     test_input = "The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white unicorns were previously unknown to science"

    #     _, cache = model.run_with_cache(
    #         test_input,
    #         prepend_bos=True,
    #         names_filter=[self.cfg.hook_name],
    #         stop_at_layer=self.cfg.hook_layer + 1,
    #     )
    #     acts = cache[self.cfg.hook_name]

    #     encoded_acts = self.encode(acts)
    #     decoded_acts = self.decode(encoded_acts)

    #     flattened_acts = einops.rearrange(acts, "b l d -> (b l) d")
    #     reconstructed_acts = self(flattened_acts)
    #     # match flattened_acts with decoded_acts
    #     reconstructed_acts = reconstructed_acts.reshape(acts.shape)

    #     assert torch.allclose(reconstructed_acts, decoded_acts)

    #     l0 = (encoded_acts[:, 1:] > 0).float().sum(-1).detach()
    #     print(f"average l0: {l0.mean().item()}")
