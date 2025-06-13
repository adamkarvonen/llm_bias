from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import field

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict


class InferenceMode(str, Enum):
    VLLM_INFERENCE = "vllm_inference"
    GPU_INFERENCE = "gpu_inference"
    GPU_FORWARD_PASS = "gpu_forward_pass"
    PERFORM_ABLATIONS = "perform_ablations"
    PROJECTION_ABLATIONS = "projection_ablations"
    OPEN_ROUTER = "open_router"
    LOGIT_LENS = "logit_lens"
    LOGIT_LENS_WITH_INTERVENTION = "logit_lens_with_intervention"

    def __str__(self):
        return self.value


class EvalConfig(BaseModel, extra="forbid"):
    inference_mode: InferenceMode

    random_seed: int = 42

    model_name: str = ""

    anti_bias_statement_file: str = ""
    job_description_file: str = ""
    system_prompt_filename: str = "yes_no.txt"
    anti_bias_statement_folder: str = "anti_bias_statements"
    job_description_folder: str = "job_descriptions"

    email_domain: str = "gmail"
    # Note: this is currently not used and is hardcoded in dataset.py
    resume_dataset_path: str = "data/resume/selected_cats_resumes.csv"
    score_output_dir: str = "score_output"

    industry: str = "INFORMATION-TECHNOLOGY"
    anthropic_dataset: bool = False
    downsample: int | None = None
    no_names: bool = False
    college_name_only: bool = False
    batch_size_multiplier: int = 2
    max_length_chars: int = 12000
    overwrite_existing_results: bool = False
    sae_intervention_type: str = "clamping"

    # For ablation experiments only
    scale: float = 1000.0
    bias_type: str = "N/A"
    model_names_to_iterate: list[str] = field(default_factory=list)
    anti_bias_statement_files_to_iterate: list[str] = field(default_factory=list)
    job_description_files_to_iterate: list[str] = field(default_factory=list)
    bias_types_to_iterate: list[str] = field(default_factory=list)
    scales_to_iterate: list[float] = field(default_factory=list)

    use_mean_diff: bool = True
    probe_training_lr: float = 3e-4
    probe_training_weight_decay: float = 0.05
    probe_training_early_stopping_patience: int = 50
    probe_training_max_iter: int = 500
    probe_training_batch_size: int = 4096
    probe_training_begin_layer_percent: int = 0
    probe_training_downsample: int | None = None
    probe_training_dataset_name: str = "anthropic"
    probe_training_overwrite_previous: bool = True
    probe_training_anti_bias_statement_file: str = "v2.txt"
    probe_training_job_description_file: str = "base_description.txt"
    probe_vectors_dir: str = "ablation_vectors"
    test_mode: bool = False

    orthogonalize_model: bool = False

    # ------------- convenience IO helpers -------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)  # full type check

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump()))


class FrozenEvalConfig(EvalConfig):
    model_config = ConfigDict(frozen=True)
