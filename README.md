# Robustly Improving LLM Fairness in Realistic Settings via Interpretability

This repository contains the code for the paper [Robustly Improving LLM Fairness in Realistic Settings via Interpretability](https://arxiv.org/abs/2506.10922).

We evaluate and mitigate race/gender hiring bias in LLMs. The code supports both local GPU inference and API-based models via OpenRouter. For open models, we implement inference-time interventions to reduce bias.

## Setup

Create and activate a virtual environment, then:

```bash
pip install -e .
python data_setup.py
```

**For local gated models:** Run `huggingface-cli login --token {my_token}`  
**For OpenRouter models:** Create a file named `openrouter_api_key.txt` with your API key

## Running Experiments

To evaluate bias on local models:
```bash
python mypkg/main_paper_dataset.py --config configs/base_experiment.yaml
```

Other experiment configurations:
- `base_experiment_intervention.yaml` - Run bias interventions on local models
- `openrouter_experiment.yaml` - Evaluate bias using API models

- Other experiment configs are avaiable in `configs/`.

To generate graphs from results: `python paper_graphing.ipynb`

## Reproducing Paper Results

- Experiment logs from the paper: https://huggingface.co/datasets/adamkarvonen/bias_eval/blob/main/paper_data_final.zip
- To replicate specific paper figures, use `paper_graphing.ipynb` with the downloaded data