import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import einops

from datasets import load_dataset
from typing import Optional

from mypkg.pipeline.infra.hiring_bias_prompts import ResumePromptResult


def dataset_to_df(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "bias_in_bios":
        dataset = load_dataset("LabHC/bias_in_bios")
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])

    return train_df, test_df


def filter_and_balance_professions(
    df: pd.DataFrame,
    profession_ids: list[int],
    use_gender_question: bool,
    random_seed: int = 42,
    min_count: Optional[int] = 100,
) -> tuple[list[str], list[int]]:
    """
    Filter DataFrame to specified professions and balance the samples.

    Args:
        df: Input DataFrame containing 'profession' column
        profession_ids: List of profession IDs to filter for
        random_seed: Random seed for reproducibility

    Returns:
        Balanced DataFrame containing equal samples from each profession
    """
    # Filter for specified professions
    filtered_df = df[df["profession"].isin(profession_ids)]

    # Get counts for each profession
    prof_counts = filtered_df["profession"].value_counts()

    if min_count is None:
        min_count = prof_counts.min()

    # Balance the dataset by sampling min_count samples from each profession
    balanced_df = pd.concat(
        [
            filtered_df[filtered_df["profession"] == prof].sample(
                n=min_count, random_state=random_seed
            )
            for prof in profession_ids
        ]
    ).reset_index(drop=True)

    texts = balanced_df[
        "hard_text"
    ].tolist()  # replace 'text' with the actual text column name

    if use_gender_question:
        labels = balanced_df["gender"].apply(lambda x: 0 if x == 0 else 1).tolist()
        for i in range(len(texts)):
            texts[i] += (
                "\nAnswer with only one word, yes or no. Question: Is the subject of this text a man? Answer:"
            )
    else:
        labels = balanced_df["profession"].apply(lambda x: 0 if x == 21 else 1).tolist()
        for i in range(len(texts)):
            texts[i] += (
                "\nAnswer with only one word, yes or no. Question: Is the subject of this text a professor? Answer:"
            )

    # for i in range(len(texts)):
    #     texts[i] += "\nDescribe the subject of this text in detail. Answer:"

    return texts, labels


def create_simple_dataloader(
    texts: list[str],
    labels: list[int],
    prompts: list[ResumePromptResult],
    model_name: str,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 8192,
    shuffle: bool = False,
    sort_by_length: bool = True,
    padding_side: str = "left",
):
    """
    Create a PyTorch DataLoader from lists of texts and corresponding labels with dynamic padding.

    Instead of padding all texts to the length of the longest text in the dataset,
    this function tokenizes without padding, sorts the data by sequence length
    (longest first), and uses a custom collate function to pad each batch dynamically.

    Args:
        texts: List of input strings.
        labels: List of integer labels (e.g., 0 or 1).
        model_name: HuggingFace model identifier to load the tokenizer.
        batch_size: Batch size for the DataLoader.

    Returns:
        A PyTorch DataLoader with dynamically padded batches.
    """

    assert padding_side in ["left", "right"]

    assert len(texts) == len(labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        print("No pad token found, setting eos token as pad token")
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all texts without padding.
    # This returns lists of token IDs and attention masks for each sample.
    encoded = tokenizer(
        texts,
        padding=False,  # Do not pad here.
        truncation=False,
        return_tensors=None,  # Returns lists, not tensors.
        add_special_tokens=False,
    )

    # Create a list of samples as tuples: (input_ids, attention_mask, label)
    data = list(
        zip(
            encoded["input_ids"],
            encoded["attention_mask"],
            labels,
            list(range(len(texts))),
            prompts,
            strict=True,
        )
    )

    # Filter out samples exceeding max_length
    original_length = len(data)
    data = [sample for sample in data if len(sample[0]) <= max_length]
    new_length = len(data)

    if new_length < original_length:
        print(
            f"Filtered out {original_length - new_length} samples that exceeded max_length ({max_length})"
        )
        print(f"Original dataset size: {original_length}, new size: {new_length}")
        raise ValueError("We shouldn't be filtering out samples here anymore, it should be done in the main_paper_dataset.py file")

    if sort_by_length:
        assert shuffle is False, "Cannot shuffle if sorting by length"
        # Sort the data by the length of input_ids (largest sequences first)
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

    def custom_collate_fn(batch):
        """
        Collate function that dynamically pads each batch to the longest sequence in that batch.
        """
        # Determine the maximum sequence length in this batch.
        max_len = max(len(sample[0]) for sample in batch)
        padded_input_ids = []
        padded_attention_masks = []
        labels_batch = []
        idx_batch = []
        resume_prompt_results_batch = []
        for input_ids, attention_mask, label, idx, resume_prompt_results in batch:
            pad_length = max_len - len(input_ids)
            if padding_side == "right":
                padded_input_ids.append(
                    input_ids + [tokenizer.pad_token_id] * pad_length
                )
                padded_attention_masks.append(attention_mask + [0] * pad_length)
            elif padding_side == "left":
                padded_input_ids.append(
                    [tokenizer.pad_token_id] * pad_length + input_ids
                )
                padded_attention_masks.append([0] * pad_length + attention_mask)
            else:
                raise ValueError(f"Invalid padding side: {padding_side}")
            labels_batch.append(label)
            idx_batch.append(idx)
            resume_prompt_results_batch.append(resume_prompt_results)
        return (
            torch.tensor(padded_input_ids, device=device),
            torch.tensor(padded_attention_masks, device=device),
            torch.tensor(labels_batch, dtype=torch.long, device=device),
            torch.tensor(idx_batch, dtype=torch.long, device=device),
            resume_prompt_results_batch,
        )

    # Create the DataLoader using our custom collate function.
    # We set shuffle=False because our data is already sorted by length.
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
    )

    return dataloader


def dataset_to_list_of_strs(
    dataset_name: str, min_row_chars: int, total_chars: int
) -> list[str]:
    """
    Grab text data from a streaming dataset, stopping once we've collected total_chars.
    """
    # Adjust column names depending on dataset
    is_redpajama = dataset_name == "togethercomputer/RedPajama-Data-V2"
    # Example for your 'pile' dataset:
    # is_pile = dataset_name == "monology/pile-uncopyrighted"
    column_name = "raw_content" if is_redpajama else "text"

    dataset = load_dataset(
        dataset_name,
        name="sample-10B" if is_redpajama else None,
        trust_remote_code=True,
        streaming=True,
        split="train",
    )

    total_chars_so_far = 0
    result = []

    for row in dataset:
        text = row[column_name]
        if len(text) > min_row_chars:
            result.append(text)
            total_chars_so_far += len(text)
            if total_chars_so_far > total_chars:
                break
    return result


def tokenize_and_concat_dataset(
    tokenizer,
    dataset: list[str],
    seq_len: int,
    add_bos: bool = True,
    max_tokens: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """
    Concatenate text from the dataset with eos_token between chunks, then tokenize.
    Reshape into (B, seq_len) blocks. Truncates any partial remainder.
    """
    full_text = tokenizer.eos_token.join(dataset)

    # Divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [
        full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)
    ]
    all_tokens = []
    for chunk in chunks:
        token_ids = tokenizer(chunk)["input_ids"]
        all_tokens.extend(token_ids)

        # Append EOS token if missing.
        if not chunk.endswith(tokenizer.eos_token):
            chunk += tokenizer.eos_token
        token_ids = tokenizer(chunk)["input_ids"]
        all_tokens.extend(token_ids)

    tokens = torch.tensor(all_tokens)

    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len

    # Drop last partial batch if not full
    tokens = tokens[: num_batches * seq_len]
    tokens = einops.rearrange(
        tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
    )

    # Overwrite first token in each block with BOS if desired
    if add_bos:
        tokens[:, 0] = tokenizer.bos_token_id

    attention_mask = torch.ones_like(tokens)

    token_dict = {
        "input_ids": tokens,
        "attention_mask": attention_mask,
    }

    return token_dict


def load_and_tokenize_and_concat_dataset(
    dataset_name: str,
    ctx_len: int,
    num_tokens: int,
    tokenizer,
    add_bos: bool = True,
    min_row_chars: int = 100,
) -> dict[str, torch.Tensor]:
    """
    Load text from dataset_name, tokenize it, and return (B, ctx_len) blocks of tokens.
    """
    # For safety, let's over-sample from dataset (like you did with 5x)
    dataset_strs = dataset_to_list_of_strs(dataset_name, min_row_chars, num_tokens * 5)

    token_dict = tokenize_and_concat_dataset(
        tokenizer=tokenizer,
        dataset=dataset_strs,
        seq_len=ctx_len,
        add_bos=add_bos,
        max_tokens=num_tokens,
    )

    # Double-check we have enough tokens
    assert (
        token_dict["input_ids"].shape[0] * token_dict["input_ids"].shape[1]
    ) >= num_tokens, "Not enough tokens found!"
    return token_dict
