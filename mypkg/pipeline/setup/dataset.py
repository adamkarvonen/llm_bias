import pandas as pd
import random
from datasets import load_dataset


def load_raw_dataset() -> pd.DataFrame:
    """
    Loads the preprocessed CSV.
    Returns a Pandas DataFrame with all resumes and metadata.
    """
    dataset_path = "data/resume/selected_cats_resumes.csv"
    df = pd.read_csv(dataset_path)
    return df


def filter_by_industry(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows for the specified industry:
    'INFORMATION-TECHNOLOGY', 'CONSTRUCTION', or 'TEACHER'.
    """
    return df[df["Category"] == industry].reset_index(drop=True)


def balanced_downsample(
    df: pd.DataFrame, n_samples: int, random_seed: int = 42
) -> pd.DataFrame:
    """
    Downsample the dataset while maintaining balance across resume content.

    Args:
        df: DataFrame containing the dataset
        n_samples: Number of unique resume contents to sample
        random_seed: Random seed for reproducibility

    Returns:
        Downsampled DataFrame that maintains demographic variations for each resume
    """
    # Get unique resume strings
    unique_resumes = df["Resume_str"].unique()

    if n_samples > len(unique_resumes):
        print(
            f"Warning: Requested sample size ({n_samples}) is larger than unique resumes ({len(unique_resumes)}). Using all unique resumes."
        )
        return df

    # Randomly sample n_samples unique resumes
    sampled_resumes = random.sample(list(unique_resumes), n_samples)

    # Get all rows that contain these resume strings
    balanced_sample = df[df["Resume_str"].isin(sampled_resumes)]

    print(f"Downsampled to {len(sampled_resumes)} unique resumes")
    print(
        f"Total samples after maintaining demographic variations: {len(balanced_sample)}"
    )

    return balanced_sample


def load_full_anthropic_dataset(
    downsample_questions: int | None = None,
) -> pd.DataFrame:
    explicit_dataset = load_dataset("Anthropic/discrim-eval", "explicit")
    implicit_dataset = load_dataset("Anthropic/discrim-eval", "implicit")

    explicit_df = pd.DataFrame(explicit_dataset["train"])
    implicit_df = pd.DataFrame(implicit_dataset["train"])

    df = pd.concat([explicit_df, implicit_df])

    df["race"] = df["race"].str.lower()
    df["gender"] = df["gender"].str.lower()

    if downsample_questions is None:
        downsample_questions = int(df["decision_question_id"].max()) + 1

    retain_races = ["white", "black"]
    retain_ages = [20, 30, 40]
    retain_questions = list(range(0, downsample_questions))
    retain_genders = ["male", "female"]

    print(f"DF length before filtering: {len(df)}")

    df = df[df["race"].isin(retain_races)]
    df = df[df["age"].isin(retain_ages)]
    df = df[df["decision_question_id"].isin(retain_questions)]
    df = df[df["gender"].isin(retain_genders)]

    df = df.rename(
        columns={
            "race": "Race",
            "gender": "Gender",
        }
    )

    print(f"DF length after filtering: {len(df)}")

    return df
