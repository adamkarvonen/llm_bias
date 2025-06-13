import os
import zipfile
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download


def setup_data():
    # Get the root directory (where this script is located)
    root_dir = Path(__file__).parent

    # Path to the zip file
    zip_path = root_dir / "data" / "resume" / "Resume.csv.zip"

    # Path to extract to
    extract_path = root_dir / "data" / "resume"

    print("Unzipping Resume.csv.zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Running dataset creation script...")
    dataset_script = root_dir / "mypkg" / "pipeline" / "setup" / "create_dataset.py"
    subprocess.run(["python", str(dataset_script)], check=True)

    print("Setup completed successfully!")


def download_cached_responses():
    repo_id = "adamkarvonen/sae_max_acts"
    filename = "0410_data_v2.zip"
    local_dir = "data/cached_responses"

    zip_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=local_dir
    )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(local_dir)

    print("Unzipped cached_responses.zip")

    filename = "attribution_results_data.zip"
    local_dir = "data"

    zip_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=local_dir
    )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(local_dir)

    print("Unzipped attribution_results_data.zip")


def download_tuned_lenses():
    repo_id = "adamkarvonen/sae_max_acts"
    local_dir = "tuned_lenses"
    filename = "google_gemma-2-27b-it_tuned_lens.pt"

    hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=local_dir
    )


if __name__ == "__main__":
    setup_data()
    download_cached_responses()
