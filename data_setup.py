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


if __name__ == "__main__":
    setup_data()
