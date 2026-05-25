import json
from pathlib import Path

def load_data(file_path: str) -> list:
    """Loads a JSON file containing TwiBot-20 benchmark dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_data_paths(data_dir: str = "data/raw") -> dict:
    """Returns a dictionary mapping partitions to raw file paths."""
    base_path = Path(data_dir)
    return {
        "train": base_path / "train.json",
        "dev": base_path / "dev.json",
        "test": base_path / "test.json"
    }
