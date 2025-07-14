import yaml
import pandas as pd

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_data_by_version(version, config_path="config/data_versions.yaml"):
    data_versions = load_yaml(config_path)["data_versions"]
    if version not in data_versions:
        raise ValueError(f"Data version {version} not found.")
    entry = data_versions[version]
    df = pd.read_csv(entry["path"])
    return df, entry["description"]
