import yaml
from pipeline import TrainingPipeline

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    pipeline = TrainingPipeline(config)
    pipeline.run()
