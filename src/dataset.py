from datasets import load_dataset
from src.config import dataset_name

class VIT_Dataset:
    def __init__(self) -> None:
        self.ds = load_dataset(dataset_name)