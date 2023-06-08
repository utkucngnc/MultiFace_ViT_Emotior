from datasets import load_dataset
from src.config import dataset_name, classes

class VIT_Dataset:
    def __init__(self, dataset_name, classes) -> None:
        self.ds = load_dataset(dataset_name)
        self.classes = []
        
        for ds_class in classes:
            self.classes.append(self.ds[ds_class])