from transformers import ViTForImageClassification
from src.config import model_path
from src.dataset import VIT_Dataset

class VIT_Model:
    def __init__(self) -> None:
        labels = VIT_Dataset.ds['train'].features['label'].names

        self.model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
            )