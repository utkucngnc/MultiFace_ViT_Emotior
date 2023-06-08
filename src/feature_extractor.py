from transformers import ViTFeatureExtractor
from src.config import model_path

class Feature_Extractor:
    def __init__(self, model_path) -> None:
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
    
    def extract(self, image):
        return 