from transformers import ViTFeatureExtractor
from src.config import model_path
from src.dataset import VIT_Dataset

class Feature_Extractor:
    def __init__(self) -> None:
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        self.ds=VIT_Dataset.ds
    
    def extract(self):
        inputs = self.feature_extractor([x for x in self.ds['image']], return_tensors='pt')
        inputs['label'] = self.ds['label']
        return inputs
    
    def apply_transform(self):
        prepared_ds = self.ds.with_transform(self.extract)
        return prepared_ds