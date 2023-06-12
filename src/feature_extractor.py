from transformers import ViTFeatureExtractor
from src.config import model_path
from src.dataset import VIT_Dataset

class Feature_Extractor:
    def __init__(self) -> None:
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        self.ds=VIT_Dataset().ds
    
    def extract(self):
        batch = self.ds['train'][:]
        inputs = self.feature_extractor([x for x in batch['image']], return_tensors='pt')
        inputs['label'] = batch['label']
        return inputs
    
    def apply_transform(self):
        prepared_ds = self.ds.with_transform(self.extract())
        return prepared_ds