from transformers import Trainer
from src.backbone import VIT_Model
from src.config import training_args
from src.collator import collate_fn
from src.evaluate import compute_metrics
from src.feature_extractor import Feature_Extractor

class VIT_Train:
    def __init__(self) -> None:
        self.prepared_ds = Feature_Extractor().apply_transform()
        self.trainer = Trainer(
            model=VIT_Model().model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=self.prepared_ds["train"],
            tokenizer=Feature_Extractor().feature_extractor,
            )
    
    def train(self):
        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()
    
    def evaluate(self):
        metrics = self.trainer.evaluate(self.prepared_ds['validation'])
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)


