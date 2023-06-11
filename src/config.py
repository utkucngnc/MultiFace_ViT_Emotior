import torch
from transformers import TrainingArguments

dataset_name = 'sxdave/emotion_detection'

model_path = 'google/vit-base-patch16-224-in21k'

training_args = TrainingArguments(
  output_dir="./vit-emotior",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")