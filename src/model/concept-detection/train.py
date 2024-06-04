from prepare_data import dataset
from evaluate import compute_metrics
from transformers import AutoImageProcessor
from model import ConceptDetectionModel
from prepare_data import MyCollator
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
import numpy as np
from prepare_data import device
import os
from transformers import (
    TrainingArguments, Trainer
)
from prepare_data import prepare_data
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Image Captioning and Concept Detection')
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
parser.add_argument("--IMG_MODEL", type=str, required=True, help="Pretrained models")
args = parser.parse_args()

def createCollatorAndModel(image=args.IMG_MODEL):
    preprocessor = AutoImageProcessor.from_pretrained(image)
    collator = MyCollator(preprocessor=preprocessor)
    model = ConceptDetectionModel(pretrained_image_name=image,).to(device)
    return collator, model


def train(IMG_MODEL):
    os.environ["WANDB_DISABLED"] = "True"

    multi_args = TrainingArguments(
        output_dir="checkpoint",
        learning_rate = 5e-5,
        #warmup_ratio=0.05,
        weight_decay=0.05,
        gradient_accumulation_steps=2,
        seed=42, 
        evaluation_strategy='steps',
        logging_strategy='steps',
        save_strategy='steps',
        save_total_limit=2,
        eval_steps=100,
        logging_steps = 100,
        save_steps=100,
        metric_for_best_model='f1',
        per_device_train_batch_size=30,
        per_device_eval_batch_size=30,
        remove_unused_columns=False,
        num_train_epochs=5,
        fp16=True,
        #dataloader_num_workers=2,
        load_best_model_at_end=True,
        disable_tqdm=False
    )

    torch.cuda.empty_cache()
    # Initialize the actual collator and multimodal model
    collator, model = createCollatorAndModel()

    # Initialize the trainer with the dataset, collator, model, hyperparameters and evaluation metrics
    multi_trainer = Trainer(
        model,
        multi_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    
    print(sum(p.numel() for p in model.parameters()))
    return multi_trainer
    

if __name__ == "__main__":

    dataset, my_collator = prepare_data(args)
    multi_trainer = train()
    # Start the training loop
    train_multi_metrics = multi_trainer.train()