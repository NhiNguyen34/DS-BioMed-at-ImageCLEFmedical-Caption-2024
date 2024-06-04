from transformers import AutoImageProcessor
from model import ConceptDetectionModel
from prepare_data import MyCollator
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
import numpy as np
from prepare_data import device


def get_answer(preds):
    p = np.where(np.array(preds)>=0.5,1,0)
    return p

# Function to compute all relevant performance metrics, to be passed into the trainer
def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = torch.sigmoid(torch.Tensor(logits)).numpy()
    preds = get_answer(preds)
    return {"acc": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average='samples'),
            "recall": recall_score(labels, preds, average='samples'),
            "f1": f1_score(labels, preds, average='samples'),}