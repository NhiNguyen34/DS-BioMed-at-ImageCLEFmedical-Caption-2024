from transformers import (T5ForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer,
    AutoTokenizer, AutoFeatureExtractor,AutoImageProcessor,
    AutoModel,            
    TrainingArguments, Trainer,
    logging
)
import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple
from prepare_data import one_hot


class ConceptDetectionModel(nn.Module):
    """docstring for ConceptDetectionModel."""
    def __init__(self, pretrained_image_name, num_labels=len(one_hot.classes_), intermediate_dim=512, dropout=0.5):
        super(ConceptDetectionModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_image_name = pretrained_image_name
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image_name)
        self.linear = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size, intermediate_dim*2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(intermediate_dim*2, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(intermediate_dim, self.num_labels) #concat: intermediate_dim = 512
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_image_encoder(self,):
        return self.image_encoder
    
    def forward(self,
                pixel_values: torch.FloatTensor,
                labels: Optional[torch.LongTensor] = None):

        encoded_image = self.image_encoder(pixel_values=pixel_values,
                                           return_dict=True,)
        
        encoded_image = encoded_image['pooler_output'].squeeze()
        output = self.linear(encoded_image) #batch_size X dim
        
        #predict
        logits = self.classifier(output)
        
        out = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out
    