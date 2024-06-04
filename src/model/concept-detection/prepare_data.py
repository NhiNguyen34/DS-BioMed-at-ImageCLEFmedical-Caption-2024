#import gdown
import os
import shutil
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

from PIL import Image
from IPython.display import display
import random
from copy import deepcopy
import json
from collections import Counter

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

#from aoa_pytorch import AoA
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, enable_caching

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import ast
import warnings
from pandas.errors import SettingWithCopyWarning
import logging as logging1
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.model_selection import train_test_split
import zipfile
import argparse
from transformers import AutoFeatureExtractor

logging1.disable(logging.INFO)
logging1.disable(logging.WARNING)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Set environment variables
os.environ['HF_HOME'] = os.path.join(".", "cache")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_DISABLED"] = "True"

# Enable caching and set verbosity
enable_caching()
logging.set_verbosity_error()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_SEED():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    
def prepare_data(args):
    """
    Prepare the data for training and evaluation.
    """
    set_SEED()
    data_path = os.path.join(args.data_dir, 'dataset-imageclef/merged_captions_concepts_.csv')
    one_hot_path = os.path.join(args.data_dir, 'dataset-imageclef/one_hot.pkl')
    train_images_path = os.path.join(args.data_dir, 'dataset/train_images.zip')
    val_images_path = os.path.join(args.data_dir, 'dataset/valid_images.zip')

    # Load data
    data = pd.read_csv(data_path)
    with open(one_hot_path, 'rb') as f:
        one_hot = pickle.load(f)

    # Split data
    train_set, val_set = train_test_split(data, test_size=0.0375, random_state=42)

    # Load image archives
    train_archive = zipfile.ZipFile(train_images_path, 'r')
    val_archive = zipfile.ZipFile(val_images_path, 'r')

    # Preprocess data
    train_set = train_set[['ID', 'labels']]
    val_set = val_set[['ID', 'labels']]

    train_set.to_csv(os.path.join(args.data_dir, "data_train.csv"), index=None)
    val_set.to_csv(os.path.join(args.data_dir, "data_eval.csv"), index=None)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(args.data_dir, "data_train.csv"),
            "val": os.path.join(args.data_dir, "data_eval.csv")
        }
    )

    @dataclass
    class MyCollator:
        preprocessor: AutoFeatureExtractor

        def preprocess_images(self, images: List[str]):
            try:
                processed_images = self.preprocessor(images=[Image.open(train_archive.open(f'train/{image_id}.jpg')).convert('RGB') for image_id in images],
                                                     return_tensors="pt",)
            except:
                processed_images = self.preprocessor(images=[Image.open(val_archive.open(f'valid/{image_id}.jpg')).convert('RGB') for image_id in images],
                                                     return_tensors="pt",)
            return {"pixel_values": processed_images['pixel_values'].squeeze(),}

        def __call__(self, raw_batch_dict):
            return {
                **self.preprocess_images(
                    raw_batch_dict['ID']
                    if isinstance(raw_batch_dict, dict) else
                    [i['ID'] for i in raw_batch_dict]
                ),
                'labels': torch.tensor(
                    raw_batch_dict['labels']
                    if isinstance(raw_batch_dict, dict) else
                    [ast.literal_eval(i['labels']) for i in raw_batch_dict],
                    dtype=torch.float32
                ),
            }

    return dataset, MyCollator
