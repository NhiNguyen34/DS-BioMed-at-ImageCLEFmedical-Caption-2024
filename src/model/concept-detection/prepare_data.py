#import gdown
import os
import shutil

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

from transformers import (T5ForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer,
    AutoTokenizer, AutoFeatureExtractor,AutoImageProcessor,
    AutoModel,            
    TrainingArguments, Trainer,
    logging
)
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


logging1.disable(logging.INFO)
logging1.disable(logging.WARNING)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_DISABLED"] = "True"

enable_caching()
logging.set_verbosity_error()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

