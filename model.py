# Import Necessary Libraries
import os
import random
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Natural Language Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Transformers for BERT
from transformers import BertTokenizer, BertModel

# Computer Vision Libraries
import torchvision.models as models
import torchvision.transforms as transforms

# Graph Libraries
import networkx as nx
from torch_geometric.nn import GCNConv

# Genetic Algorithm Libraries
from deap import base, creator, tools, algorithms

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set Random Seeds for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Download NLTK Data
nltk.download('punkt')
nltk.download('stopwords')

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================
# 1. Data Preprocessing
# ===========================

class MABSA_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = self.preprocess_text(row['text'])
        aspect = row['aspect']
        sentiment = row['sentiment']
        image_path = row['image_path']

        # Tokenize Text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Process Image
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect': aspect,
            'image': image,
            'sentiment': sentiment
        }

# ===========================
# 2. Feature Extraction
# ===========================

class FeatureExtractor:
    def __init__(self):
        # Initialize BERT for Text
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert_model.eval()

        # Initialize ResNet for Images
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the classification layer
        self.resnet.to(device)
        self.resnet.eval()

        # Image Transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])

    def extract_text_features(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        return cls_embeddings

    def extract_image_features(self, images):
        images = images.to(device)
        with torch.no_grad():
            features = self.resnet(images)
            features = features.view(features.size(0), -1)  # Shape: (batch_size, feature_size)
        return features
