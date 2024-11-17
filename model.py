import os
import random
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel

import torchvision.models as models
import torchvision.transforms as transforms

import networkx as nx
from torch_geometric.nn import GCNConv

from deap import base, creator, tools, algorithms

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

nltk.download('punkt')
nltk.download('stopwords')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            features = features.view(features.size(0), -1) 
        return features


class MABSA_Model(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, num_classes):
        super(MABSA_Model, self).__init__()
        # Text and Image Embeddings
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.image_fc = nn.Linear(image_dim, hidden_dim)

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        # Graph Convolutional Network
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Classification Layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_features, image_features, edge_index):
        
        text_proj = F.relu(self.text_fc(text_features))  # Shape: (batch_size, hidden_dim)
        image_proj = F.relu(self.image_fc(image_features))  # Shape: (batch_size, hidden_dim)

        # Combine Text and Image Features
        combined = text_proj + image_proj  # Simple fusion; can be replaced with more complex methods

        # Prepare for Attention (Seq Length is 1)
        combined = combined.unsqueeze(1)  # Shape: (seq_length=1, batch_size, hidden_dim)
        attn_output, _ = self.attention(combined, combined, combined)  # Self-Attention

        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, hidden_dim)

        # Graph Convolutional Layers
        gcn_output = self.gcn1(attn_output, edge_index)
        gcn_output = F.relu(gcn_output)
        gcn_output = self.gcn2(gcn_output, edge_index)

        # Classification
        logits = self.classifier(gcn_output)

        return logits


def evaluate_model(individual, model, dataloader, criterion):

    learning_rate, batch_size, num_layers, dropout_rate = individual

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop (Single Epoch for Evaluation)
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image = batch['image'].to(device)
        sentiments = batch['sentiment'].to(device)

        # Forward Pass
        text_features = feature_extractor.extract_text_features(input_ids, attention_mask)
        image_features = feature_extractor.extract_image_features(image)
        
        batch_size_current = text_features.size(0)
        edge_index = torch.combinations(torch.arange(batch_size_current), r=2).t().contiguous().to(device)

        outputs = model(text_features, image_features, edge_index)
        loss = criterion(outputs, sentiments)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            sentiments = batch['sentiment'].to(device)

            text_features = feature_extractor.extract_text_features(input_ids, attention_mask)
            image_features = feature_extractor.extract_image_features(image)
            
            # Dummy edge_index for GCN
            batch_size_current = text_features.size(0)
            edge_index = torch.combinations(torch.arange(batch_size_current), r=2).t().contiguous().to(device)

            outputs = model(text_features, image_features, edge_index)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sentiments.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return (acc, )

# Setup Genetic Algorithm
def setup_ga():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize Accuracy
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define Hyperparameter Ranges
    toolbox.register("learning_rate", random.choice, [0.001, 0.005, 0.01])
    toolbox.register("batch_size", random.choice, [16, 32, 64])
    toolbox.register("num_layers", random.choice, [2, 3, 4])
    toolbox.register("dropout_rate", random.choice, [0.3, 0.5, 0.7])

    # Structure initial individuals
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.learning_rate, toolbox.batch_size, toolbox.num_layers, toolbox.dropout_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define Evaluation, Selection, Crossover, Mutation
    toolbox.register("evaluate", evaluate_model)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def train_and_evaluate(individual):

    learning_rate, batch_size, num_layers, dropout_rate = individual

    # Update DataLoader with New Batch Size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = MABSA_Model(text_dim=768, image_dim=2048, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Evaluate Model
    acc = evaluate_model(individual, model, val_loader, criterion)[0]
    return acc

if __name__ == "__main__":
    hidden_dim = 256
    num_classes = 3 

    data_path = 'dataset.csv'
    data = pd.read_csv(data_path)

    train_df = data.sample(frac=0.8, random_state=SEED)
    val_df = data.drop(train_df.index)

    feature_extractor = FeatureExtractor()

    train_dataset = MABSA_Dataset(train_df, feature_extractor.tokenizer, feature_extractor.image_transform)
    val_dataset = MABSA_Dataset(val_df, feature_extractor.tokenizer, feature_extractor.image_transform)

    toolbox = setup_ga()
    population = toolbox.population(n=10)  # Population size
    ngen = 5  

    print("Starting Genetic Algorithm Optimization...")
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, 
                        stats=stats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    print(f"Best Individual: {best_individual}")
    print(f"Best Fitness (Accuracy): {best_individual.fitness.values[0]}")

    learning_rate, batch_size, num_layers, dropout_rate = best_individual

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    final_model = MABSA_Model(text_dim=768, image_dim=2048, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

    epochs = 10
    for epoch in range(epochs):
        final_model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            sentiments = batch['sentiment'].to(device)

            # Extract Features
            text_features = feature_extractor.extract_text_features(input_ids, attention_mask)
            image_features = feature_extractor.extract_image_features(image)

            # Dummy edge_index for GCN (Fully Connected)
            batch_size_current = text_features.size(0)
            edge_index = torch.combinations(torch.arange(batch_size_current), r=2).t().contiguous().to(device)

            # Forward Pass
            outputs = final_model(text_features, image_features, edge_index)
            loss = criterion(outputs, sentiments)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Validation
        final_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image = batch['image'].to(device)
                sentiments = batch['sentiment'].to(device)

                text_features = feature_extractor.extract_text_features(input_ids, attention_mask)
                image_features = feature_extractor.extract_image_features(image)

                # Dummy edge_index for GCN
                batch_size_current = text_features.size(0)
                edge_index = torch.combinations(torch.arange(batch_size_current), r=2).t().contiguous().to(device)

                outputs = final_model(text_features, image_features, edge_index)
                _, preds = torch.max(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(sentiments.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f"Validation Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Final Evaluation on Validation Set
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            sentiments = batch['sentiment'].to(device)

            text_features = feature_extractor.extract_text_features(input_ids, attention_mask)
            image_features = feature_extractor.extract_image_features(image)

            # Dummy edge_index for GCN
            batch_size_current = text_features.size(0)
            edge_index = torch.combinations(torch.arange(batch_size_current), r=2).t().contiguous().to(device)

            outputs = final_model(text_features, image_features, edge_index)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sentiments.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print("\nFinal Evaluation on Validation Set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save the Trained Model
    torch.save(final_model.state_dict(), 'mabsa_model.pth')
    print("Model saved as mabsa_model.pth")