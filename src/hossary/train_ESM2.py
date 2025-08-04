import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForSequenceClassification, EsmTokenizer

from torch.optim import AdamW  # Preferred method
# OR if you need Hugging Face's specific version:
# from transformers.optimization import AdamWimport pandas as pd

import numpy as np
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PeptideDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, use_helm=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_helm = use_helm

        # Convert label encoding if needed
        if isinstance(df['Label encoding'].iloc[0], str):
            self.labels = df['Label encoding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).values
        else:
            self.labels = df['Label encoding'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use HELM notation if specified and available
        if self.use_helm and pd.notna(self.df.iloc[idx]['HELM notation']):
            sequence = str(self.df.iloc[idx]['HELM notation'])
        else:
            sequence = str(self.df.iloc[idx]['Sequence'])

        labels = self.labels[idx]

        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

def prepare_data(df_path, test_size=0.2, random_state=42):
    # Load data
    df = pd.read_excel(df_path)

    # Convert string representations to lists if needed
    if isinstance(df['Function'].iloc[0], str):
        df['Function'] = df['Function'].apply(lambda x: x.split(',') if isinstance(x, str) else x)

    # If we don't have label encoding, create it
    if True or 'Label encoding' not in df.columns or df['Label encoding'].isna().all():
        # Using functions from preprocessing_analysis.py
        mlb = MultiLabelBinarizer()
        df['Function'] = df['Function'].apply(lambda x: [x] if isinstance(x, str) else x)
        binary_labels = mlb.fit_transform(df['Function'])
        df['Label encoding'] = list(binary_labels)
    else:
        # Load existing label binarizer or create new one
        try:
            with open('label_binarizer.pkl', 'rb') as f:
                mlb = pickle.load(f)
        except:
            mlb = MultiLabelBinarizer()
            # Need to fit even if we have label encoding to get class names
            mlb.fit(df['Function'].apply(lambda x: [x] if isinstance(x, str) else x))

    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_df, val_df, mlb

class ESM2ForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels, model_name="facebook/esm2_t6_8M_UR50D"): #esm2_t36_3B_UR50D
        super().__init__()
        self.esm = EsmForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Modify classifier head
        self.esm.classifier = nn.Sequential(
            nn.Linear(self.esm.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None
        )

        logits = outputs.logits[:, 0, :]
        probabilities = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return loss, probabilities

def train_model(train_df, val_df, num_labels, config):
    # Initialize tokenizer and model
    tokenizer = EsmTokenizer.from_pretrained(config['model_name'])
    model = ESM2ForMultiLabelClassification(num_labels=num_labels, model_name=config['model_name'])
    model.to(device)

    # Create datasets - can choose to use HELM notation
    use_helm = config.get('use_helm', False)
    train_dataset = PeptideDataset(train_df, tokenizer, max_length=config['max_length'], use_helm=use_helm)
    val_dataset = PeptideDataset(val_df, tokenizer, max_length=config['max_length'], use_helm=use_helm)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # # Convert labels to shape [batch_size, sequence_length, num_classes]
            # labels = labels.unsqueeze(1).expand(-1, config['max_length'], -1)  # [16, 512, 47]

            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                loss, probs = model(input_ids, attention_mask, labels)
                val_loss += loss.item()

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Calculate metrics
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        preds = (all_preds > 0.5).astype(int)

        f1_micro = f1_score(all_labels, preds, average='micro')
        f1_macro = f1_score(all_labels, preds, average='macro')
        accuracy = accuracy_score(all_labels, preds)
        roc_auc = roc_auc_score(all_labels, all_preds)

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"F1 Micro: {f1_micro:.4f} | F1 Macro: {f1_macro:.4f}")
        print(f"Accuracy: {accuracy:.4f} | ROC AUC: {roc_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")

        scheduler.step(f1_micro)

    return model

def convert_labels_to_functions(label_encoding, mlb):
    """Convert label encoding back to function names"""
    return mlb.inverse_transform(np.array(label_encoding).reshape(1, -1))[0]

def convert_functions_to_labels(functions, mlb):
    """Convert function names to label encoding"""
    return mlb.transform([functions])[0]

def predict_function(sequence, model_path="final_model", binarizer_path="label_binarizer.pkl", use_helm=False):
    # Load model and tokenizer
    model = ESM2ForMultiLabelClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenizer = EsmTokenizer.from_pretrained(model_path)

    # Load label binarizer
    with open(binarizer_path, 'rb') as f:
        mlb = pickle.load(f)

    # Tokenize input
    encoding = tokenizer(
        sequence,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        _, probs = model(input_ids, attention_mask)
        probs = probs.cpu().numpy().flatten()

    # Get predictions
    preds = (probs > 0.5).astype(int)
    functions = mlb.inverse_transform([preds])

    return functions[0], probs

def parse_helm_notation(helm_str):
    """
    Parse HELM notation to extract sequence and modifications
    Example: "PEPTIDE1{[ac].AIB.E.Y.[am]}$$$$" would return:
    - sequence: "AEY"
    - n_term: "ac" (acetylation)
    - c_term: "am" (amidation)
    - modifications: ["AIB"] (2-aminoisobutyric acid)
    """
    if pd.isna(helm_str):
        return None

    match = re.match(r'PEPTIDE1\{([^}]*)}\$\$\$\$', helm_str)
    if not match:
        return None

    parts = match.group(1).split('.')
    sequence = []
    modifications = []

    n_term = None
    c_term = None

    # Check for N-terminal modification
    if parts[0].startswith('[') and parts[0].endswith(']'):
        n_term = parts[0][1:-1]
        parts = parts[1:]

    # Check for C-terminal modification
    if parts[-1].startswith('[') and parts[-1].endswith(']'):
        c_term = parts[-1][1:-1]
        parts = parts[:-1]

    # Process remaining parts
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            modifications.append(part[1:-1])
        else:
            sequence.append(part)

    return {
        'sequence': ''.join(sequence),
        'n_term_mod': n_term,
        'c_term_mod': c_term,
        'modifications': modifications
    }

if __name__ == "__main__":
    # Configuration
    config = {
        'model_name': "facebook/esm2_t6_8M_UR50D", # esm2_t36_3B_UR50D
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 10,
        'max_length': 512,
        'test_size': 0.2,
        'use_helm': False  # Set to True to use HELM notation instead of raw sequence
    }

    # Prepare data
    data_folder = Path(r"C:\Users\aalhossary\OneDrive - wesleyan.edu\HtTYM" if os.name == 'nt' else '/smithlab/home/aalhossary/HtTYM/')
    datafile = "TPDB/main-D_GE5.xlsx"

    train_df, val_df, mlb = prepare_data(data_folder/datafile, test_size=config['test_size'])
    num_labels = len(mlb.classes_)
    print(f"Number of function classes: {num_labels}")

    # Save label binarizer
    with open('label_binarizer.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    # Train model
    model = train_model(train_df, val_df, num_labels, config)

    # Save final model and tokenizer
    model.esm.save_pretrained("final_model")
    tokenizer = EsmTokenizer.from_pretrained(config['model_name'])
    tokenizer.save_pretrained("final_model")

    # Example usage
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # Your peptide sequence
    functions, probabilities = predict_function(sequence)
    print(f"Predicted functions: {functions}")
    print(f"Probabilities: {probabilities}")

    # # Example HELM notation usage
    # helm_notation = "PEPTIDE1{[ac].AIB.E.Y.[am]}$$$$"
    # parsed_helm = parse_helm_notation(helm_notation)
    # print(f"Parsed HELM notation: {parsed_helm}")

    # Convert between label encoding and functions
    example_functions = ['Antimicrobial', 'Anticancer']
    label_encoding = convert_functions_to_labels(example_functions, mlb)
    print(f"Label encoding for {example_functions}: {label_encoding}")

    converted_back = convert_labels_to_functions(label_encoding, mlb)
    print(f"Converted back: {converted_back}")

