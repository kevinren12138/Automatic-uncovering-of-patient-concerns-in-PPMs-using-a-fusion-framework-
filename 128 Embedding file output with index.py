#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler, random_split
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AlbertModel, AlbertTokenizer, AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from itertools import product
from tqdm.notebook import tqdm


# In[20]:


def encode_data(tokenizer, texts, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length', 
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# In[10]:


class RobertaClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super(RobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if return_embedding:
            # Return the last hidden state
            last_hidden_state = outputs[0]  # The first element is the last hidden state
            return last_hidden_state

        # Continue with the normal workflow for classification
        cls_output = outputs[1]  # CLS token representation
        cls_output = self.classifier(cls_output)
        return cls_output

    
class RedditBERTClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super(RedditBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('Fan-s/reddit-tc-bert')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if return_embedding:
            # Return the last hidden state
            last_hidden_state = outputs[0]  # The first element is the last hidden state
            return last_hidden_state

        # Continue with the normal workflow for classification
        cls_output = outputs[1]  # CLS token representation
        cls_output = self.classifier(cls_output)
        return cls_output

    
class BioBERTClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super(BioBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if return_embedding:
            # Return the last hidden state
            last_hidden_state = outputs[0]  # The first element is the last hidden state
            return last_hidden_state

        # Continue with the normal workflow for classification
        cls_output = outputs[1]  # CLS token representation
        cls_output = self.classifier(cls_output)
        return cls_output

class TweetBERTClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super(TweetBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if return_embedding:
            # Return the last hidden state
            last_hidden_state = outputs[0]  # The first element is the last hidden state
            return last_hidden_state

        # Continue with the normal workflow for classification
        cls_output = outputs[1]  # CLS token representation
        cls_output = self.classifier(cls_output)
        return cls_output


# In[4]:


def undersample_data(df, class_label, n_splits):
    class_df = df[df['label'] == class_label]
    return np.array_split(class_df, n_splits)


# In[5]:


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, file_names, input_ids, attention_masks, labels):
        self.texts = texts
        self.file_names = file_names
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'file_names': self.file_names[idx],
            'input_ids': self.input_ids[idx],
            'attention_masks': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


# In[ ]:


data = pd.read_csv('file path')
df = pd.DataFrame(data=data)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

texts = df['Content'].values
labels = df['label'].values
file_names = df['FileName'].values

exp = 1
device = torch.device(f'cuda:{exp}’)


# In[13]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_indices = []
val_indices = []

for train_idx, val_idx in skf.split(texts, labels):
    train_indices.append(train_idx)
    val_indices.append(val_idx)


# In[14]:


batch_size = 4
epochs = 5
learning_rate = 1e-5


# In[ ]:


print("===========ROBERTA Balanced Weight============")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model_name = "roberta-BalancedWeight_128"
input_ids, attention_masks = encode_data(tokenizer, df['Content'].values)
labels = torch.tensor(df['label'].values)

metrics = {
    'Accuracy': [],
    'Recall_WEIGHTED': [],
    'Precision_WEIGHTED': [],
    'F1-score_WEIGHTED': [],
    'Recall_MACRO': [],
    'Precision_MACRO': [],
    'F1-score_MACRO': []
}

for fold in range(len(train_indices)):
    print(f"ROBERTA Balanced Weight Training on fold {fold + 1}") 
    
    train_texts, train_labels = texts[train_indices[fold]], labels[train_indices[fold]]
    val_texts, val_labels = texts[val_indices[fold]], labels[val_indices[fold]]
    
    train_file_names = file_names[train_indices[fold]]
    val_file_names = file_names[val_indices[fold]]
    
    train_inputs, train_masks = encode_data(tokenizer, train_texts)
    val_inputs, val_masks = encode_data(tokenizer, val_texts)
    
    train_dataset = TextDataset(train_texts, train_file_names, train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    val_dataset = TextDataset(val_texts, val_file_names, val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
    model = RobertaClassifier(num_labels=4)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels.numpy())
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_iterator = tqdm(train_dataloader, desc="Training")
        
        for step, batch in enumerate(train_iterator):
            
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_masks'].to(device)
            b_labels = batch['labels'].to(device)
    
            model.zero_grad()

            logits = model(b_input_ids, attention_mask=b_input_mask)
            
            loss = loss_fn(logits, b_labels) 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss}')
        
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Evaluation and get embedding file for testing set
    model.eval()
    validation_iterator = tqdm(val_dataloader, desc="Validation")
    
    for batch in val_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    df_embeddings.to_csv(f'{model_name}_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)
    
    print(f"\nROBERTA Balanced Weight Classification Report on fold {fold + 1}:\n")
    print(classification_report(all_true_labels, all_predictions, target_names=label_encoder.classes_))
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    precision2 = precision_score(all_true_labels, all_predictions, average='macro')
    recall2 = recall_score(all_true_labels, all_predictions, average='macro')
    f12 = f1_score(all_true_labels, all_predictions, average='macro')
    
    metrics['Accuracy'].append(accuracy)
    metrics['Recall_WEIGHTED'].append(recall)
    metrics['Precision_WEIGHTED'].append(precision)
    metrics['F1-score_WEIGHTED'].append(f1)
    metrics['Recall_MACRO'].append(recall2)
    metrics['Precision_MACRO'].append(precision2)
    metrics['F1-score_MACRO'].append(f12)
    
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Get embedding file for training set
    model.eval()
    train_iterator = tqdm(train_dataloader, desc="Training")
    
    for batch in train_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            # Obtain embeddings
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            # Obtain logits for predictions
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    # Prepare data for DataFrame
    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    # Add other information
    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    # Save to CSV
    df_embeddings.to_csv(f'{model_name}_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)

avg_accuracy = np.mean(metrics['Accuracy'])
avg_weighted_recall = np.mean(metrics['Recall_WEIGHTED'])
avg_weighted_precision = np.mean(metrics['Precision_WEIGHTED'])
avg_weighted_f1 = np.mean(metrics['F1-score_WEIGHTED'])
avg_macro_recall = np.mean(metrics['Recall_MACRO'])
avg_macro_precision = np.mean(metrics['Precision_MACRO'])
avg_macro_f1 = np.mean(metrics['F1-score_MACRO'])

print(f"ROBERTA_BalancedWeight_10folds_Average Accuracy: {avg_accuracy}")
print(f"ROBERTA_BalancedWeight_10folds_Average Weighted Recall: {avg_weighted_recall}")
print(f"ROBERTA_BalancedWeight_10folds_Average Weighted Precision: {avg_weighted_precision}")
print(f"ROBERTA_BalancedWeight_10folds_Average Weighted F1 Score: {avg_weighted_f1}")
print(f"ROBERTA_BalancedWeight_10folds_Average Macro Recall: {avg_macro_recall}")
print(f"ROBERTA_BalancedWeight_10folds_Average Macro Precision: {avg_macro_precision}")
print(f"ROBERTA_BalancedWeight_10folds_Average Macro F1 Score: {avg_macro_f1}")
print("---------------------------------------------------------------------")
    


# In[ ]:


print("===========REDDITBert Oversampling============")
model_name = "REDDITBert-Oversampling_128"
tokenizer = AutoTokenizer.from_pretrained('Fan-s/reddit-tc-bert')
input_ids, attention_masks = encode_data(tokenizer, df['Content'].values)
labels = torch.tensor(df['label'].values)

metrics = {
    'Accuracy': [],
    'Recall_WEIGHTED': [],
    'Precision_WEIGHTED': [],
    'F1-score_WEIGHTED': [],
    'Recall_MACRO': [],
    'Precision_MACRO': [],
    'F1-score_MACRO': []
}

for fold in range(len(train_indices)):
    print(f"ROBERTA Balanced Weight Training on fold {fold + 1}") 
    
    train_texts, train_labels = texts[train_indices[fold]], labels[train_indices[fold]]
    val_texts, val_labels = texts[val_indices[fold]], labels[val_indices[fold]]
    
    train_file_names = file_names[train_indices[fold]]
    val_file_names = file_names[val_indices[fold]]

    indices = np.arange(len(train_texts))

    X = np.column_stack((np.array(train_texts).reshape(-1, 1), indices.reshape(-1, 1)))
    y = train_labels

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    train_texts_resampled = X_resampled[:, 0]
    resampled_indices = X_resampled[:, 1].astype(int)

    train_file_names_resampled = [train_file_names[i] for i in resampled_indices]

    train_texts_resampled = train_texts_resampled.flatten()
    train_labels_resampled = torch.tensor(y_resampled)
    
    
    train_inputs, train_masks = encode_data(tokenizer, train_texts)
    train_inputs_resampled, train_masks_resampled = encode_data(tokenizer, train_texts_resampled)
    
    val_inputs, val_masks = encode_data(tokenizer, val_texts)
    
    
    train_dataset = TextDataset(train_texts, train_file_names, train_inputs, train_masks, train_labels)
    train_dataset_resampled = TextDataset(train_texts_resampled, train_file_names_resampled, train_inputs_resampled, train_masks_resampled, train_labels_resampled)
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    train_dataloader_resampled = DataLoader(train_dataset_resampled, sampler=RandomSampler(train_dataset_resampled), batch_size=batch_size)

    val_dataset = TextDataset(val_texts, val_file_names, val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
    model = RedditBERTClassifier(num_labels=4)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_iterator = tqdm(train_dataloader_resampled, desc="Training")
        
        for step, batch in enumerate(train_iterator):
            
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_masks'].to(device)
            b_labels = batch['labels'].to(device)
    
            model.zero_grad()

            logits = model(b_input_ids, attention_mask=b_input_mask)
            
            loss = loss_fn(logits, b_labels) 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader_resampled)
        print(f'Average training loss: {avg_train_loss}')
        
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Evaluation and get embedding file for testing set
    model.eval()
    validation_iterator = tqdm(val_dataloader, desc="Validation")
    
    for batch in val_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    df_embeddings.to_csv(f'{model_name}_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)
    
    print(f"\nREDDITBERT Oversampling Classification Report on fold {fold + 1}:\n")
    print(classification_report(all_true_labels, all_predictions, target_names=label_encoder.classes_))
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    precision2 = precision_score(all_true_labels, all_predictions, average='macro')
    recall2 = recall_score(all_true_labels, all_predictions, average='macro')
    f12 = f1_score(all_true_labels, all_predictions, average='macro')
    
    metrics['Accuracy'].append(accuracy)
    metrics['Recall_WEIGHTED'].append(recall)
    metrics['Precision_WEIGHTED'].append(precision)
    metrics['F1-score_WEIGHTED'].append(f1)
    metrics['Recall_MACRO'].append(recall2)
    metrics['Precision_MACRO'].append(precision2)
    metrics['F1-score_MACRO'].append(f12)
    
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Get embedding file for training set
    model.eval()
    train_iterator = tqdm(train_dataloader, desc="Training")
    
    for batch in train_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            # Obtain embeddings
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            # Obtain logits for predictions
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    # Prepare data for DataFrame
    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    # Add other information
    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    # Save to CSV
    df_embeddings.to_csv(f'{model_name}_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)

avg_accuracy = np.mean(metrics['Accuracy'])
avg_weighted_recall = np.mean(metrics['Recall_WEIGHTED'])
avg_weighted_precision = np.mean(metrics['Precision_WEIGHTED'])
avg_weighted_f1 = np.mean(metrics['F1-score_WEIGHTED'])
avg_macro_recall = np.mean(metrics['Recall_MACRO'])
avg_macro_precision = np.mean(metrics['Precision_MACRO'])
avg_macro_f1 = np.mean(metrics['F1-score_MACRO'])

print(f"REDDITBERT_Oversampling_10folds_Average Accuracy: {avg_accuracy}")
print(f"REDDITBERT_Oversampling_10folds_Average Weighted Recall: {avg_weighted_recall}")
print(f"REDDITBERT_Oversampling_10folds_Average Weighted Precision: {avg_weighted_precision}")
print(f"REDDITBERT_Oversampling_10folds_Average Weighted F1 Score: {avg_weighted_f1}")
print(f"REDDITBERT_Oversampling_10folds_Average Macro Recall: {avg_macro_recall}")
print(f"REDDITBERT_Oversampling_10folds_Average Macro Precision: {avg_macro_precision}")
print(f"REDDITBERT_Oversampling_10folds_Average Macro F1 Score: {avg_macro_f1}")
print("---------------------------------------------------------------------")
    


# In[ ]:


print("===========BioBert Balanced Weight============")
model_name = "BioBert-BalancedWeight_128"
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
input_ids, attention_masks = encode_data(tokenizer, df['Content'].values)
labels = torch.tensor(df['label'].values)

metrics = {
    'Accuracy': [],
    'Recall_WEIGHTED': [],
    'Precision_WEIGHTED': [],
    'F1-score_WEIGHTED': [],
    'Recall_MACRO': [],
    'Precision_MACRO': [],
    'F1-score_MACRO': []
}

for fold in range(len(train_indices)):
    print(f"BioBert Balanced Weight Training on fold {fold + 1}") 
    
    train_texts, train_labels = texts[train_indices[fold]], labels[train_indices[fold]]
    val_texts, val_labels = texts[val_indices[fold]], labels[val_indices[fold]]
    
    train_file_names = file_names[train_indices[fold]]
    val_file_names = file_names[val_indices[fold]]
    
    train_inputs, train_masks = encode_data(tokenizer, train_texts)
    val_inputs, val_masks = encode_data(tokenizer, val_texts)
    
    train_dataset = TextDataset(train_texts, train_file_names, train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    val_dataset = TextDataset(val_texts, val_file_names, val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
    model = BioBERTClassifier(num_labels=4)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels.numpy())
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_iterator = tqdm(train_dataloader, desc="Training")
        
        for step, batch in enumerate(train_iterator):
            
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_masks'].to(device)
            b_labels = batch['labels'].to(device)
    
            model.zero_grad()

            logits = model(b_input_ids, attention_mask=b_input_mask)
            
            loss = loss_fn(logits, b_labels) 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss}')
        
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Evaluation and get embedding file for testing set
    model.eval()
    validation_iterator = tqdm(val_dataloader, desc="Validation")
    
    for batch in val_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    df_embeddings.to_csv(f'{model_name}_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)
    
    print(f"\nBioBERT BalancedWeight Classification Report on fold {fold + 1}:\n")
    print(classification_report(all_true_labels, all_predictions, target_names=label_encoder.classes_))
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    precision2 = precision_score(all_true_labels, all_predictions, average='macro')
    recall2 = recall_score(all_true_labels, all_predictions, average='macro')
    f12 = f1_score(all_true_labels, all_predictions, average='macro')
    
    metrics['Accuracy'].append(accuracy)
    metrics['Recall_WEIGHTED'].append(recall)
    metrics['Precision_WEIGHTED'].append(precision)
    metrics['F1-score_WEIGHTED'].append(f1)
    metrics['Recall_MACRO'].append(recall2)
    metrics['Precision_MACRO'].append(precision2)
    metrics['F1-score_MACRO'].append(f12)
    
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Get embedding file for training set
    model.eval()
    train_iterator = tqdm(train_dataloader, desc="Training")
    
    for batch in train_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            # Obtain embeddings
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            # Obtain logits for predictions
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    # Prepare data for DataFrame
    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    # Add other information
    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    # Save to CSV
    df_embeddings.to_csv(f'{model_name}_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)

avg_accuracy = np.mean(metrics['Accuracy'])
avg_weighted_recall = np.mean(metrics['Recall_WEIGHTED'])
avg_weighted_precision = np.mean(metrics['Precision_WEIGHTED'])
avg_weighted_f1 = np.mean(metrics['F1-score_WEIGHTED'])
avg_macro_recall = np.mean(metrics['Recall_MACRO'])
avg_macro_precision = np.mean(metrics['Precision_MACRO'])
avg_macro_f1 = np.mean(metrics['F1-score_MACRO'])

print(f"BioBERT_BalancedWeight_10folds_Average Accuracy: {avg_accuracy}")
print(f"BioBERT_BalancedWeight_10folds_Average Weighted Recall: {avg_weighted_recall}")
print(f"BioBERT_BalancedWeight_10folds_Average Weighted Precision: {avg_weighted_precision}")
print(f"BioBERT_BalancedWeight_10folds_Average Weighted F1 Score: {avg_weighted_f1}")
print(f"BioBERT_BalancedWeight_10folds_Average Macro Recall: {avg_macro_recall}")
print(f"BioBERT_BalancedWeight_10folds_Average Macro Precision: {avg_macro_precision}")
print(f"BioBERT_BalancedWeight_10folds_Average Macro F1 Score: {avg_macro_f1}")
print("---------------------------------------------------------------------")
    


# In[ ]:


print("===========BERTweet Basic============")
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
model_name = "bertweet-base_128"
input_ids, attention_masks = encode_data(tokenizer, df['Content'].values)
labels = torch.tensor(df['label'].values)

metrics = {
    'Accuracy': [],
    'Recall_WEIGHTED': [],
    'Precision_WEIGHTED': [],
    'F1-score_WEIGHTED': [],
    'Recall_MACRO': [],
    'Precision_MACRO': [],
    'F1-score_MACRO': []
}

for fold in range(len(train_indices)):
    print(f"BERTweet Basic Training on fold {fold + 1}") 
    
    train_texts, train_labels = texts[train_indices[fold]], labels[train_indices[fold]]
    val_texts, val_labels = texts[val_indices[fold]], labels[val_indices[fold]]
    
    train_file_names = file_names[train_indices[fold]]
    val_file_names = file_names[val_indices[fold]]
    
    train_inputs, train_masks = encode_data(tokenizer, train_texts)
    val_inputs, val_masks = encode_data(tokenizer, val_texts)
    
    train_dataset = TextDataset(train_texts, train_file_names, train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

    val_dataset = TextDataset(val_texts, val_file_names, val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    
    model = TweetBERTClassifier(num_labels=4)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_iterator = tqdm(train_dataloader, desc="Training")
        
        for step, batch in enumerate(train_iterator):
            
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_masks'].to(device)
            b_labels = batch['labels'].to(device)
    
            model.zero_grad()

            logits = model(b_input_ids, attention_mask=b_input_mask)
            
            loss = loss_fn(logits, b_labels) 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss}')
        
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Evaluation and get embedding file for testing set
    model.eval()
    validation_iterator = tqdm(val_dataloader, desc="Validation")
    
    for batch in val_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    df_embeddings.to_csv(f'{model_name}_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)
    
    print(f"\nBERTweet Basic Classification Report on fold {fold + 1}:\n")
    print(classification_report(all_true_labels, all_predictions, target_names=label_encoder.classes_))
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    precision2 = precision_score(all_true_labels, all_predictions, average='macro')
    recall2 = recall_score(all_true_labels, all_predictions, average='macro')
    f12 = f1_score(all_true_labels, all_predictions, average='macro')
    
    metrics['Accuracy'].append(accuracy)
    metrics['Recall_WEIGHTED'].append(recall)
    metrics['Precision_WEIGHTED'].append(precision)
    metrics['F1-score_WEIGHTED'].append(f1)
    metrics['Recall_MACRO'].append(recall2)
    metrics['Precision_MACRO'].append(precision2)
    metrics['F1-score_MACRO'].append(f12)
    
    all_embeddings = []
    all_predictions = []
    all_true_labels = []
    all_texts = []
    all_file_names = []
    
    #Get embedding file for training set
    model.eval()
    train_iterator = tqdm(train_dataloader, desc="Training")
    
    for batch in train_dataloader:
        batch_texts = batch['texts']
        batch_file_names = batch['file_names']
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_masks = batch['attention_masks'].to(device)
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            # Obtain embeddings
            embeddings = model(batch_input_ids, attention_mask=batch_attention_masks, return_embedding=True)
            # Obtain logits for predictions
            logits = model(batch_input_ids, attention_mask=batch_attention_masks)

        embeddings = embeddings.mean(dim=1).detach().cpu().numpy()  # Take the mean of the embeddings
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_embeddings.extend(embeddings)
        all_true_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(preds)
        all_texts.extend(batch_texts)
        all_file_names.extend(batch_file_names)

    # Prepare data for DataFrame
    embedding_columns = [f'Embedding_{i}' for i in range(all_embeddings[0].shape[0])]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    # Add other information
    #df_embeddings['Text'] = all_texts
    df_embeddings['FileName'] = all_file_names
    df_embeddings['True Label'] = all_true_labels
    df_embeddings['Predicted Label'] = all_predictions

    # Save to CSV
    df_embeddings.to_csv(f'{model_name}_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv', index=False)

avg_accuracy = np.mean(metrics['Accuracy'])
avg_weighted_recall = np.mean(metrics['Recall_WEIGHTED'])
avg_weighted_precision = np.mean(metrics['Precision_WEIGHTED'])
avg_weighted_f1 = np.mean(metrics['F1-score_WEIGHTED'])
avg_macro_recall = np.mean(metrics['Recall_MACRO'])
avg_macro_precision = np.mean(metrics['Precision_MACRO'])
avg_macro_f1 = np.mean(metrics['F1-score_MACRO'])

print(f"BERTweet_Basic_10folds_Average Accuracy: {avg_accuracy}")
print(f"BERTweet_Basic_10folds_Average Weighted Recall: {avg_weighted_recall}")
print(f"BERTweet_Basic_10folds_Average Weighted Precision: {avg_weighted_precision}")
print(f"BERTweet_Basic_10folds_Average Weighted F1 Score: {avg_weighted_f1}")
print(f"BERTweet_Basic_10folds_Average Macro Recall: {avg_macro_recall}")
print(f"BERTweet_Basic_10folds_Average Macro Precision: {avg_macro_precision}")
print(f"BERTweet_Basic_10folds_Average Macro F1 Score: {avg_macro_f1}")
print("---------------------------------------------------------------------")
    


# In[ ]:




