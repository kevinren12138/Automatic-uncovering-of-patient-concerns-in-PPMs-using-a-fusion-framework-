#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
import numpy as np
import regex as re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math
# Import necessary libraries
import pandas as pd
import numpy as np                               # Import numpy
from skimage import data, io   # Import skimage library (data - Test images and example data.
#                          io - Reading, saving, and displaying images.)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt                  # Import matplotlib.pyplot (Plotting framework in Python.)
get_ipython().run_line_magic('matplotlib', 'inline')
import os                                        # This module provides a portable way of using operating system dependent functionality.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')
from IPython.display import display
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 32)
#from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import seaborn as sns
import os
import spacy
from keras.layers import Dense, Embedding, LSTM, Reshape, Bidirectional
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
import keras.utils
from numpy import newaxis
from keras.initializers import random_uniform
from keras.optimizers import SGD
import tensorflow as tf
import tensorflow.keras.backend as K
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

from collections import Counter
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import torch


import numpy as np
import regex as re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math
import os
#import spacy
import numpy as np
import regex as re
import string
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D, Dropout, BatchNormalization, MaxPool2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import  xgboost, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import math
import os
import spacy
from sklearn.model_selection import GroupShuffleSplit

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel, AutoTokenizer, AutoModel, AutoModel 

from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim

from collections import Counter
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

import pandas as pd
import random, time
from babel.dates import format_date, format_datetime, format_time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

import torch
from torch import Tensor
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

import transformers, os
from transformers import BertModel, AutoModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #encode 
from sklearn import decomposition, ensemble

import  xgboost, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel

from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

from collections import Counter
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import torch
import tensorflow as tf
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.preprocessing import LabelEncoder
import keras.utils
import torch.optim as optim


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[117]:


model1_name = 'roberta-BalancedWeight'
model2_name = 'BioBert-BalancedWeight'
model3_name = 'bertweet-base'
exp = 3


# In[118]:


def classify_labels(row):
    labels = [row['Predicted Label_model1'], row['Predicted Label_model2'], row['Predicted Label_model3']]
    unique_labels = len(set(labels))
    if unique_labels == 3:
        return 'all_different'
    elif unique_labels == 2:
        return 'two_different'
    else:
        return 'all_same'


# In[119]:


all_same_testing_aggregated = pd.DataFrame()
diff_label_testing_aggregated = pd.DataFrame()
two_different_testing_aggregated = pd.DataFrame()
all_different_testing_aggregated = pd.DataFrame()

for fold in range(10):
    model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
    merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
    merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
    merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

    merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
    merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
    merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

    filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

    filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

    merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
    merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

    all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
    all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

    all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
    all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    
    all_same_testing_aggregated = all_same_testing_aggregated.append(all_same_testing, ignore_index=True)
    diff_label_testing_aggregated = diff_label_testing_aggregated.append(diff_label_testing, ignore_index=True)
    two_different_testing_aggregated = two_different_testing_aggregated.append(two_different_testing, ignore_index=True)
    all_different_testing_aggregated = all_different_testing_aggregated.append(all_different_testing, ignore_index=True)


   


# In[120]:


two_different_testing_aggregated.shape


# In[121]:


all_same_testing_aggregated.shape


# In[122]:


all_different_testing_aggregated.shape


# In[126]:


accuracy = accuracy_score(all_same_testing_aggregated['True Label'], all_same_testing_aggregated['Predicted Label_model3'])
accuracy


# In[127]:


accuracy1 = accuracy_score(two_different_testing_aggregated['True Label'], two_different_testing_aggregated['Predicted Label_model1'])
accuracy2 = accuracy_score(two_different_testing_aggregated['True Label'], two_different_testing_aggregated['Predicted Label_model2'])
accuracy3 = accuracy_score(two_different_testing_aggregated['True Label'], two_different_testing_aggregated['Predicted Label_model3'])

print(accuracy1)
print(accuracy2)
print(accuracy3)


# In[128]:


accuracy1 = accuracy_score(all_different_testing_aggregated['True Label'], all_different_testing_aggregated['Predicted Label_model1'])
accuracy2 = accuracy_score(all_different_testing_aggregated['True Label'], all_different_testing_aggregated['Predicted Label_model2'])
accuracy3 = accuracy_score(all_different_testing_aggregated['True Label'], all_different_testing_aggregated['Predicted Label_model3'])

print(accuracy1)
print(accuracy2)
print(accuracy3)


# In[129]:


def majority_vote_classification_report(df):
    predicted_cols = ['Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']
    
    df['Majority Vote'] = df[predicted_cols].mode(axis=1)[0]

    accuracy = accuracy_score(df['True Label'], df['Majority Vote'])

    return accuracy


# In[130]:


accuracy = majority_vote_classification_report(two_different_testing_aggregated)
print(accuracy)


# In[131]:


accuracy = majority_vote_classification_report(all_different_testing_aggregated)
print(accuracy)


# In[132]:


all_different_testing_aggregated


# In[569]:


#Majority Vote


# In[570]:


for fold in range(10):
    model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
    df1 = model1_testing_data[['FileName', 'True Label', 'Predicted Label']]
    df2 = model2_testing_data[['FileName', 'True Label', 'Predicted Label']]
    df3 = model3_testing_data[['FileName', 'True Label', 'Predicted Label']]

    combined_df = pd.concat([df1, df2, df3])

    majority_vote = combined_df.groupby('FileName')['Predicted Label'].agg(lambda x: mode(x.tolist()))
    majority_vote = majority_vote.reset_index()

    final_df = majority_vote.merge(df1[['FileName', 'True Label']], on='FileName')

    true_labels = final_df['True Label']
    predicted_labels = final_df['Predicted Label']

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #print(f"Fold {fold+1} Majority Vote Confusion Matrix:\n", conf_matrix)

    class_report = classification_report(true_labels, predicted_labels, target_names=['A', 'L', 'P', 'U'])
    print(f"Fold {fold+1} Majority Vote Classification Report:\n", class_report)


# In[ ]:


#Attention


# In[544]:


exp=3


# In[545]:


class AttentionClassifier(nn.Module):
    def __init__(self):
        super(AttentionClassifier, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(3, 1))
        self.fc1 = nn.Linear(768 * 3, 32)  
        self.dropout = nn.Dropout(0.1)   
        self.fc2 = nn.Linear(32, 4)       

    def forward(self, x):
        embeddings = x.view(-1, 3, 768)
        attention_scores = F.softmax(self.attention_weights, dim=0)
        attention_scores = attention_scores.view(1, 3, 1).expand_as(embeddings)
        attended_embeddings = embeddings * attention_scores
        attended_embeddings = attended_embeddings.view(-1, 768 * 3)
        
        x = F.relu(self.fc1(attended_embeddings))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

model = AttentionClassifier()


# In[546]:


for fold in range(10):
    model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
    merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
    merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
    merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

    merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
    merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
    merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

    filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

    filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

    merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
    merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

    all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
    all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

    all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
    diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
    all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


    model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
    model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
    model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

    model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
    model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
    model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

    embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

    model3_suffix_columns = {col: col + '_model3' for col in embeddings}
    model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
    model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

    combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

    combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
    embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
    X_train = combined_training_df[embeddings].values
    y_train = combined_training_df['True Label'].values

    X_test = combined_testing_df[embeddings].values
    y_test = combined_testing_df['True Label'].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            embedding = self.embeddings[idx]
            label = self.labels[idx]
            return embedding, label
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    if torch.cuda.is_available():
        class_weights_tensor = class_weights_tensor.cuda()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    num_epochs = 100 

    train_dataset = EmbeddingDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8)
    
    test_dataset = EmbeddingDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    for epoch in range(num_epochs):
        for embeddings, labels in train_loader:
            outputs = model(embeddings.float())
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for embeddings, labels in test_loader:
            outputs = model(embeddings.float())
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    conf_matrix = confusion_matrix(y_true, y_pred)
    #print(f"Fold {fold+1} Attention Diff Set Confusion Matrix:\n", conf_matrix)

    class_report = classification_report(y_true, y_pred)
    #print(f"Fold {fold+1} Attention Diff Set Classification Report:\n", class_report)
    
    y_true_same = all_same_testing['True Label']
    y_pred_same = all_same_testing['Predicted Label_model1']

    conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

    combined_conf_matrix = conf_matrix + conf_matrix_same

    y_true = np.array([])
    y_pred = np.array([])

    for true_class in range(4):
        for pred_class in range(4):
            count = combined_conf_matrix[true_class, pred_class]
            y_true = np.append(y_true, [true_class] * count)
            y_pred = np.append(y_pred, [pred_class] * count)

    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    print(f"Fold {fold+1} Attention Full Set Classification Report:\n", report)


# In[ ]:


#2D CNN with 3 embeddings


# In[510]:


exp=3


# In[536]:


fold = 1

class CNN2DClassifier0(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier0, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
 
model = CNN2DClassifier0()


# In[537]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[3]:


class CNN2DClassifier1(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier1, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier1()


# In[6]:


from tensorflow.keras import layers, models

# Define the Keras equivalent of the PyTorch model
class KerasCNN2DClassifier1(models.Model):
    def __init__(self, num_classes=4, input_shape=(768, 768, 1)):
        super(KerasCNN2DClassifier1, self).__init__()
        self.conv1 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape)
        self.pool = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.1)
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = layers.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = layers.ReLU()(self.conv2(x))
        x = self.pool(x)
        x = layers.ReLU()(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)  # Flatten the tensor
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Creating an instance of the Keras model
keras_model = KerasCNN2DClassifier1()

# Print the model summary
keras_model.build((None, 768, 768, 1))  # Specify the input shape
print(keras_model.summary())


# In[14]:


from tensorflow.keras import layers, models
import numpy as np
import visualkeras

# Convert to a sequential model
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', input_shape=(768, 768, 1)))
model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
model.add(layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(4, activation='softmax'))  # Adjust the number of classes if different

# Create a dummy input and pass it through the model
dummy_input = np.random.random((1, 768, 768, 1))
model.predict(dummy_input)

# Visualize the model
layered_view = visualkeras.layered_view(model, scale_xy=1, scale_z=1, max_z=1000, legend=True, font=font, color_map=color_map)
layered_view.show()  # Or save it if you want


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[514]:


y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])

for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(report)


# In[ ]:





# In[516]:


fold = 2


# In[517]:


class CNN2DClassifier2(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier2()


# In[518]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[519]:


fold = 3


# In[520]:


class CNN2DClassifier3(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier3, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=1, padding=1)

        # Calculate the size after convolutions and pooling
        # Adjusted to account for removing one convolutional layer
        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2), 64)  # Adjusted input size
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier3()


# In[521]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[522]:


fold = 4

class CNN2DClassifier4(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier4, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier4()


# In[523]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[524]:


fold = 5




# In[525]:


model = CNN2DClassifier4()


# In[526]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 50
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[527]:


fold = 6

class CNN2DClassifier6(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier6, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier6()


# In[528]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[529]:


fold = 7

class CNN2DClassifier7(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier7, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier7()


# In[530]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[531]:


fold = 8

class CNN2DClassifier8(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier8, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier8()


# In[532]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]


model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[533]:


fold = 9

class CNN2DClassifier9(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier9, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 3 * (768 // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier9()


# In[534]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_training = pd.merge(merged_training, model3_training_data[['FileName', 'Predicted Label']], on='FileName')
merged_training.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

merged_testing = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing = pd.merge(merged_testing, model3_testing_data[['FileName', 'Predicted Label']], on='FileName')
merged_testing.rename(columns={'Predicted Label': 'Predicted Label_model3'}, inplace=True)

filtered_merged_training = merged_training[(merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model2']) | 
                        (merged_training['Predicted Label_model1'] != merged_training['Predicted Label_model3']) | 
                        (merged_training['Predicted Label_model2'] != merged_training['Predicted Label_model3'])]

filtered_merged_testing = merged_testing[(merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model2']) | 
                        (merged_testing['Predicted Label_model1'] != merged_testing['Predicted Label_model3']) | 
                        (merged_testing['Predicted Label_model2'] != merged_testing['Predicted Label_model3'])]

merged_training['Classification'] = merged_training.apply(classify_labels, axis=1)
merged_testing['Classification'] = merged_testing.apply(classify_labels, axis=1)

all_different_training = merged_training[merged_training['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_training = merged_training[merged_training['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_training = merged_training[(merged_training['Classification'] == 'all_different') | (merged_training['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_training = merged_training[merged_training['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

all_different_testing = merged_testing[merged_testing['Classification'] == 'all_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
two_different_testing = merged_testing[merged_testing['Classification'] == 'two_different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]
diff_label_testing = merged_testing[(merged_testing['Classification'] == 'all_different') | (merged_testing['Classification'] == 'two_different')][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3', 'Classification']]
all_same_testing = merged_testing[merged_testing['Classification'] == 'all_same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2', 'Predicted Label_model3']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(diff_label_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(diff_label_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(diff_label_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(diff_label_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(diff_label_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

model3_suffix_columns = {col: col + '_model3' for col in embeddings}
model3_training_filtered = model3_training_filtered.rename(columns=model3_suffix_columns)
model3_testing_filtered = model3_testing_filtered.rename(columns=model3_suffix_columns)

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_training_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2')).merge(model3_testing_filtered[['FileName'] + [col + '_model3' for col in embeddings]], on='FileName', suffixes=('', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 4) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 3, 768)
X_test_reshaped = X_test.reshape(-1, 1, 3, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 100
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = all_same_testing['True Label']
y_pred_same = all_same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[535]:


#2D CNN with Majority Vote


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


from tensorflow.keras import layers, models
import numpy as np
import visualkeras

# Define the Keras equivalent of the updated PyTorch model using Functional API
input_shape = (2, 768, 1)  # Replace with actual input dimensions
inputs = layers.Input(shape=input_shape)

# Define the layers
x = layers.Conv2D(128, (2, 3), strides=1, padding='same')(inputs)
x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
x = layers.Conv2D(64, (2, 3), strides=1, padding='same')(x)
x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
x = layers.Conv2D(32, (2, 3), strides=1, padding='same')(x)
x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
x = layers.Flatten()(x)

# Manually calculate the input size for the first dense layer
# Replace the calculation below with the actual calculation based on your input size
fc1_input_size = 32 * 5 * 96  # Placeholder, adjust based on actual input size
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(4, activation='softmax')(x)

# Create the model
model = models.Model(inputs=inputs, outputs=outputs)

# Build the model with a sample input shape
model.build((None, *input_shape))

# Visualize the model
layered_view = visualkeras.layered_view(model, scale_xy=1, scale_z=1, max_z=1000, legend=True, font=font, color_map=color_map)
layered_view.show()  # Or save it if you want


# In[803]:


fold=0
exp=1

class CNN2DClassifier11(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier11, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 3), stride=1, padding=1)

        # Update the linear layer input size
        self.fc1 = nn.Linear(32 * 5 * 96, 64)  # Updated input size
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier11()



# In[774]:


model = CNN2DClassifier11()


def classify_labels12(row):
    labels = [row['Predicted Label_model1'], row['Predicted Label_model2']]
    unique_labels = len(set(labels))
    if unique_labels == 2:
        return 'different'
    else:
        return 'same'


# In[775]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_12 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing_12 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))

merged_training_12['Classification'] = merged_training_12.apply(classify_labels12, axis=1)
merged_testing_12['Classification'] = merged_testing_12.apply(classify_labels12, axis=1)

different_training = merged_training_12[merged_training_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_training = merged_training_12[merged_training_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

different_testing = merged_testing_12[merged_testing_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_testing = merged_testing_12[merged_testing_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 3) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 40
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[776]:


y_pred12=y_pred
y_true12=y_true


# In[777]:


def classify_labels13(row):
    labels = [row['Predicted Label_model1'], row['Predicted Label_model3']]
    unique_labels = len(set(labels))
    if unique_labels == 2:
        return 'different'
    else:
        return 'same'


# In[804]:


model = CNN2DClassifier11()


# In[807]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_13 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))
merged_testing_13 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))

merged_training_13['Classification'] = merged_training_13.apply(classify_labels13, axis=1)
merged_testing_13['Classification'] = merged_testing_13.apply(classify_labels13, axis=1)

different_training = merged_training_13[merged_training_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_training = merged_training_13[merged_training_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

different_testing = merged_testing_13[merged_testing_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_testing = merged_testing_13[merged_testing_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [1, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 5
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[780]:


y_pred13=y_pred
y_true13=y_true
y_true


# In[808]:


def classify_labels23(row):
    labels = [row['Predicted Label_model2'], row['Predicted Label_model3']]
    unique_labels = len(set(labels))
    if unique_labels == 2:
        return 'different'
    else:
        return 'same'
    
model = CNN2DClassifier11()


# In[809]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_23 = pd.merge(model2_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))
merged_testing_23 = pd.merge(model2_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))

merged_training_23['Classification'] = merged_training_23.apply(classify_labels23, axis=1)
merged_testing_23['Classification'] = merged_testing_23.apply(classify_labels23, axis=1)

different_training = merged_training_23[merged_training_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_training = merged_training_23[merged_training_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

different_testing = merged_testing_23[merged_testing_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_testing = merged_testing_23[merged_testing_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model2_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))

combined_testing_df = model2_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [2, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 10
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model2']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[ ]:





# In[810]:


fold=1
exp=1

class CNN2DClassifier11(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier11, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 3), stride=1, padding=1)

        # Update the linear layer input size
        self.fc1 = nn.Linear(32 * 5 * 96, 64)  # Updated input size
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier11()



# In[811]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_12 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing_12 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))

merged_training_12['Classification'] = merged_training_12.apply(classify_labels12, axis=1)
merged_testing_12['Classification'] = merged_testing_12.apply(classify_labels12, axis=1)

different_training = merged_training_12[merged_training_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_training = merged_training_12[merged_training_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

different_testing = merged_testing_12[merged_testing_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_testing = merged_testing_12[merged_testing_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 3) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 10
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[785]:


model = CNN2DClassifier11()


# In[786]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_13 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))
merged_testing_13 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))

merged_training_13['Classification'] = merged_training_13.apply(classify_labels13, axis=1)
merged_testing_13['Classification'] = merged_testing_13.apply(classify_labels13, axis=1)

different_training = merged_training_13[merged_training_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_training = merged_training_13[merged_training_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

different_testing = merged_testing_13[merged_testing_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_testing = merged_testing_13[merged_testing_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [1, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 60
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[787]:


model = CNN2DClassifier11()


# In[788]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_23 = pd.merge(model2_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))
merged_testing_23 = pd.merge(model2_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))

merged_training_23['Classification'] = merged_training_23.apply(classify_labels23, axis=1)
merged_testing_23['Classification'] = merged_testing_23.apply(classify_labels23, axis=1)

different_training = merged_training_23[merged_training_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_training = merged_training_23[merged_training_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

different_testing = merged_testing_23[merged_testing_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_testing = merged_testing_23[merged_testing_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model2_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))

combined_testing_df = model2_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [2, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 40
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model2']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[ ]:





# In[812]:


fold=2
exp=1

class CNN2DClassifier11(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier11, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 3), stride=1, padding=1)

        # Update the linear layer input size
        self.fc1 = nn.Linear(32 * 5 * 96, 64)  # Updated input size
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier11()



# In[790]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_12 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing_12 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))

merged_training_12['Classification'] = merged_training_12.apply(classify_labels12, axis=1)
merged_testing_12['Classification'] = merged_testing_12.apply(classify_labels12, axis=1)

different_training = merged_training_12[merged_training_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_training = merged_training_12[merged_training_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

different_testing = merged_testing_12[merged_testing_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_testing = merged_testing_12[merged_testing_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 3) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 80
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[791]:


model = CNN2DClassifier11()


# In[ ]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_13 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))
merged_testing_13 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))

merged_training_13['Classification'] = merged_training_13.apply(classify_labels13, axis=1)
merged_testing_13['Classification'] = merged_testing_13.apply(classify_labels13, axis=1)

different_training = merged_training_13[merged_training_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_training = merged_training_13[merged_training_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

different_testing = merged_testing_13[merged_testing_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_testing = merged_testing_13[merged_testing_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [1, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 30
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[793]:


model = CNN2DClassifier11()


# In[794]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_23 = pd.merge(model2_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))
merged_testing_23 = pd.merge(model2_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))

merged_training_23['Classification'] = merged_training_23.apply(classify_labels23, axis=1)
merged_testing_23['Classification'] = merged_testing_23.apply(classify_labels23, axis=1)

different_training = merged_training_23[merged_training_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_training = merged_training_23[merged_training_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

different_testing = merged_testing_23[merged_testing_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_testing = merged_testing_23[merged_testing_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model2_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))

combined_testing_df = model2_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [2, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 60
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model2']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[814]:


fold=3
exp=1

class CNN2DClassifier11(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier11, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 3), stride=1, padding=1)

        # Update the linear layer input size
        self.fc1 = nn.Linear(32 * 5 * 96, 64)  # Updated input size
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier11()



# In[796]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_12 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing_12 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))

merged_training_12['Classification'] = merged_training_12.apply(classify_labels12, axis=1)
merged_testing_12['Classification'] = merged_testing_12.apply(classify_labels12, axis=1)

different_training = merged_training_12[merged_training_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_training = merged_training_12[merged_training_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

different_testing = merged_testing_12[merged_testing_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_testing = merged_testing_12[merged_testing_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 3) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 80
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[797]:


model = CNN2DClassifier11()

model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_13 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))
merged_testing_13 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))

merged_training_13['Classification'] = merged_training_13.apply(classify_labels13, axis=1)
merged_testing_13['Classification'] = merged_testing_13.apply(classify_labels13, axis=1)

different_training = merged_training_13[merged_training_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_training = merged_training_13[merged_training_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

different_testing = merged_testing_13[merged_testing_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_testing = merged_testing_13[merged_testing_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [1, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 40
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[815]:


model = CNN2DClassifier11()

model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_23 = pd.merge(model2_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))
merged_testing_23 = pd.merge(model2_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))

merged_training_23['Classification'] = merged_training_23.apply(classify_labels23, axis=1)
merged_testing_23['Classification'] = merged_testing_23.apply(classify_labels23, axis=1)

different_training = merged_training_23[merged_training_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_training = merged_training_23[merged_training_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

different_testing = merged_testing_23[merged_testing_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_testing = merged_testing_23[merged_testing_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model2_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))

combined_testing_df = model2_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [2, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 20
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model2']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[816]:


fold=4
exp=1

class CNN2DClassifier11(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier11, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 3), stride=1, padding=1)

        # Update the linear layer input size
        self.fc1 = nn.Linear(32 * 5 * 96, 64)  # Updated input size
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN2DClassifier11()



# In[818]:


model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_12 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model2_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))
merged_testing_12 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model2_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model2'))

merged_training_12['Classification'] = merged_training_12.apply(classify_labels12, axis=1)
merged_testing_12['Classification'] = merged_testing_12.apply(classify_labels12, axis=1)

different_training = merged_training_12[merged_training_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_training = merged_training_12[merged_training_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

different_testing = merged_testing_12[merged_testing_12['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]
same_testing = merged_testing_12[merged_testing_12['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model2']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model2_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model2'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in range(1, 3) for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 5
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[801]:


model = CNN2DClassifier11()

model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_13 = pd.merge(model1_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))
merged_testing_13 = pd.merge(model1_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model1', '_model3'))

merged_training_13['Classification'] = merged_training_13.apply(classify_labels13, axis=1)
merged_testing_13['Classification'] = merged_testing_13.apply(classify_labels13, axis=1)

different_training = merged_training_13[merged_training_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_training = merged_training_13[merged_training_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

different_testing = merged_testing_13[merged_testing_13['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]
same_testing = merged_testing_13[merged_testing_13['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model1', 'Predicted Label_model3']]

model1_training_filtered = model1_training_data[model1_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model1_testing_filtered = model1_testing_data[model1_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model1_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))

combined_testing_df = model1_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model1', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [1, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 80
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model1']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[802]:


model = CNN2DClassifier11()

model1_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model1_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model1_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model2_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model2_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_training_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_training_embeddings_fold_{fold + 1}_exp_{exp}.csv')
model3_testing_data = pd.read_csv(f'/Users/yangren/Downloads/embs_w_index/{model3_name}_128_complete_testing_embeddings_fold_{fold + 1}_exp_{exp}.csv')
    
merged_training_23 = pd.merge(model2_training_data[['FileName', 'True Label', 'Predicted Label']], model3_training_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))
merged_testing_23 = pd.merge(model2_testing_data[['FileName', 'True Label', 'Predicted Label']], model3_testing_data[['FileName', 'Predicted Label']], on='FileName', suffixes=('_model2', '_model3'))

merged_training_23['Classification'] = merged_training_23.apply(classify_labels23, axis=1)
merged_testing_23['Classification'] = merged_testing_23.apply(classify_labels23, axis=1)

different_training = merged_training_23[merged_training_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_training = merged_training_23[merged_training_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

different_testing = merged_testing_23[merged_testing_23['Classification'] == 'different'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]
same_testing = merged_testing_23[merged_testing_23['Classification'] == 'same'][['FileName', 'True Label', 'Predicted Label_model2', 'Predicted Label_model3']]

model2_training_filtered = model2_training_data[model2_training_data['FileName'].isin(different_training['FileName'])]
model3_training_filtered = model3_training_data[model3_training_data['FileName'].isin(different_training['FileName'])]

model2_testing_filtered = model2_testing_data[model2_testing_data['FileName'].isin(different_testing['FileName'])]
model3_testing_filtered = model3_testing_data[model3_testing_data['FileName'].isin(different_testing['FileName'])]

embeddings = ['Embedding_' + str(i) for i in range(0, 768)]

combined_training_df = model2_training_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_training_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))

combined_testing_df = model2_testing_filtered[['FileName'] + ['True Label'] + embeddings].merge(model3_testing_filtered[['FileName'] + embeddings], on='FileName', suffixes=('_model2', '_model3'))
    
embeddings = [f'Embedding_{i}_model{j}' for j in [2, 3] for i in range(768)]
X_train = combined_training_df[embeddings].values
y_train = combined_training_df['True Label'].values

X_test = combined_testing_df[embeddings].values
y_test = combined_testing_df['True Label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
    
X_train_reshaped = X_train.reshape(-1, 1, 2, 768)
X_test_reshaped = X_test.reshape(-1, 1, 2, 768)
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
train_dataset = EmbeddingDataset(X_train_reshaped, y_train)
test_dataset = EmbeddingDataset(X_test_reshaped, y_test)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
    
num_epochs = 60
for epoch in range(num_epochs):
    for embeddings, labels in train_loader:
        # Forward pass
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_report = classification_report(y_true, y_pred)

print(f"Fold {fold+1} 2DCNN Diff Set Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print(f"Fold {fold+1} 2DCNN Diff Set Classification Report:\n", class_report)
    
y_true_same = same_testing['True Label']
y_pred_same = same_testing['Predicted Label_model2']

conf_matrix_same = confusion_matrix(y_true_same, y_pred_same)

combined_conf_matrix = conf_matrix + conf_matrix_same

y_true = np.array([])
y_pred = np.array([])
    
for true_class in range(4):
    for pred_class in range(4):
        count = combined_conf_matrix[true_class, pred_class]
        y_true = np.append(y_true, [true_class] * count)
        y_pred = np.append(y_pred, [pred_class] * count)

report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
print(f"Fold {fold+1} 2DCNN Full Set Classification Report:\n", report)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class CNN2DClassifier11(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2DClassifier11, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)  # Batch normalization
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(2, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)  # Batch normalization

        # Updated the linear layer input size
        self.fc1 = nn.Linear(32 * 5 * 96, 128)  # Increased size
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)  # Increased dropout

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# In[ ]:





# In[ ]:





# In[154]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




