import pandas as pd
from tqdm import tqdm
from sympy import sympify
from hanspell import spell_checker
import ast
import json
import re

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
#from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
# from transformers  import MT5Model, T5Tokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


train_df = pd.read_csv("train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("eval.tsv", sep="\t").astype(str)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])


# prefix = '~/Git/T5_test/data/binary_classification/'
# Read data
# data = pd.read_csv(prefix + "train.csv", header=None)
# data = data.rename(columns={0:'sentiment', 1:'review'})
# #data = data.iloc[:2000]

# Define pretrained tokenizer and model
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

#tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-large')
tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-large')
# model = MT5Model.from_pretrained('google/mt5-small') #, num_labels=2)

X_train = train_df.input_text.tolist()
y_train = train_df.target_text.tolist()
X_val = eval_df.input_text.tolist()
y_val = eval_df.target_text.tolist()

# ----- 1. Preprocess data -----#
# Preprocess data
# X = list(data["review"])
# y = list(data["sentiment"])
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
# y_train = tokenizer([str(x) for x in y_train]).input_ids
# y_val = tokenizer([str(x) for x in y_val]).input_ids
y_train_tokenized = tokenizer(y_train, padding=True, truncation=True, max_length=128).input_ids
y_val_tokenized = tokenizer(y_val, padding=True, truncation=True, max_length=128).input_ids


train_dataset = Dataset(X_train_tokenized, y_train_tokenized)
val_dataset = Dataset(X_val_tokenized, y_val_tokenized)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
# def compute_metrics(p):
#     pred, labels = p
#     pred = np.argmax(pred, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     recall = recall_score(y_true=labels, y_pred=pred)
#     precision = precision_score(y_true=labels, y_pred=pred)
#     f1 = f1_score(y_true=labels, y_pred=pred)
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer

import wandb
ndb.init(project="ape210k", 
           name="mt5")
           #tags=["baseline", "high-lr"],
           #group="bert"

args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir="true",
    evaluation_strategy="epoch",
    #eval_steps=20000,
    save_strategy="epoch",
    dataloader_num_workers=4,
    #save_steps=100,
    learning_rate=0.0001,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=40,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    seed=0,
    load_best_model_at_end=True,
    report_to="wandb",
)

# args_dict = {
#   "output_dir": './models/tpu',
#   "per_gpu_eval_batch_size": 8,
#   "num_cores": 8,
#   'training_script': 'train_t5_squad.py',
#   "model_name_or_path": 't5-base',
#   "max_len": 512 ,
#   "target_max_len": 16,
#   "overwrite_output_dir": True,
#   "per_gpu_train_batch_size": 8,
#   "gradient_accumulation_steps": 4,
#   "learning_rate": 1e-4,
#   "tpu_num_cores": 8,
#   "num_train_epochs": 4,
#   "do_train": True
# }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    #compute_metrics=compute_metrics,
    #prediction_loss_only=True
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()

# ----- 3. Predict -----#

# # Load test data
#test_data = pd.read_csv(prefix + "test.csv", header=None)
#test_data = test_data.rename(columns={0:'sentiment', 1:'review'})
test_df = pd.read_csv("train.tsv", sep="\t").astype(str)
# X_test = list(test_data["review"])[:100]
# y_test = list(test_data["sentiment"])[:100]
# X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
# y_test_tokenized = tokenizer([str(x) for x in y_test]).input_ids
#

X_test = test_df.input_text.tolist()
y_test = test_df.target_text.tolist()

X_test_tokenized = tokenizer(X_test[:100], padding=True, truncation=True, max_length=512)
y_test_tokenized = tokenizer(y_test[:100], padding=True, truncation=True, max_length=128).input_ids


# # # Create torch dataset
test_dataset = Dataset(X_test_tokenized, y_test_tokenized)

# #
# # Load trained model
model_path = "output/checkpoint-2438"
# #model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
#
#
# # Define test trainer
args = TrainingArguments(
    output_dir="output",
    #overwrite_output_dir="true",
    #evaluation_strategy="steps",
    #eval_steps=100,
    #save_strategy="epoch",
    #save_steps=100,
    #per_device_train_batch_size=32,
    #gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    seed=0,
    #load_best_model_at_end=True,
)
test_trainer = Trainer(model, args=args)
#
# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)
#
# Preprocess raw predictions
#y_pred = np.argmax(raw_pred, axis=3)
y_pred= torch.from_numpy(raw_pred[0]).max(dim=2)[1]

# X_test_tokenized_ids = tokenizer(X_test, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids
# model = model.to("cuda:0")
# y_pred = model.generate(X_test_tokenized_ids.to("cuda:0"))

tokenizer.batch_decode(y_pred)


