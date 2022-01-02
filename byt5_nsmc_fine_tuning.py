import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from transformers import AutoTokenizer

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    #self.save_hyperparameters()
    self.myparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def is_logger(self):
    return True # self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )

  def _step(self, batch):
    labels = batch["target_ids"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        labels=labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    # {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    return None

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.myparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.myparams.learning_rate, eps=self.myparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  # handled automatically by PyTorch Lightning
  # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
  #   if self.trainer.use_tpu:
  #     xm.optimizer_step(optimizer)
  #   else:
  #     optimizer.step()
  #   optimizer.zero_grad()
  #   self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.myparams)
    dataloader = DataLoader(train_dataset, batch_size=self.myparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.myparams.train_batch_size * max(1, self.myparams.n_gpu)))
        // self.myparams.gradient_accumulation_steps
        * float(self.myparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.myparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.myparams)
    return DataLoader(val_dataset, batch_size=self.myparams.eval_batch_size, num_workers=4)

args_dict = dict(
    output_dir="", # path to save the checkpoints
    model_name_or_path='google/byt5-small',
    tokenizer_name_or_path='google/byt5-small',
    max_seq_length=128,
    #max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=12, # max_len_128 : ok
    #train_batch_size=8, # max_len_512 : ok
    #train_batch_size=4,
    #train_batch_size=2,
    #train_batch_size=1,
    eval_batch_size=12, # max_len_128 : ok
    #eval_batch_size=8, # max_len_512 : ok
    #eval_batch_size=4,
    #eval_batch_size=2,
    #eval_batch_size=1,
    gradient_accumulation_steps=16,
    #gradient_accumulation_steps=2,
    #gradient_accumulation_steps=1,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,
    #opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    opt_level='O0',
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

ids_neg = tokenizer.encode('negative </s>')
ids_pos = tokenizer.encode('positive </s>')
len(ids_neg), len(ids_pos)

from torch.utils.data import Dataset, DataLoader
from random import random
from datasets import load_dataset

# NSMC 데이터 처리
from Korpora import Korpora
Korpora.fetch("nsmc")

corpus = Korpora.load("nsmc")

df_train_text = pd.DataFrame(corpus.train.texts, columns=['text'])
df_train_labels = pd.DataFrame(corpus.train.labels, columns=['labels'])

df_train = pd.concat([df_train_text, df_train_labels], axis=1)

df_test_text = pd.DataFrame(corpus.test.texts, columns=['text'])
df_test_labels = pd.DataFrame(corpus.test.labels, columns=['labels'])

df_test = pd.concat([df_test_text, df_test_labels], axis=1)

class NSMCDataset(Dataset):
  def __init__(self, tokenizer, df, text_labels, max_len=512):
    self.df = df
    self.text_labels = text_labels
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for idx, row in self.df.iterrows():
      text = row['text']
      line = text.strip()
      line = REPLACE_NO_SPACE.sub("", line) 
      line = REPLACE_WITH_SPACE.sub("", line)
      line = line

      ##### ALERT #####
      ### randomly filters to 1/4 size of dataset
      ### so we can do prototyping
      #################
      '''
      if random() > 0.25:
        continue
      '''

      target = self.text_labels[row['labels']]

      # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line], max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt"
      )
      # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=10, padding='max_length', truncation=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)

dataset = NSMCDataset(tokenizer, df_train, ['negative</s>', 'positive</s>'], max_len=512)

args_dict.update({#'output_dir': 't5_nsmc_sentiment', 
                  'output_dir': 't5_base_nsmc_sentiment_maxlen128',
                  #'num_train_epochs': 1, 
                  'num_train_epochs': 5, 
                  'vocab_file': 'tokenizer_config.json'})
args = argparse.Namespace(**args_dict)

# add this in?:
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# callbacks=[EarlyStopping(monitor='val_loss')]

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    #early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    #amp_level=args.opt_level,
    #amp_level=None,
    gradient_clip_val=args.max_grad_norm#,
    #max_steps=100  # error 방지
)

def get_dataset(tokenizer, df, args):
  return NSMCDataset(tokenizer, df, ['negative</s>', 'positive</s>'], max_len=512)

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    #self.save_hyperparameters()
    self.myparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def is_logger(self):
    return True # self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )

  def _step(self, batch):
    labels = batch["target_ids"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        labels=labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    # {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    return None

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.myparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.myparams.learning_rate, eps=self.myparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  # handled automatically by PyTorch Lightning
  # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
  #   if self.trainer.use_tpu:
  #     xm.optimizer_step(optimizer)
  #   else:
  #     optimizer.step()
  #   optimizer.zero_grad()
  #   self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, df=df_train, args=self.myparams)
    dataloader = DataLoader(train_dataset, batch_size=self.myparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.myparams.train_batch_size * max(1, self.myparams.n_gpu)))
        // self.myparams.gradient_accumulation_steps
        * float(self.myparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.myparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, df=df_test, args=self.myparams)
    return DataLoader(val_dataset, batch_size=self.myparams.eval_batch_size, num_workers=4)

model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

## save the model this way so next time you can load it using T5ForConditionalGeneration.from_pretrained
#model.model.save_pretrained('t5_base_nsmc_sentiment')
model.model.save_pretrained('t5_base_nsmc_sentiment_maxlen128')
#model.model.eval()
