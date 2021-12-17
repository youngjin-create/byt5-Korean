import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/byt5-large')

from Korpora import Korpora
corpus = Korpora.load("kowikitext")
ids = tokenizer(corpus.train.texts).input_ids
ids_train = ids[:int(len(ids)*0.8)]
ids_eval = ids[int(len(ids)*0.8):]

class KoreanDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        self.examples = ids_train if not evaluate else ids_eval

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])
