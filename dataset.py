import torch
from torch.utils.data import Dataset
from tokenizer import ByT5KoreanTokenizer
tokenizer = ByT5KoreanTokenizer()
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

from Korpora import Korpora
corpus = Korpora.load("kowikitext")
ids = tokenizer(corpus.train.texts, padding=True, truncation=True, max_length=512).input_ids
ids_train = ids[:int(len(ids)*0.8)]
ids_eval = ids[int(len(ids)*0.8):]

class KoreanDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        self.examples = ids_train if not evaluate else ids_eval

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return { 'input_ids': torch.tensor(self.examples[i]), 'labels': torch.tensor(self.examples[i]) }
        # return { 'input_ids': torch.tensor(self.examples[i]), 'decoder_input_ids': torch.tensor(self.examples[i]), 'label_ids': torch.tensor(self.examples[i]) }
        # return { 'input_ids': torch.tensor(self.examples[i]), 'label_ids': torch.tensor([0]) }
