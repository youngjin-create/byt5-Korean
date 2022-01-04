import torch
from torch.utils.data import Dataset
import tensorflow.compat.v2 as tf
from tokenizer import ByT5KoreanTokenizer
tokenizer = ByT5KoreanTokenizer()
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

from t5.preprocessors import random_spans_helper, random_spans_noise_mask
from preprocessors import noise_span_to_unique_sentinel, nonnoise_span_to_unique_sentinel # customized

import json

# parameters
inputs_length = 1024
noise_density = 0.15
mean_noise_span_length = 20
extra_tokens_per_span_inputs = 1
extra_tokens_per_span_targets = 1

def random_span_masking(ids, noise_density, seeds, sentinel_id, mean_noise_span_length):
    noise_mask = random_spans_noise_mask(tf.size(ids), noise_density, seeds, mean_noise_span_length)
    input_ids = noise_span_to_unique_sentinel(ids, noise_mask, sentinel_id)
    labels = nonnoise_span_to_unique_sentinel(ids, noise_mask, sentinel_id)
    return input_ids, labels

def add_eos(ids, eos_id=1):
    return tf.pad(ids, paddings=tf.constant([[0, 1]]), constant_values=tf.constant(eos_id))
    
tokens_length, targets_length = random_spans_helper(inputs_length, noise_density, mean_noise_span_length, extra_tokens_per_span_inputs, extra_tokens_per_span_targets)
# ids = tokenizer('안녕하세요.', padding=True, truncation=True, max_length=tokens_length, add_special_tokens=False).input_ids
# input_ids, labels = random_span_masking(tf.constant(ids), noise_density, [(2, 3), (4, 5)], 259, mean_noise_span_length)

# dataset
c4_ko = []
with open('/data/shared/c4/c4/multilingual/c4-ko.tfrecord-00000-of-01024.json') as f:
    for line in f:
        c4_ko.append(json.loads(line))
        # c4_ko.append(tokenizer(json.loads(line)['text'], add_special_tokens=False).input_ids)

# from Korpora import Korpora
# corpus = Korpora.load("kowikitext")
# ids = tokenizer(corpus.train.texts, padding=True, truncation=True, max_length=512).input_ids
# ids_train = ids[:int(len(ids)*0.8)]
# ids_eval = ids[int(len(ids)*0.8):]

class KoreanDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        # self.examples = ids_train if not evaluate else ids_eval
        return

    def __len__(self):
        return len(c4_ko)

    def __getitem__(self, i):
        ids = tokenizer(c4_ko[i]['text'], padding=True, truncation=True, max_length=tokens_length, add_special_tokens=False).input_ids
        input_ids, labels = random_span_masking(tf.constant(ids), noise_density, [(2, 3), (4, 5)], 259, mean_noise_span_length)

        return { 'input_ids': torch.tensor(input_ids.numpy()), 'labels': torch.tensor(labels.numpy()) }
        # return { 'input_ids': torch.tensor(self.examples[i]), 'decoder_input_ids': torch.tensor(self.examples[i]), 'label_ids': torch.tensor(self.examples[i]) }
        # return { 'input_ids': torch.tensor(self.examples[i]), 'label_ids': torch.tensor([0]) }

if __name__ == "__main__":
    ds = KoreanDataset()
    print(ds[0])
    print('Done.')
