import torch
from torch.utils.data import Dataset
import tensorflow.compat.v2 as tf

# prevent tensorflow from pre-allocating memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tokenizer import ByT5KoreanTokenizer
tokenizer = ByT5KoreanTokenizer()
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

from t5.preprocessors import random_spans_helper, random_spans_noise_mask
from preprocessors import noise_span_to_unique_sentinel, nonnoise_span_to_unique_sentinel # customized

import glob
import gzip
import json
from tqdm import tqdm

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
# c4_ko = []
c4_ko_train = []
c4_ko_eval = []

# with open('/data/shared/c4/c4/multilingual/c4-ko.tfrecord-00000-of-01024.json') as f:
#     for line in f:
#         c4_ko.append(json.loads(line))
#         # c4_ko.append(tokenizer(json.loads(line)['text'], add_special_tokens=False).input_ids)

n_texts = 0
for filename in tqdm(glob.glob("/data/shared/c4/c4/multilingual/*ko*.gz")[:500]):
    with gzip.open(filename) as f:
        for line in f:
            n_texts += 1
            if n_texts % 100 != 0 or len(c4_ko_eval) >= 1000:
                c4_ko_train.append(json.loads(line))
            else:
                c4_ko_eval.append(json.loads(line))
# from Korpora import Korpora
# corpus = Korpora.load("kowikitext")
# ids = tokenizer(corpus.train.texts, padding=True, truncation=True, max_length=512).input_ids
# ids_train = ids[:int(len(ids)*0.8)]
# ids_eval = ids[int(len(ids)*0.8):]

class KoreanDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        self.examples = c4_ko_train if not evaluate else c4_ko_eval
        return

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids = tokenizer(self.examples[i]['text'], padding=True, truncation=True, max_length=tokens_length, add_special_tokens=False).input_ids
        input_ids, labels = random_span_masking(tf.constant(ids), noise_density, [(i, i), (i, i)], 259, mean_noise_span_length)
        input_ids = add_eos(input_ids)
        labels = add_eos(labels)

        return { 'input_ids': torch.tensor(input_ids.numpy()), 'labels': torch.tensor(labels.numpy()) }
        # return { 'input_ids': torch.tensor(self.examples[i]), 'decoder_input_ids': torch.tensor(self.examples[i]), 'label_ids': torch.tensor(self.examples[i]) }
        # return { 'input_ids': torch.tensor(self.examples[i]), 'label_ids': torch.tensor([0]) }

if __name__ == "__main__":
    ds_train = KoreanDataset(evaluate=False)
    ds_eval = KoreanDataset(evaluate=True)
    print(ds_train[0])
    print(ds_eval[0])
    print('Done.')
