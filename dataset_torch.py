
# The following code includes modification from byt5, see LICENSE.

# we are using tensorflow just for preprocessing (using codes from google/t5)
# so force to use cpus
import os
cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # prevent tensorflow from pre-allocating memory
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

# import tensorflow.compat.v1 as tf
# tf.config.set_visible_devices([], 'GPU') # disable GPU for tensorflow


"""Add ByT5 Tasks to registry."""
import functools

from typing import Optional

# from multilingual_t5 import preprocessors
# from multilingual_t5 import utils
# from multilingual_t5.evaluation import metrics as mt5_metrics
# from multilingual_t5.tasks import DEFAULT_OUTPUT_FEATURES as DEFAULT_MT5_OUTPUT_FEATURES

import seqio
from seqio.vocabularies import Vocabulary
import t5.data
import t5.data.tasks
import tensorflow_datasets as tfds


import preprocessors

class ByteVocabularyKorean(Vocabulary):
  """Byte-level vocabulary.

  Encode/decode text directly to 256 "byte IDs" using UTF-8 encoding. Three
  special IDs are reserved (0=padding, 1=EOS, 2=UNK), so our encoded byte IDs
  are +3 greater than UTF-8 byte values.

  This is the vocabulary used by the ByT5 models:
  https://arxiv.org/abs/2105.13626
  """

  def __init__(self, extra_ids: int = 0):
    """Create a ByteVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.

    Args:
      extra_ids: an optional integer
    """
    self._byte_size = 256 + 19 + 21 + 28
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3
    super().__init__(extra_ids=extra_ids)

  @property
  def eos_id(self) -> Optional[int]:
    return 1

  @property
  def unk_id(self) -> Optional[int]:
    return 2

  def _convert_char_to_ids_Korean(self, c):
      o = ord(c)
      print(o)
      if 44032 <= o and o <= 55203: # 44032: 가, 55203: 힣 
          o -= 44032
          return [256 + (o // 588), 256 + 19 + ((o % 588) // 28), 256 + 19 + 21 + (o % 28)]
      return list(c.encode("utf-8"))

  def _convert_strings_to_ids(self, s):
    """Convert a python string to integers based on UTF-8 encoding.

    Args:
      s: a string
    Returns:
      ids: a list of integers
    """
    return sum([self._convert_char_to_ids_Korean(c) for c in s], [])
    # return list(s.encode("utf-8"))

  def _convert_ids_to_strings(self, ids):
    """Convert ids to a python string based on UTF-8 encoding.

    Args:
      ids: a list of integers
    Returns:
      s: a string
    """
    return bytes(ids).decode("utf-8", errors="ignore")

  def _filter_non_string_ids(self, ids):
    """Filter special token ids and extra ids if there are any.

    Args:
      ids: a list of integers
    Returns:
      ids: a list of integers
    """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    return [id for id in ids if lower_bound <= id < upper_bound]

  @property
  def _base_vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size

  def _encode(self, s):
    """Encode a python string as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    ids = self._convert_strings_to_ids(s)
    return [i + self._num_special_tokens for i in ids]

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    The special tokens of PAD, EOS, and UNK will not be represented in the
    output string. This is different from the SentencePieceVocabulary, where
    UNK will show up as a '?' character.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """

    ids = self._filter_non_string_ids(ids)
    ids = [i - self._num_special_tokens for i in ids]
    return self._convert_ids_to_strings(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    # s = tf.Print(s, [s], message="Encoding: ")
    tf_ids = tf.dtypes.cast(tf.io.decode_raw(s, tf.uint8), tf.int32)
    if len(tf_ids) < 3:
        return tf_ids
    l1 = tf.concat([tf_ids[1:], [0]], 0)
    l2 = tf.concat([tf_ids[2:], [0, 0]], 0)
    o = tf.where(tf.math.logical_and(224 <= tf_ids, tf_ids < 240), ((tf_ids % 16) * 2**12 + (l1 % 64) * 2**6 + (l2 % 64)), tf.zeros_like(tf_ids))
    mask = tf.math.logical_and(44032 <= o, o <= 55203)
    b1 = tf.where(mask, 256 + ((o - 44032) // 588), tf.zeros_like(o))
    b2 = tf.where(mask, 256 + 19 + (((o - 44032) % 588) // 28), tf.zeros_like(o))
    b3 = tf.where(mask, 256 + 19 + 21 + ((o - 44032) % 28), tf.zeros_like(o))
    b2 = tf.concat([[0], b2[0:-1]], 0)
    b3 = tf.concat([[0, 0], b3[0:-2]], 0)
    kor = b1 + b2 + b3
    tf_ids = tf.where(kor > 0, kor, tf_ids) + self._num_special_tokens
    # tf_ids = tf.Print(tf_ids, [len(tf_ids)], message="len: ")
    return tf_ids
    # tf_ids = tf.io.decode_raw(s, tf.uint8) + self._num_special_tokens
    # return tf.dtypes.cast(tf_ids, tf.int32)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    return tf.py_function(func=self.decode, inp=[ids], Tout=tf.string)

  def __eq__(self, other):
    if not isinstance(other, ByteVocabularyKorean):
      return False
    return (self.extra_ids == other.extra_ids and
            self.eos_id == other.eos_id and
            self.unk_id == other.unk_id)

MEAN_NOISE_SPAN_LENGTH = 20
DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

# DEFAULT_PREPROCESSORS = [
#     seqio.preprocessors.tokenize,
#     seqio.CacheDatasetPlaceholder(),
#     seqio.preprocessors.append_eos_after_trim,
# ]

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=t5.data.ByteVocabulary()),
    "targets": t5.data.Feature(vocabulary=t5.data.ByteVocabulary())
}

KOREAN_BYTE_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=ByteVocabularyKorean()),
    "targets": t5.data.Feature(vocabulary=ByteVocabularyKorean())
}

MC4_LANGS = tfds.text.c4.MC4_LANGUAGES
MC4_LANGS = ['ko'] # byt5-korean

# =========================== Pretraining Tasks/Mixtures =======================
# mC4
for lang in MC4_LANGS:
    seqio.TaskRegistry.add(
        "byt5_google.{}".format(lang.replace("-", "_")),
        source=seqio.TfdsDataSource(
            tfds_name="c4/multilingual:3.0.1",
            splits={
                "train": lang,
                "validation": f"{lang}-validation"
            }),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.rekey,
                key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            functools.partial(t5.data.preprocessors.span_corruption,
                                mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[])

# byt5_korean = ["byt5_korean.{}".format(lang.replace("-", "_")) for lang in MC4_LANGS]
# seqio.MixtureRegistry.add("byt5_korean", byt5_korean, default_rate=DEFAULT_MIX_RATE)

for lang in MC4_LANGS:
    seqio.TaskRegistry.add(
        "byt5_extra.{}".format(lang.replace("-", "_")),
        source=seqio.TfdsDataSource(
            tfds_name="c4/multilingual:3.0.1",
            splits={
                "train": lang,
                "validation": f"{lang}-validation"
            }),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.rekey,
                key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            functools.partial(preprocessors.span_corruption, mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH, sentinel_start=259, sentinel_inc=1),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[])

for lang in MC4_LANGS:
    seqio.TaskRegistry.add(
        "byt5_korean.{}".format(lang.replace("-", "_")),
        source=seqio.TfdsDataSource(
            tfds_name="c4/multilingual:3.0.1",
            splits={
                "train": lang,
                "validation": f"{lang}-validation"
            }),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.rekey,
                key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            functools.partial(preprocessors.span_corruption, mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH, sentinel_start=ByteVocabularyKorean()._base_vocab_size, sentinel_inc=1),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=KOREAN_BYTE_OUTPUT_FEATURES,
        metric_fns=[])

import mesh_tensorflow.transformer.dataset as transformer_dataset

def mesh_train_dataset_fn(
    mixture_or_task_name,
    sequence_length,
    vocabulary=None,
    dataset_split=tfds.Split.TRAIN,
    shuffle=True,
    seed=None,
    use_cached=False,
    pack=True):
  """Returns the tf.data.Dataset for training on a given mixture.

  This uses the format required for utils.run's `train_dataset_fn` argument in
  the Mesh TF transformer standalone.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.
    sequence_length: dict mapping feature key to the int length for that feature
      the max sequence length.
    vocabulary: unused argument, maintains compatibility with other dataset_fns.
    dataset_split: string, which split of the dataset to load. In most cases
      this should be "train".
    shuffle: Whether or not to shuffle dataset.
    seed: tf.int64 scalar tf.Tensor (or None). Used for both the global seed and
      shuffle seed for tf.data
    use_cached: bool, whether to load the cached version of this dataset.
    pack: bool, whether to pack the dataset.

  Returns:
    A tf.data.Dataset of preprocessed, tokenized, and batched examples.
  """
  del vocabulary
  mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)

  ds = mixture_or_task.get_dataset(
      sequence_length, split=dataset_split, use_cached=use_cached,
      shuffle=shuffle, num_epochs=None, seed=seed)

  # Select just the output features which are present in the dataset.
  feature_keys = tuple(k for k in mixture_or_task.output_features
                       if k in tf.data.get_output_shapes(ds))

  # Filtering feature keys is done in pack_or_pad function. However, when
  # packing is turned off, input_features aren't filtered leading to training
  # problems due to strings showing up in the input example. Filtering features
  # ensures that we don't rely on pack_or_pad to filter features for training.
  def _filter_features(ex):
    return {k: ex[k] for k in feature_keys}

  ds = ds.map(
      _filter_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  eos_keys = set(
      k for k, f in mixture_or_task.output_features.items() if f.add_eos)
  ds = transformer_dataset.pack_or_pad(
      ds, sequence_length, pack=pack,
      feature_keys=feature_keys, ensure_eos=eos_keys)
  return ds


# from tokenizer import ByT5KoreanTokenizer
# tokenizer = ByT5KoreanTokenizer()

import torch
from torch.utils.data import IterableDataset, DataLoader

class MyIterableDataset(IterableDataset):
    def __init__(self, mixture_or_task_name='byt5_google.ko', input_length=1024, target_length=189):
        print('\033[92m' + 'preparing dataset...' + '\033[0m')
        super(MyIterableDataset).__init__()
        t5.data.set_tfds_data_dir_override('/data/shared/tfds/')
        self.ds = mesh_train_dataset_fn(mixture_or_task_name, sequence_length={'inputs': input_length, 'targets': target_length}, shuffle=True, seed=1)
        self.ds_iter = tf.compat.v1.data.make_one_shot_iterator(self.ds)
        print('\033[92m' + 'dataset ready.' + '\033[0m')
        
    def __iter__(self):
        return self

    def __next__(self):
        example = next(self.ds_iter)
        return { 'input_ids': torch.tensor(example['inputs'].numpy()), 'labels': torch.tensor(example['targets'].numpy()) }


if __name__ == "__main__":
    # ds = MyIterableDataset(mixture_or_task_name='byt5_google.ko')
    # ds = MyIterableDataset(mixture_or_task_name='byt5_extra.ko')
    ds = MyIterableDataset(mixture_or_task_name='byt5_korean.ko')
    loader = DataLoader(ds, batch_size=8)
    for i, batch in enumerate(loader):
        print(batch)
        if i == 5:
            break
    for i in range(10):
        print(next(iter(ds)))
    print('Done.')
