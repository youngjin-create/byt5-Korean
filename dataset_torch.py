
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

# from multilingual_t5 import preprocessors
# from multilingual_t5 import utils
# from multilingual_t5.evaluation import metrics as mt5_metrics
# from multilingual_t5.tasks import DEFAULT_OUTPUT_FEATURES as DEFAULT_MT5_OUTPUT_FEATURES

import seqio
import t5.data
import t5.data.tasks
import tensorflow_datasets as tfds

import preprocessors

MEAN_NOISE_SPAN_LENGTH = 20
DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=t5.data.ByteVocabulary()),
    "targets": t5.data.Feature(vocabulary=t5.data.ByteVocabulary())
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
    ds = MyIterableDataset(mixture_or_task_name='byt5_extra.ko')
    loader = DataLoader(ds, batch_size=8)
    for i, batch in enumerate(loader):
        print(batch)
        if i == 5:
            break
    for i in range(10):
        print(next(iter(ds)))
    print('Done.')
