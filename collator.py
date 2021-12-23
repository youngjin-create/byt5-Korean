from typing import Dict

import torch
# import t5.preprocessors

class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, decoder_start_token_id, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data



def random_spans_helper(inputs_length,
                        noise_density,
                        mean_noise_span_length,
                        extra_tokens_per_span_inputs,
                        extra_tokens_per_span_targets,
                        verbose=False):
  """Training parameters to avoid padding with random_spans_noise_mask.

  When training a model with random_spans_noise_mask, we would like to set the
  other training hyperparmeters in a way that avoids padding.  This function
  helps us compute these hyperparameters.

  We assume that each noise span in the input is replaced by
  extra_tokens_per_span_inputs sentinel tokens, and each non-noise span in the
  targets is replaced by extra_tokens_per_span_targets sentinel tokens.

  This function tells us the required number of tokens in the raw example (for
  split_tokens()) as well as the length of the encoded targets.

  Note that this function assumes the inputs and targets will have EOS appended
  and includes that in the reported length.

  Args:
    inputs_length: an integer - desired length of the tokenized inputs sequence
    noise_density: a float
    mean_noise_span_length: a float
    extra_tokens_per_span_inputs: an integer
    extra_tokens_per_span_targets: an integer
    verbose: a bool indicating whether to log sequence lengths
  Returns:
    tokens_length: length of original text in tokens
    targets_length: an integer - length in tokens of encoded targets sequence
  """
  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    return (
        num_nonnoise_tokens +
        num_noise_spans * extra_tokens_per_span_inputs + 1,
        num_noise_tokens +
        num_noise_spans * extra_tokens_per_span_targets + 1)

  tokens_length = inputs_length
  while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
         <= inputs_length):
    tokens_length += 1
  inputs_length, targets_length = (
      _tokens_length_to_inputs_length_targets_length(tokens_length))
  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  if verbose:
    logging.info(
        'tokens_length=%s inputs_length=%s targets_length=%s '
        'noise_density=%s mean_noise_span_length=%s ',
        tokens_length, inputs_length, targets_length,
        noise_density, mean_noise_span_length)
  return tokens_length, targets_length

# TODO(adarob): Add a test.
def span_corruption(dataset,
                    sequence_length,
                    output_features,
                    mean_noise_span_length=3.0,
                    noise_density=0.15,
                    input_feature_key='inputs',
                    merge_examples_to_reduce_padding=True):
  """Final pretraining objective used in Raffel et al., 2019."""
  input_length, targets_length = random_spans_helper(
      extra_tokens_per_span_inputs=1,
      extra_tokens_per_span_targets=1,
      inputs_length=sequence_length[input_feature_key],
      mean_noise_span_length=mean_noise_span_length,
      noise_density=noise_density)

  if sequence_length['targets'] < targets_length:
    raise ValueError(
        f'Expected targets length for span corruption ({targets_length}) is '
        f'greater than configured targets length '
        f"({sequence_length['targets']})")

  ds = dataset
  ds = select_random_chunk(
      ds,
      output_features=output_features,
      feature_key='targets',
      max_length=65536)
  if merge_examples_to_reduce_padding:
    ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  ds = split_tokens(
      ds,
      feature_key='targets',
      min_tokens_per_segment=None,
      max_tokens_per_segment=input_length)
  ds = denoise(
      ds,
      output_features,
      inputs_fn=noise_span_to_unique_sentinel,
      targets_fn=nonnoise_span_to_unique_sentinel,
      noise_density=noise_density,
      noise_mask_fn=functools.partial(
          random_spans_noise_mask,
          mean_noise_span_length=mean_noise_span_length),
      input_feature_key=input_feature_key)
  return ds

if __name__ == "__main__":
    r = random_spans_helper(inputs_length=512, noise_density=0.15, mean_noise_span_length=20, extra_tokens_per_span_inputs=1, extra_tokens_per_span_targets=1, verbose=False)
    print(r)

