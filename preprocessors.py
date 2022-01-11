import tensorflow.compat.v2 as tf
from t5.preprocessors import random_spans_helper, random_spans_noise_mask

import gin
gin.parse_config_file('config.gin')

def noise_span_to_unique_sentinel(tokens, noise_mask, sentinel_id, extra_ids_increment=1):
    """Replace each run of consecutive noise tokens with a different sentinel.

    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.

    We want to generate training examples like
    "We hold X to be Y that" -> "X these truths Y self evident Z"

    Sentinels assigned in decreasing order within the sequence starting at
    vocabulary.size - 1.  That is, we appropriate the last tokens in the
    vocabulary for additional use as sentinels.

    TODO(noam): we may want to try enlarging the vocabulary and leaving room
    for the sentinels instead.  However, this requires enlarging the embedding
    tables in the model, so that is a bigger change.

    Args:
        tokens: a 1d integer Tensor
        noise_mask: a boolean Tensor with the same shape as tokens
        vocabulary: a vocabulary.Vocabulary
        seeds: an unused int32 Tensor
    Returns:
        a Tensor with the same shape and dtype as tokens
    """
    prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = tf.logical_and(
        noise_mask, tf.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

    sentinel = sentinel_id + extra_ids_increment * (-1 + tf.cumsum(
        tf.cast(first_noise_tokens, tokens.dtype)))

    tokens = tf.where(first_noise_tokens, sentinel, tokens)
    return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))

def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary, extra_ids_increment=1):
    return noise_span_to_unique_sentinel(
        tokens, tf.logical_not(noise_mask), vocabulary, extra_ids_increment)

def random_span_masking(ids,
                        noise_density,
                        seeds,
                        sentinel_id,
                        mean_noise_span_length):
    noise_mask = random_spans_noise_mask(
        tf.size(ids), noise_density, seeds, mean_noise_span_length)
    input_ids = noise_span_to_unique_sentinel(
        ids, noise_mask, sentinel_id)
    labels = nonnoise_span_to_unique_sentinel(
        ids, noise_mask, sentinel_id)
    # self.assertAllEqual(output, expected_output)
    return input_ids, labels

def test_random_spans_noise_mask():
    length = 32
    noise_density = 0.25
    mean_noise_span_length = 2.0
    # there should be 4 noise spans with a total length of 8.
    noise_mask = random_spans_noise_mask(
        length, noise_density, [(1, 2), (3, 4)], mean_noise_span_length)
    output = tf.cast(noise_mask, tf.int32)
    expected_output = [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    # self.assertAllEqual(output, expected_output)
    return output, expected_output

def test_noise_span_to_unique_sentinel():
    # vocabulary = test_utils.MockVocabulary({'foo': [10]}, vocab_size=1000)
    tokens = tf.constant([9, 10, 11, 12, 13, 14, 15])
    noise_mask = tf.constant([False, True, True, False, False, True, False])
    input_ids = noise_span_to_unique_sentinel(
        tokens, noise_mask, 259)
    labels = nonnoise_span_to_unique_sentinel(
        tokens, noise_mask, 259)
    # self.assertAllEqual(output, expected_output)
    return input_ids, labels

if __name__ == "__main__":
    tokens_length, targets_length = random_spans_helper()
    print(tokens_length) # 1193
    print(targets_length) # 189
    test_random_spans_noise_mask()
    test_noise_span_to_unique_sentinel()