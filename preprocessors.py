import seqio
import tensorflow.compat.v2 as tf

def random_spans_noise_mask(length,
                            noise_density,
                            seeds,
                            mean_noise_span_length=3.0):
    """Noise mask consisting of random spans of noise tokens.

    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:

        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(
        num_noise_tokens / mean_noise_span_length)

    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.

    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        seeds: an int32 Tensor, shaped (2, 2)
        mean_noise_span_length: a number

    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length
    # increase length to avoid degeneracy
    length = tf.maximum(length, 2)
    def to_int(x):
        return tf.cast(x, tf.int32)
    def to_float(x):
        return tf.cast(x, tf.float32)
    num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
    num_noise_spans = to_int(
        tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = tf.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens
    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments, seed):
        """Partition a sequence of items randomly into non-empty segments.

        Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]
        seed: an integer seed
        Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
        """
        first_in_segment = tf.pad(
            seqio.stateless_shuffle(
                to_int(tf.range(num_items - 1) < num_segments - 1),
                seed),
            [[1, 0]])
        segment_id = tf.cumsum(first_in_segment)
        segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
        return segment_length
    noise_span_lengths = _random_segmentation(
        num_noise_tokens, num_noise_spans, seeds[0])
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans, seeds[1])
    interleaved_span_lengths = tf.reshape(
        tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2])
    span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = tf.math.unsorted_segment_sum(
        tf.ones_like(span_starts), span_starts, length)
    span_num = tf.cumsum(span_start_indicator)
    is_noise = tf.equal(span_num % 2, 1)
    return is_noise[:orig_length]

def noise_span_to_unique_sentinel(tokens, noise_mask, sentinel_id):
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

    sentinel = sentinel_id - 1 + tf.cumsum(
        tf.cast(first_noise_tokens, tokens.dtype))

    tokens = tf.where(first_noise_tokens, sentinel, tokens)
    return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))

def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary):
    return noise_span_to_unique_sentinel(
        tokens, tf.logical_not(noise_mask), vocabulary)

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
    expected_output = [259, 12, 13, 260, 15]
    input_ids = noise_span_to_unique_sentinel(
        tokens, noise_mask, 259)
    labels = nonnoise_span_to_unique_sentinel(
        tokens, noise_mask, 259)
    # self.assertAllEqual(output, expected_output)
    return input_ids, labels, expected_output

if __name__ == "__main__":
    test_random_spans_noise_mask()
    test_noise_span_to_unique_sentinel()