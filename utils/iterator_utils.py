import tensorflow as tf
import collections


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_in", "target_out", "source_length", "target_length"))):
    pass


class InferenceInput(
    collections.namedtuple("InferenceInput",
                           ("initializer", "source", "source_length"))):
    pass


class EvalInput(
    collections.namedtuple("EvalInput",
                           ("initializer", "source", "target_out", "source_text", "source_length", "target_length"))):
    pass


def create_dataset(hparams, src_dataset, tgt_dataset, src_vocab, tgt_vocab, output_buffer_size, num_parallel_calls=4):
    src_dataset = src_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    tgt_dataset = tgt_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_dataset = src_dataset.map(lambda x: tf.cast(src_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    tgt_dataset = tgt_dataset.map(lambda x: tf.cast(tgt_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if hparams.src_max_len > 0:
        src_dataset = src_dataset.map(lambda x: x[:hparams.src_max_len])
    if hparams.tgt_max_len > 0:
        tgt_dataset = tgt_dataset.map(lambda x: x[:hparams.tgt_max_len])

    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    return dataset


def get_iterator(hparams, datasets, max_rows=0, num_parallel_calls=4):
    output_buffer_size = hparams.batch_size * 1000
    src_vocab, tgt_vocab, src_dataset, tgt_dataset, _, _ = datasets
    if max_rows > 0:
        src_dataset = src_dataset.take(max_rows)
        tgt_dataset = tgt_dataset.take(max_rows)
    src_dataset = src_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    tgt_dataset = tgt_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    sos_id = tf.cast(tgt_vocab.lookup(tf.constant(hparams.sos)), tf.int32)
    eos_id = tf.cast(tgt_vocab.lookup(tf.constant(hparams.eos)), tf.int32)
    pad_id = tf.cast(tgt_vocab.lookup(tf.constant(hparams.pad)), tf.int32)
    src_dataset = src_dataset.map(lambda x: tf.cast(src_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    tgt_dataset = tgt_dataset.map(lambda x: tf.cast(tgt_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if hparams.src_max_len > 0:
        src_dataset = src_dataset.map(lambda x: x[:hparams.src_max_len])
    if hparams.tgt_max_len > 0:
        tgt_dataset = tgt_dataset.map(lambda x: x[:hparams.tgt_max_len])

    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(lambda src, tgt: (src,
                                            tf.concat(([sos_id], tgt), axis=0),
                                            tf.concat((tgt, [eos_id]), axis=0),
                                            tf.size(src),
                                            tf.size(tgt) + 1),
                          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def batching_func(x):
        if hparams.src_max_len > 0:
            return x.padded_batch(hparams.batch_size,
                                  padded_shapes=(tf.TensorShape([hparams.src_max_len]),
                                                 tf.TensorShape([hparams.tgt_max_len]),
                                                 tf.TensorShape([hparams.tgt_max_len]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([])),
                                  padding_values=(pad_id, pad_id, pad_id, 0, 0))
        else:
            return x.padded_batch(hparams.batch_size,
                                  padded_shapes=(tf.TensorShape([None]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([])),
                                  padding_values=(pad_id, pad_id, pad_id, 0, 0))

    num_buckets = hparams.bucket
    if num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            bucket_width = (hparams.infer_src_max_len + num_buckets - 1) // num_buckets

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=hparams.batch_size))
    else:
        batched_dataset = batching_func(dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    src_ids, tgt_in_ids, tgt_out_ids, src_length, tgt_length = batched_iter.get_next()
    batched_input = BatchedInput(initializer=batched_iter.initializer,
                                 source=src_ids,
                                 target_in=tgt_in_ids,
                                 target_out=tgt_out_ids,
                                 source_length=src_length,
                                 target_length=tgt_length)

    return batched_input


def get_inference_iterator(hparams, datasets, num_parallel_calls=4):
    output_buffer_size = hparams.batch_size * 1000
    src_vocab, tgt_vocab, src_dataset, _, _, _ = datasets
    src_dataset = src_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_dataset = src_dataset.map(lambda x: tf.cast(src_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_dataset = src_dataset.map(lambda x: x[:hparams.infer_src_max_len],
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # eos_id = tf.cast(src_vocab.lookup(tf.constant(hparams.eos)), tf.int32)
    pad_id = tf.cast(src_vocab.lookup(tf.constant(hparams.pad)), tf.int32)
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_dataset = src_dataset.shuffle(buffer_size=1000)
    src_dataset = src_dataset.padded_batch(hparams.batch_size,
                                           padded_shapes=(tf.TensorShape([hparams.infer_src_max_len]),
                                                          tf.TensorShape([])),
                                           padding_values=(pad_id, 0))
    batched_iter = src_dataset.make_initializable_iterator()
    src_ids, src_length = batched_iter.get_next()
    batched_input = InferenceInput(initializer=batched_iter.initializer,
                                   source=src_ids,
                                   source_length=src_length)

    return batched_input


def get_eval_iterator(hparams, datasets, eos, num_parallel_calls=4, shuffle=True):
    output_buffer_size = hparams.batch_size * 1000
    src_vocab, tgt_vocab, src_dataset, tgt_dataset, tgt_reverse_vocab, src_vocab_size, tgt_vocab_size = datasets
    src_text_dataset = src_dataset.map(lambda x: tf.string_split([x]).values,
                                       num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_dataset = src_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    tgt_dataset = tgt_dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    eos_id = tf.cast(tgt_vocab.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda x: tf.cast(src_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    tgt_dataset = tgt_dataset.map(lambda x: tf.cast(tgt_vocab.lookup(x), tf.int32),
                                  num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    src_dataset = src_dataset.map(lambda x: x[:hparams.infer_src_max_len])
    tgt_dataset = tgt_dataset.map(lambda x: x[:hparams.infer_tgt_max_len])

    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, src_text_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(lambda src, tgt, src_text: (src,
                                                      tf.concat((tgt, [eos_id]), axis=0),
                                                      src_text,
                                                      tf.size(src),
                                                      tf.size(tgt)
                                                      ),
                          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    def batching_func(x):
        return x.padded_batch(hparams.batch_size,
                              padded_shapes=(tf.TensorShape([None]),
                                             tf.TensorShape([None]),
                                             tf.TensorShape([None]),
                                             tf.TensorShape([]),
                                             tf.TensorShape([])
                                             ),
                              padding_values=(eos_id, eos_id, '', 0, 0))

    batched_dataset = batching_func(dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    src_ids, tgt_out_ids, src_texts, src_length, tgt_length = batched_iter.get_next()
    batched_input = EvalInput(initializer=batched_iter.initializer,
                              source=src_ids,
                              target_out=tgt_out_ids,
                              source_text=src_texts,
                              source_length=src_length,
                              target_length=tgt_length)

    return batched_input
