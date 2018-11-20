import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper, helper, basic_decoder, decoder, beam_search_decoder
from tensorflow.python.ops import init_ops, variable_scope
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
import math
from models.seq_encoder import simple_embed_rnn, bidirectional_embed_rnn, single_rnn_cell


class AttentionModel(object):

    def __init__(self,
                 hparams,
                 mode,
                 src_vocab_table,
                 tgt_vocab_table,
                 reverse_target_vocab_table=None,
                 iterator=None,
                 dtype=tf.float32
                 ):
        self.num_encoder_symbols = hparams.vocab_size_source
        self.num_decoder_symbols = hparams.vocab_size_target
        self.num_units = hparams.num_units
        self.batch_size = tf.size(iterator.source_length)
        self.num_layers_encoder = hparams.num_layers_encoder
        self.num_layers_decoder = hparams.num_layers_decoder
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.dtype = dtype
        self.mode = mode
        self.hparams = hparams
        self.num_residual_layers = hparams.num_residual_layers
        self.iterator = iterator
        self.src_vocab_table = src_vocab_table
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.dropout = hparams.dropout
        else:
            self.beam_width = hparams.beam_width
            self.dropout = 0
            self.tgt_vocab_table = tgt_vocab_table
            self.reverse_target_vocab_table = reverse_target_vocab_table

        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        self.initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

        with tf.variable_scope("nmt") as scope:
            # hidden layer
            with tf.name_scope("hidden"):
                encoder_outputs, encoder_state = self._build_encoder()
                output_ids, outputs = self._build_decoder(encoder_outputs, encoder_state)

            self.logits = output_ids
            self.predictions = outputs
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                logits = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='logits')
                max_time = logits.shape[-1].value
                mask = tf.sequence_mask(self.iterator.target_length, max_time, dtype=tf.int32)
                self.accuracy = tf.equal(tf.to_int32(logits * mask), tf.to_int32(self.iterator.target_out * mask))
                self.accuracy = tf.reduce_mean(tf.to_float(self.accuracy))

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False, dtype=dtype, name="learning_rate")
                self._add_loss()
                self._add_train_op()

                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("lr", self.learning_rate)
                tf.summary.scalar("accuracy", self.accuracy)
                self.summary_merge = tf.summary.merge_all()
            else:
                self.confidence = tf.reduce_mean(tf.reduce_max(tf.nn.softmax(tf.to_float(self.predictions), axis=-1), axis=-1), axis=-1)
                if hasattr(self.iterator, 'target_out'):
                    self.target = self.reverse_target_vocab_table.lookup(tf.to_int64(self.iterator.target_out))
                if hasattr(self.iterator, 'source_text'):
                    self.source_text = self.iterator.source_text
                self.predictions = self.reverse_target_vocab_table.lookup(tf.to_int64(self.logits))

    def _build_encoder(self):
        with tf.name_scope("seq_encoder"):
            # encoder_outputs = (batch_size, num_steps, hidden_size)
            # encoder_state = (batch_size, hidden_size) * num_layers
            if self.hparams.encoder_type == 'uni':
                encoder_outputs, encoder_state = simple_embed_rnn(inputs=self.iterator.source,
                                                                  batch_size=self.batch_size,
                                                                  num_units=self.num_units,
                                                                  num_layers=self.num_layers_encoder,
                                                                  num_residual_layers=self.num_residual_layers,
                                                                  num_classes=self.num_encoder_symbols,
                                                                  sequence_length=self.iterator.source_length,
                                                                  # initializer=self.initializer,
                                                                  dropout=self.dropout,
                                                                  unit_type=self.hparams.unit_type)
            elif self.hparams.encoder_type == 'bi':
                encoder_outputs, encoder_state = bidirectional_embed_rnn(inputs=self.iterator.source,
                                                                         num_units=self.num_units,
                                                                         num_layers=self.num_layers_encoder,
                                                                         num_residual_layers=self.num_residual_layers,
                                                                         num_classes=self.num_encoder_symbols,
                                                                         sequence_length=self.iterator.source_length,
                                                                         # initializer=self.initializer,
                                                                         dropout=self.dropout,
                                                                         unit_type=self.hparams.unit_type)
            else:
                raise ValueError("Unknown encoder_type %s" % self.hparams.encoder_type)

            if self.num_layers_encoder > 1:
                encoder_state = tuple([encoder_state for _ in range(self.num_layers_encoder)])
            return encoder_outputs, encoder_state

    def _build_decoder(self, encoder_outputs, encoder_state):
        with tf.name_scope("seq_decoder"):
            batch_size = self.batch_size
            # sequence_length = tf.fill([self.batch_size], self.num_steps)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                sequence_length = self.iterator.target_length
            else:
                sequence_length = self.iterator.source_length
            if (self.mode != tf.contrib.learn.ModeKeys.TRAIN) and self.beam_width > 1:
                batch_size = batch_size * self.beam_width
                encoder_outputs = beam_search_decoder.tile_batch(
                    encoder_outputs, multiplier=self.beam_width)
                encoder_state = nest.map_structure(
                    lambda s: beam_search_decoder.tile_batch(s, self.beam_width), encoder_state)
                sequence_length = beam_search_decoder.tile_batch(
                    sequence_length, multiplier=self.beam_width)

            single_cell = single_rnn_cell(self.hparams.unit_type, self.num_units, self.dropout)
            decoder_cell = MultiRNNCell([single_cell for _ in range(self.num_layers_decoder)])
            decoder_cell = InputProjectionWrapper(decoder_cell, num_proj=self.num_units)
            attention_mechanism = create_attention_mechanism(self.hparams.attention_mechanism,
                                                             self.num_units,
                                                             memory=encoder_outputs,
                                                             source_sequence_length=sequence_length)
            decoder_cell = wrapper.AttentionWrapper(decoder_cell,
                                                    attention_mechanism,
                                                    attention_layer_size=self.num_units,
                                                    output_attention=True,
                                                    alignment_history=False)

            # AttentionWrapperState의 cell_state를 encoder의 state으로 설정한다.
            initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            embeddings_decoder = tf.get_variable("embedding_decoder",
                                                 [self.num_decoder_symbols, self.num_units],
                                                 initializer=self.initializer,
                                                 dtype=tf.float32)
            output_layer = Dense(units=self.num_decoder_symbols, use_bias=True, name="output_layer")

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                decoder_inputs = tf.nn.embedding_lookup(embeddings_decoder, self.iterator.target_in)
                decoder_helper = helper.TrainingHelper(decoder_inputs, sequence_length=sequence_length)

                dec = basic_decoder.BasicDecoder(decoder_cell,
                                                 decoder_helper,
                                                 initial_state,
                                                 output_layer=output_layer)
                final_outputs, final_state, _ = decoder.dynamic_decode(dec,
                                                                       swap_memory=True)
                output_ids = final_outputs.rnn_output
                outputs = final_outputs.sample_id
            else:

                def embedding_fn(inputs):
                    return tf.nn.embedding_lookup(embeddings_decoder, inputs)

                decoding_length_factor = 2.0
                max_encoder_length = tf.reduce_max(self.iterator.source_length)
                maximum_iterations = tf.to_int32(tf.round(
                    tf.to_float(max_encoder_length) * decoding_length_factor))

                tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.hparams.sos)), tf.int32)
                tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.hparams.eos)), tf.int32)
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if self.beam_width == 1:
                    decoder_helper = helper.GreedyEmbeddingHelper(embedding=embedding_fn,
                                                                  start_tokens=start_tokens,
                                                                  end_token=end_token)
                    dec = basic_decoder.BasicDecoder(decoder_cell,
                                                     decoder_helper,
                                                     initial_state,
                                                     output_layer=output_layer)
                else:
                    dec = beam_search_decoder.BeamSearchDecoder(cell=decoder_cell,
                                                                embedding=embedding_fn,
                                                                start_tokens=start_tokens,
                                                                end_token=end_token,
                                                                initial_state=initial_state,
                                                                output_layer=output_layer,
                                                                beam_width=self.beam_width)
                final_outputs, final_state, _ = decoder.dynamic_decode(dec,
                                                                       # swap_memory=True,
                                                                       maximum_iterations=maximum_iterations)
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN or self.beam_width == 1:
                    output_ids = final_outputs.sample_id
                    outputs = final_outputs.rnn_output
                else:
                    output_ids = final_outputs.predicted_ids
                    outputs = final_outputs.beam_search_decoder_output.scores

            return output_ids, outputs

    def _add_loss(self):
        with tf.variable_scope("loss"):
            labels = self.iterator.target_out
            # sparse_softmax_cross_entropy_with_logits
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)

            max_time = labels.shape[1].value
            target_weights = tf.sequence_mask(self.iterator.target_length, max_time, dtype=tf.float32)
            self.loss = tf.reduce_sum(crossent * target_weights) / (tf.to_float(self.batch_size * tf.reduce_mean(self.iterator.target_length)))

    def _add_train_op(self):
        with tf.variable_scope("train_op"):
            if self.hparams.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
            elif self.hparams.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="optimizer")
            else:
                raise ValueError("Unknown optimizer %s" % self.hparams.optimizer)

            # train_op = optimizer.minimize(loss)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=5.0)
            self.learning_rate = _learning_rate_decay(self.hparams.learning_rate,
                                                      self.global_step,
                                                      self.hparams.warmup_steps,
                                                      self.hparams.final_learning_rate)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.train_step = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def step(self, session):
        loss, global_step, learning_rate, accuracy, summary, _ = \
            session.run([self.loss,
                         self.global_step,
                         self.learning_rate,
                         self.accuracy,
                         self.summary_merge,
                         self.train_step])
        return loss, global_step, learning_rate, accuracy, summary

    def eval(self, session):
        predictions, source, target, source_text, confidence = session.run([self.predictions,
                                                                            self.iterator.source,
                                                                            self.target,
                                                                            self.source_text,
                                                                            self.confidence])
        return predictions, source, target, source_text, confidence

    def inference(self, session):
        predictions, confidence, source = session.run([self.predictions, self.confidence, self.iterator.source])

        return predictions, confidence, source


def _learning_rate_decay(init_lr, global_step, warmup_steps, final_learning_rate=0.00001):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    learing_rate = init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
    return tf.maximum(learing_rate, final_learning_rate)


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def _create_attention_images_summary(final_context_state):
    """create attention image and attention summary."""
    attention_images = (final_context_state.alignment_history.stack())
    # Reshape to (batch, src_seq_len, tgt_seq_len,1)
    attention_images = tf.expand_dims(tf.transpose(attention_images, [1, 2, 0]), -1)
    # Scale to range [0, 255]
    attention_images *= 255
    attention_summary = tf.summary.image("attention_images", attention_images)
    return attention_summary
