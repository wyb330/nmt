import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
import math
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell, GRUCell, LayerNormBasicLSTMCell, NASCell, \
    DropoutWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers import core as layers_core
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def single_rnn_cell(unit_type, num_units, dropout, residual_connection=False, residual_fn=None):
    """Create an instance of a single RNN cell."""
    # Cell Type
    if unit_type == "lstm":
        single_cell = BasicLSTMCell(num_units)
    elif unit_type == "gru":
        single_cell = GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        single_cell = LayerNormBasicLSTMCell(num_units, layer_norm=True)
    elif unit_type == "nas":
        single_cell = NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Residual
    if residual_connection:
        single_cell = ResidualWrapper(single_cell, residual_fn=residual_fn)
    if dropout > 0.0:
        single_cell = DropoutWrapper(cell=single_cell, input_keep_prob=(1 - dropout))
    return single_cell


def simple_embed_rnn(inputs,
                     batch_size,
                     num_units,
                     num_layers,
                     num_residual_layers,
                     num_classes,
                     sequence_length=None,
                     initializer=None,
                     dropout=0,
                     unit_type='lstm'):
    if initializer is None:
        initializer = get_initializer('uniform', init_weight=math.sqrt(3))
    embeddings = tf.get_variable('embedding_simple_rnn',
                                 [num_classes, num_units],
                                 initializer=initializer,
                                 dtype=tf.float32)
    encoder_inputs = tf.nn.embedding_lookup(embeddings, inputs)
    if num_layers > 1:
        cells = []
        for i in range(num_layers):
            cells.append(single_rnn_cell(unit_type,
                                         num_units,
                                         dropout,
                                         residual_connection=(i >= num_layers - num_residual_layers)))
        cell = MultiRNNCell(cells)
    else:
        cell = single_rnn_cell(unit_type, num_units, dropout)
    state = cell.zero_state(batch_size, tf.float32)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell,
                                                       encoder_inputs,
                                                       sequence_length=sequence_length,
                                                       initial_state=state)

    return encoder_outputs, encoder_state


def bidirectional_embed_rnn(inputs,
                            num_units,
                            num_layers,
                            num_residual_layers,
                            num_classes,
                            sequence_length=None,
                            initializer=None,
                            dropout=0,
                            unit_type='lstm'):
    if initializer is None:
        initializer = get_initializer('uniform', init_weight=math.sqrt(3))
    embeddings = tf.get_variable('embedding_birectional_rnn',
                                 [num_classes, num_units],
                                 initializer=initializer,
                                 dtype=tf.float32)
    encoder_inputs = tf.nn.embedding_lookup(embeddings, inputs)
    single_cell = single_rnn_cell(unit_type, num_units, dropout)
    if num_layers > 1:
        cells_fw = []
        for i in range(num_layers):
            cells_fw.append(single_rnn_cell(unit_type,
                                            num_units,
                                            dropout,
                                            residual_connection=(i >= num_layers - num_residual_layers)))
        cells_bw = []
        for i in range(num_layers):
            cells_bw.append(single_rnn_cell(unit_type,
                                            num_units,
                                            dropout,
                                            residual_connection=(i >= num_layers - num_residual_layers)))
        cell_fw = MultiRNNCell(cells_fw)
        cell_bw = MultiRNNCell(cells_bw)
    else:
        cell_fw = single_cell
        cell_bw = single_cell
    encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                     cell_bw=cell_bw,
                                                                     inputs=encoder_inputs,
                                                                     sequence_length=sequence_length,
                                                                     dtype=tf.float32,
                                                                     swap_memory=True)

    return tf.concat(encoder_outputs, -1), encoder_state


BaseAttentionMechanism = attention_wrapper._BaseAttentionMechanism  # pylint: disable=protected-access


class MultiHeadAttention(BaseAttentionMechanism):
    def __init__(self,
                 hidden_size,
                 memory,
                 num_heads,
                 memory_sequence_length=None,
                 dtype=None,
                 name="MultiHeadAttention"):
        """Construct the AttentionMechanism mechanism.

        Args:
          hidden_size: The depth of the attention mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional) Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          dtype: The data type for the memory layer of the attention mechanism.
          name: Name to use when creating ops.
        """
        probability_fn = nn_ops.softmax
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(MultiHeadAttention, self).__init__(
            query_layer=None,
            memory_layer=layers_core.Dense(
                hidden_size, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=None,
            name=name)

        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        Args:
          x: A tensor with shape [batch_size, length, hidden_size]
        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
          x: A tensor [batch_size, num_heads, max_time]
        Returns:
          A tensor with shape [batch_size, max_time]
        """
        with tf.name_scope("combine_heads"):
            # batch_size = tf.shape(x)[0]
            # return tf.reshape(x, [batch_size, -1])
            return tf.reduce_mean(x, 1)

    def _score(self, query, keys):
        # Reshape from [batch_size, depth] to [batch_size, 1, depth]
        # for matmul.
        query = array_ops.expand_dims(query, 1)

        # Inner product along the query units dimension.
        # matmul shapes: query is [batch_size, 1, depth] and
        #                keys is [batch_size, max_time, depth].
        # the inner product is asked to **transpose keys' inner shape** to get a
        # batched matmul on:
        #   [batch_size, 1, depth] . [batch_size, depth, max_time]
        # resulting in an output shape of:
        #   [batch_size, 1, max_time].
        # we then squeeze out the center singleton dimension.

        # Split q, k, v into heads.
        depth = (self.hidden_size // self.num_heads)
        q = self.split_heads(query)  # [batch_size, num_heads, 1, depth]
        k = self.split_heads(keys)  # [batch_size, num_heads, max_time, depth]
        q *= depth ** -0.5
        score = tf.matmul(q, k, transpose_b=True)  # [batch_size, num_heads, 1, max_time]
        score = tf.squeeze(score, [2])  # [batch_size, num_heads, max_time]
        # Recombine heads --> [batch_size, max_time]
        score = self.combine_heads(score)

        return score

    def __call__(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "multihead_attention", [query]):
            score = self._score(query, self._keys)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

