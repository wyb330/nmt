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


class TranformerAttention(BaseAttentionMechanism):
    def __init__(self,
                 hidden_size,
                 memory,
                 num_heads,
                 attention_dropout,
                 memory_sequence_length=None,
                 dtype=None,
                 name="TranformerAttention"):
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
        super(TranformerAttention, self).__init__(
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
        self.attention_dropout = attention_dropout
        self.train = True
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

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
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

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

        """Apply attention mechanism to x and y.
        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.
        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        with variable_scope.variable_scope(None, "transformer_attention", [query]):
            # score = _transformer_score(query, self._keys)
            # alignments = self._probability_fn(score, state)
            # next_state = alignments
            # return alignments, next_state

            x = array_ops.expand_dims(query, 1)
            y = self._keys
            # Linearly project the query (q), key (k) and value (v) using different
            # learned projections. This is in preparation of splitting them into
            # multiple heads. Multi-head attention uses multiple queries, keys, and
            # values rather than regular attention (which uses a single q, k, v).
            q = self.q_dense_layer(x)
            k = self.k_dense_layer(y)
            v = self.v_dense_layer(y)

            # Split q, k, v into heads.
            q = self.split_heads(q)
            k = self.split_heads(k)
            v = self.split_heads(v)

            # Scale q to prevent the dot product between q and k from growing too large.
            depth = (self.hidden_size // self.num_heads)
            q *= depth ** -0.5

            # Calculate dot product attention
            logits = tf.matmul(q, k, transpose_b=True)
            # logits += bias
            weights = tf.nn.softmax(logits, name="attention_weights")
            if self.train:
                weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
            attention_output = tf.matmul(weights, v)

            # Recombine heads --> [batch_size, length, hidden_size]
            attention_output = self.combine_heads(attention_output)

            # Run the combined outputs through another linear projection layer.
            attention_output = self.output_dense_layer(attention_output)

        return attention_output, attention_output




