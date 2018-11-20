import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
import math
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell, GRUCell, LayerNormBasicLSTMCell, NASCell, \
    DropoutWrapper, ResidualWrapper


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

