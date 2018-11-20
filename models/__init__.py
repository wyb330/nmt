import tensorflow as tf
from models.attention_model import AttentionModel


def create_model(hparams, mode, iterator, src_vocab_table, tgt_vocab_table, tgt_reverse_vocab, model_type='attention'):
    dtype = tf.float32
    if model_type == 'attention':
        model = AttentionModel(hparams=hparams,
                               mode=mode,
                               src_vocab_table=src_vocab_table,
                               tgt_vocab_table=tgt_vocab_table,
                               reverse_target_vocab_table=tgt_reverse_vocab,
                               iterator=iterator,
                               dtype=dtype)
    else:
        raise Exception("Unsupported model type")

    return model


def load_model(hparams, mode, iterator, src_vocab_table, tgt_vocab_table, tgt_reverse_vocab=None):
    sess = tf.Session()
    model = create_model(hparams, mode, iterator, src_vocab_table, tgt_vocab_table, tgt_reverse_vocab)
    return sess, model
