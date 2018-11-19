import os
from collections import Counter
import utils.text_utils as text_utils
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from vocab import learn_bpe, apply_bpe


def create_bpe(text_file, bpe_file, size, min_frequency=2):
    learn_bpe.train(text_file, bpe_file, size, min_frequency)


def get_vocaburary_list(text_file, vocab_size, tokenize_fn=None):
    sents = text_utils.read_text(text_file)
    sent_words = []
    for sent in sents:
        if tokenize_fn is not None:
            words = tokenize_fn(sent)
        else:
            words = sent.split()
        sent_words.append(words)

    counter = Counter()
    for words in sent_words:
        counter.update(words)

    vocab = sorted(counter, reverse=True)
    if vocab_size > 0:
        vocab_list = list(vocab[:vocab_size])
    else:
        vocab_list = list(vocab)

    return vocab_list


def build_vocab_bpe(text_file, hparams, vocab_file):
    vocab_list = get_vocaburary_list(text_file, 0, tokenize_fn=None)
    vocab_list = [hparams.unk, hparams.sos, hparams.eos, hparams.pad] + vocab_list
    if not os.path.exists(vocab_file):
        with open(vocab_file, 'w', encoding='utf8') as f:
            for word in vocab_list:
                f.write(word + '\n')
    return vocab_list


def build_vocab_table(text_file, hparams, vocab_file):
    vocab_list = build_vocab_bpe(text_file, hparams, vocab_file)
    vocab_size = len(vocab_list)
    print('{} vocab size={}'.format(text_file, vocab_size))
    vocabulary_list = tf.constant(vocab_list, dtype=tf.string)
    return lookup_ops.index_table_from_tensor(vocabulary_list, default_value=0), vocab_size


def build_reverse_vocab_table(vocab_bpe_file, hparams):
    assert os.path.exists(vocab_bpe_file)
    with open(vocab_bpe_file, 'r', encoding='utf8') as f:
        vocab_list = f.readlines()
    vocab_list = [word.strip() for word in vocab_list]
    vocabulary_list = tf.constant(vocab_list, dtype=tf.string)
    return lookup_ops.index_to_string_table_from_tensor(vocabulary_list, default_value=hparams.unk)


def load_vocab_table(vocab_bpe_file):
    assert os.path.exists(vocab_bpe_file)
    with open(vocab_bpe_file, 'r', encoding='utf8') as f:
        vocab_list = f.readlines()
    vocab_list = [word.strip() for word in vocab_list]
    return lookup_ops.index_table_from_tensor(vocab_list, default_value=0), len(vocab_list)


def convert_to_bpe(input_file, bpe_file):
    with open(input_file, "r", encoding='utf8') as f:
        fb = open(bpe_file, "r", encoding='utf8')
        bpe = apply_bpe.BPE(fb, separator="@")
        sents = []
        for line in f.readlines():
            if not line.strip():
                continue
            words = text_utils.tokenize_sent(line.strip())
            sent = []
            for word in words:
                segments = bpe.segment_word(word)
                sent.append(' '.join(segments))
            sents.append(' '.join(sent))
        fb.close()

    return sents
