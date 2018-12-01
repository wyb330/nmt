import logging
import os
import tensorflow as tf
from models import load_model
import utils.iterator_utils as iterator_utils
import utils.text_utils as text_utils
from vocab.vocab_utils import build_reverse_vocab_table, convert_to_bpe, load_vocab_table
import numpy as np
from hparams import hparams
from argparse import ArgumentParser

logger = logging.getLogger("nmt")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")


def check_vocab(args):
    if not hparams.use_bpe:
        return
    model_path = args.model_path
    source = args.source
    src_bpe_file = os.path.join(model_path, 'bpe-{}.src'.format(hparams.bpe_num_symbols))
    src_sents = convert_to_bpe(source, src_bpe_file)
    in_bpe_file = os.path.join(model_path, 'bpe-input-{}.src'.format(hparams.bpe_num_symbols))
    with open(in_bpe_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(src_sents))


def get_data(args):
    if hparams.use_bpe:
        in_file = os.path.join(args.model_path, 'bpe-input-{}.src'.format(hparams.bpe_num_symbols))
    else:
        in_file = args.source
    with open(in_file, 'r', encoding='utf8') as f:
        sents = f.readlines()

    sents = [sent.strip() for sent in sents]
    return sents


def load_dataset(args, src_placeholder):
    model_path = args.model_path
    src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
    src_vocab_file = os.path.join(model_path, 'vocab.src')
    tgt_vocab_file = os.path.join(model_path, 'vocab.tgt')
    src_vocab, src_vocab_size = load_vocab_table(src_vocab_file)
    tgt_vocab, tgt_vocab_size = load_vocab_table(tgt_vocab_file)
    tgt_reverse_vocab = build_reverse_vocab_table(tgt_vocab_file, hparams)

    return src_vocab, tgt_vocab, src_dataset, tgt_reverse_vocab, src_vocab_size, tgt_vocab_size


def bytes2sent(bytes, end_syms):
    sent = text_utils.format_bpe_text(bytes, end_syms)
    sent = sent.replace(' ##', '')
    return sent


def main(args):
    model_path = args.model_path
    hparams.set_hparam('batch_size', 1)
    hparams.add_hparam('is_training', False)
    check_vocab(args)
    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    datasets = load_dataset(args, src_placeholder)
    iterator = iterator_utils.get_inference_iterator(hparams, datasets)
    src_vocab, tgt_vocab, _, tgt_reverse_vocab, src_vocab_size, tgt_vocab_size = datasets
    hparams.add_hparam('vocab_size_source', src_vocab_size)
    hparams.add_hparam('vocab_size_target', tgt_vocab_size)

    sess, model = load_model(hparams, tf.contrib.learn.ModeKeys.INFER, iterator, src_vocab, tgt_vocab, tgt_reverse_vocab)

    ckpt = tf.train.latest_checkpoint(args.model_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    if ckpt:
        saver.restore(sess, ckpt)
    else:
        raise Exception("can not found checkpoint file")

    src_vocab_file = os.path.join(model_path, 'vocab.src')
    src_reverse_vocab = build_reverse_vocab_table(src_vocab_file, hparams)
    sess.run(tf.tables_initializer())

    index = 1
    inputs = np.array(get_data(args), dtype=np.str)
    with sess:
        logger.info("starting inference...")
        sess.run(iterator.initializer, feed_dict={src_placeholder: inputs})
        eos = hparams.eos.encode()
        pad = hparams.pad.encode()
        while True:
            try:
                predictions, confidence, source = model.inference(sess)
                source_sent = src_reverse_vocab.lookup(tf.constant(list(source[0]), tf.int64))
                source_sent = sess.run(source_sent)
                print(index, text_utils.format_bpe_text(source_sent, [eos, pad]))
                if hparams.beam_width == 1:
                    print(bytes2sent(list(predictions[0]), [eos, pad]))
                else:
                    print(bytes2sent(list(predictions[0][:, 0]), [eos, pad]))
                if confidence is not None:
                    print(confidence[0])
                print()
                if index > args.max_data_size:
                    break
                index += 1
            except tf.errors.OutOfRangeError:
                logger.info('Done inference')
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--max_data_size', default=10000)
    args = parser.parse_args()
    main(args)
