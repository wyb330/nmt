import logging
import os
import time
import tensorflow as tf
from models import load_model
import utils.iterator_utils as iterator_utils
import utils.text_utils as text_utils
from utils import bleu, metrics
from vocab.vocab_utils import build_reverse_vocab_table, convert_to_bpe, load_vocab_table
from utils.bleu_moses import moses_multi_bleu
from hparams import hparams
from argparse import ArgumentParser

logger = logging.getLogger("eval")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")


def check_vocab(args):
    source = args.source
    target = args.target
    src_bpe_file = os.path.join(args.model_path, 'bpe-{}.src'.format(hparams.bpe_num_symbols))
    tgt_bpe_file = os.path.join(args.model_path, 'bpe-{}.tgt'.format(hparams.bpe_num_symbols))

    in_src_bpe_file = os.path.join(args.model_path, 'bpe-input-{}.src'.format(hparams.bpe_num_symbols))
    src_sents = convert_to_bpe(source, src_bpe_file)
    with open(in_src_bpe_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(src_sents))

    in_tgt_bpe_file = os.path.join(args.model_path, 'bpe-input-{}.tgt'.format(hparams.bpe_num_symbols))
    tgt_sents = convert_to_bpe(target, tgt_bpe_file)
    with open(in_tgt_bpe_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(tgt_sents))


def load_dataset():
    in_src_bpe_file = os.path.join(args.model_path, 'bpe-input-{}.src'.format(hparams.bpe_num_symbols))
    in_tgt_bpe_file = os.path.join(args.model_path, 'bpe-input-{}.tgt'.format(hparams.bpe_num_symbols))

    src_dataset = tf.data.TextLineDataset(in_src_bpe_file)
    tgt_dataset = tf.data.TextLineDataset(in_tgt_bpe_file)
    src_vocab_bpe_file = os.path.join(args.model_path, 'vocab.bpe.{}.src'.format(hparams.bpe_num_symbols))
    tgt_vocab_bpe_file = os.path.join(args.model_path, 'vocab.bpe.{}.tgt'.format(hparams.bpe_num_symbols))
    src_vocab, src_vocab_size = load_vocab_table(src_vocab_bpe_file)
    tgt_vocab, tgt_vocab_size = load_vocab_table(tgt_vocab_bpe_file)
    tgt_reverse_vocab = build_reverse_vocab_table(tgt_vocab_bpe_file, hparams)

    with open(in_src_bpe_file, 'r', encoding='utf8') as f:
        src_data_size = len(f.readlines())

    return (src_vocab, tgt_vocab, src_dataset, tgt_dataset, tgt_reverse_vocab, src_vocab_size, tgt_vocab_size), src_data_size


def bytes2sent(byte_sents, eos):
    sents = []
    for byte_sent in byte_sents:
        sent = text_utils.format_bpe_text(byte_sent, eos)
        sent = sent.replace(' ##', '')
        sents.append(sent)
    return sents


def bpe2sent(bpe_sents, eos):
    sents = []
    for bpe_sent in bpe_sents:
        sent = text_utils.format_bpe_text(bpe_sent, eos)
        sents.append(sent)

    return sents


def compute_bleu_score(references, translations, max_order=4, smooth=False):
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(references, translations, max_order, smooth)
    print(bleu_score)
    return bleu_score * 100


def main(args, max_data_size=0, shuffle=True, display=False):
    hparams.set_hparam('batch_size', 10)
    hparams.add_hparam('is_training', False)
    check_vocab(args)
    datasets, src_data_size = load_dataset()
    iterator = iterator_utils.get_eval_iterator(hparams, datasets, hparams.eos, shuffle=shuffle)
    src_vocab, tgt_vocab, src_dataset, tgt_dataset, tgt_reverse_vocab, src_vocab_size, tgt_vocab_size = datasets
    hparams.add_hparam('vocab_size_source', src_vocab_size)
    hparams.add_hparam('vocab_size_target', tgt_vocab_size)

    sess, model = load_model(hparams, tf.contrib.learn.ModeKeys.EVAL, iterator, src_vocab, tgt_vocab, tgt_reverse_vocab)

    if args.restore_step:
        checkpoint_path = os.path.join(args.model_path, 'nmt.ckpt')
        ckpt = '%s-%d' % (checkpoint_path, args.restore_step)
    else:
        ckpt = tf.train.latest_checkpoint(args.model_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    if ckpt:
        saver.restore(sess, ckpt)
    else:
        raise Exception("can not found checkpoint file")

    src_vocab_bpe_file = os.path.join(args.model_path, 'vocab.bpe.{}.src'.format(hparams.bpe_num_symbols))
    src_reverse_vocab = build_reverse_vocab_table(src_vocab_bpe_file, hparams)
    sess.run(tf.tables_initializer())

    step_count = 1
    with sess:
        logger.info("starting evaluating...")
        sess.run(iterator.initializer)
        eos = hparams.eos.encode()
        references = []
        translations = []
        start_time = time.time()
        while True:
            try:
                if (max_data_size > 0) and (step_count * hparams.batch_size > max_data_size):
                    break
                if step_count % 10 == 0:
                    t = time.time() - start_time
                    logger.info('step={0} total={1} time={2:.3f}'.format(step_count, step_count * hparams.batch_size, t))
                    start_time = time.time()
                predictions, source, target, source_text, confidence = model.eval(sess)
                reference = bpe2sent(target, eos)
                if hparams.beam_width == 1:
                    translation = bytes2sent(list(predictions), eos)
                else:
                    translation = bytes2sent(list(predictions[:, 0]), eos)

                for s, r, t in zip(source, reference, translation):
                    if display:
                        source_sent = src_reverse_vocab.lookup(tf.constant(list(s), tf.int64))
                        source_sent = sess.run(source_sent)
                        source_sent = text_utils.format_bpe_text(source_sent, eos)
                        print('{}\n{}\n{}\n'.format(source_sent, r, t))
                    references.append(r)
                    translations.append(t)

                if step_count % 100 == 0:
                    bleu_score = moses_multi_bleu(references, translations, args.model_path)
                    logger.info('bleu score = {0:.3f}'.format(bleu_score))

                step_count += 1
            except tf.errors.OutOfRangeError:
                logger.info('Done eval data')
                break

        logger.info('compute bleu score...')
        # bleu_score = compute_bleu_score(references, translations)
        bleu_score = moses_multi_bleu(references, translations, args.model_path)
        logger.info('bleu score = {0:.3f}'.format(bleu_score))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--restore_step', default=0)
    args = parser.parse_args()
    main(args)
