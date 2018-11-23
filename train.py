import time
import os
import math
import logging
import tensorflow as tf
from models import load_model
from vocab.vocab_utils import build_vocab_table, get_vocaburary_list, convert_to_bpe, create_bpe, load_vocab_table
import utils.iterator_utils as iterator_utils
from argparse import ArgumentParser
from hparams import hparams
from pprint import pprint


logger = logging.getLogger("train")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")


def load_dataset(args, vocab_dir):
    src_file = args.source
    tgt_file = args.target

    src_bpe_text = os.path.join(vocab_dir, '{}.bpe'.format(os.path.basename(src_file)))
    tgt_bpe_text = os.path.join(vocab_dir, '{}.bpe'.format(os.path.basename(tgt_file)))
    src_dataset = tf.data.TextLineDataset(src_bpe_text)
    tgt_dataset = tf.data.TextLineDataset(tgt_bpe_text)

    src_vocab_bpe_file = os.path.join(vocab_dir, 'vocab.bpe.{}.src'.format(hparams.bpe_num_symbols))
    tgt_vocab_bpe_file = os.path.join(vocab_dir, 'vocab.bpe.{}.tgt'.format(hparams.bpe_num_symbols))
    if os.path.exists(src_vocab_bpe_file) and os.path.exists(tgt_vocab_bpe_file):
        src_vocab, src_vocab_size = load_vocab_table(src_vocab_bpe_file)
        tgt_vocab, tgt_vocab_size = load_vocab_table(tgt_vocab_bpe_file)
    else:
        src_vocab, src_vocab_size = build_vocab_table(src_bpe_text, hparams, src_vocab_bpe_file)
        tgt_vocab, tgt_vocab_size = build_vocab_table(tgt_bpe_text, hparams, tgt_vocab_bpe_file)
    logger.info('{} vocab bpe size={}'.format(src_vocab_bpe_file, src_vocab_size))
    logger.info('{} vocab bpe size={}'.format(tgt_vocab_bpe_file, tgt_vocab_size))

    return src_vocab, tgt_vocab, src_dataset, tgt_dataset, src_vocab_size, tgt_vocab_size


def create_vocab(text_file, vocab_file, vocab_size, tokenize_fn):
    vocaburary_list = get_vocaburary_list(text_file, vocab_size=vocab_size, tokenize_fn=tokenize_fn)
    with open(vocab_file, 'wt', encoding='utf8') as f:
        for word in vocaburary_list:
            f.write(word + '\n')


def check_vocab(args, vocab_dir):
    src_file = args.source
    tgt_file = args.target

    src_bpe_file = os.path.join(vocab_dir, 'bpe-{}.src'.format(hparams.bpe_num_symbols))
    tgt_bpe_file = os.path.join(vocab_dir, 'bpe-{}.tgt'.format(hparams.bpe_num_symbols))
    if not os.path.exists(src_bpe_file) or not os.path.exists(tgt_bpe_file):
        create_bpe(src_file, src_bpe_file, hparams.bpe_num_symbols)
        create_bpe(tgt_file, tgt_bpe_file, hparams.bpe_num_symbols)

    src_bpe_text = os.path.join(vocab_dir, '{}.bpe'.format(os.path.basename(src_file)))
    tgt_bpe_text = os.path.join(vocab_dir, '{}.bpe'.format(os.path.basename(tgt_file)))
    if not os.path.exists(src_bpe_text) or not os.path.exists(tgt_bpe_text):
        src_sents = convert_to_bpe(src_file, src_bpe_file)
        with open(src_bpe_text, 'w', encoding='utf8') as f:
            f.write('\n'.join(src_sents))
        tgt_sents = convert_to_bpe(tgt_file, tgt_bpe_file)
        with open(tgt_bpe_text, 'w', encoding='utf8') as f:
            f.write('\n'.join(tgt_sents))


def write_summary(writer, summary, num_step):
    writer.add_summary(summary, num_step)


def main(args, max_data_size=0):
    vocab_dir = args.vocab_dir
    log_file_handler = logging.FileHandler(os.path.join(vocab_dir, 'train.log'))
    logger.addHandler(log_file_handler)

    check_vocab(args, vocab_dir)
    datasets = load_dataset(args, vocab_dir)
    iterator = iterator_utils.get_iterator(hparams, datasets, max_rows=max_data_size)
    src_vocab, tgt_vocab, _, _, src_vocab_size, tgt_vocab_size = datasets
    hparams.add_hparam('is_training', True)
    hparams.add_hparam('vocab_size_source', src_vocab_size)
    hparams.add_hparam('vocab_size_target', tgt_vocab_size)
    pprint(hparams.values())
    sess, model = load_model(hparams, tf.contrib.learn.ModeKeys.TRAIN, iterator, src_vocab, tgt_vocab)

    if args.restore_step > 0:
        checkpoint_path = os.path.join(vocab_dir, 'nmt.ckpt')
        ckpt = '%s-%d' % (checkpoint_path, hparams.restore_step)
    else:
        ckpt = tf.train.latest_checkpoint(vocab_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    if ckpt:
        saver.restore(sess, ckpt)
    else:
        sess.run(tf.global_variables_initializer())
        print("Created model with fresh parameters.")

    sess.run(tf.tables_initializer())
    with sess:
        writer = tf.summary.FileWriter(vocab_dir, sess.graph)
        logger.info("starting training...")
        epochs = 1
        step_in_epoch = 0
        learning_rate = hparams.learning_rate
        checkpoint_path = os.path.join(vocab_dir, "nmt.ckpt")

        sess.run(iterator.initializer)
        while epochs <= args.num_train_epochs:
            start_time = time.time()
            try:
                loss, global_step, learning_rate, accuracy, summary = model.step(sess)
                step_in_epoch += 1
                if global_step % args.summary_per_steps == 0:
                    write_summary(writer, summary, global_step)

            except tf.errors.OutOfRangeError:
                logger.info('{} epochs finished'.format(epochs))
                # saver.save(sess, checkpoint_path, global_step=global_step)
                epochs += 1
                step_in_epoch = 1
                sess.run(iterator.initializer)
                continue

            sec_per_step = time.time() - start_time
            logger.info("Epoch %-3d Step %-d - %-d [%.3f sec, loss=%.4f, acc=%.3f, lr=%f]" %
                        (epochs, global_step, step_in_epoch, sec_per_step, loss, accuracy, learning_rate))

            if global_step % args.steps_per_checkpoint == 0:
                model_checkpoint_path = saver.save(sess, checkpoint_path, global_step=global_step)
                logger.info("Saved checkpoint to {}".format(model_checkpoint_path))

            if math.isnan(loss) or math.isinf(loss):
                raise Exception('loss overflow')

        writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vocab_dir')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--model_type', default='attention')
    parser.add_argument('--restore_step', default=0)
    parser.add_argument('--num_train_epochs', default=10000)
    parser.add_argument('--steps_per_checkpoint', default=1000)
    parser.add_argument('--summary_per_steps', default=500)
    args = parser.parse_args()
    main(args)
