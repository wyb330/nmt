import os
from argparse import ArgumentParser
from vocab.vocab_utils import convert_to_bpe, create_bpe, build_vocab_bpe
from hparams import hparams


def main(args):
    out_dir = args.out_dir
    src_file = hparams.source
    tgt_file = hparams.target

    src_bpe_file = os.path.join(out_dir, 'bpe-{}.src'.format(hparams.bpe_num_symbols))
    tgt_bpe_file = os.path.join(out_dir, 'bpe-{}.tgt'.format(hparams.bpe_num_symbols))
    if not os.path.exists(src_bpe_file) or not os.path.exists(tgt_bpe_file):
        print('step [1/6] creating source bpe file')
        create_bpe(src_file, src_bpe_file, hparams.bpe_num_symbols)
        print('step [2/6] creating target bpe file')
        create_bpe(tgt_file, tgt_bpe_file, hparams.bpe_num_symbols)
    else:
        print('step [1/6] passed')
        print('step [2/6] passed')

    src_bpe_text = os.path.join(out_dir, '{}.bpe'.format(os.path.basename(src_file)))
    tgt_bpe_text = os.path.join(out_dir, '{}.bpe'.format(os.path.basename(tgt_file)))
    if not os.path.exists(src_bpe_text) or not os.path.exists(tgt_bpe_text):
        print('step [3/6] converting source file to bpe file')
        src_sents = convert_to_bpe(src_file, src_bpe_file)
        with open(src_bpe_text, 'w', encoding='utf8') as f:
            f.write('\n'.join(src_sents))
        print('step [4/6] converting target file to bpe file')
        tgt_sents = convert_to_bpe(tgt_file, tgt_bpe_file)
        with open(tgt_bpe_text, 'w', encoding='utf8') as f:
            f.write('\n'.join(tgt_sents))
    else:
        print('step [3/6] passed')
        print('step [4/6] passed')

    src_vocab_bpe_file = os.path.join(out_dir, 'vocab.bpe.{}.src'.format(hparams.bpe_num_symbols))
    tgt_vocab_bpe_file = os.path.join(out_dir, 'vocab.bpe.{}.tgt'.format(hparams.bpe_num_symbols))
    print('step [5/6] building source vocab bpe file')
    build_vocab_bpe(src_bpe_text, hparams, src_vocab_bpe_file)
    print('step [6/6] building target vocab bpe file')
    build_vocab_bpe(tgt_bpe_text, hparams, tgt_vocab_bpe_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    main(args)
