import tensorflow as tf


hparams = tf.contrib.training.HParams(
    source='./data/sample.en',
    target='./data/sample.kr',
    bpe_num_symbols=128,
    sos='<s>',
    eos='</s>',
    pad='<pad>',
    unk='<unk>',

)
