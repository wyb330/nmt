import tensorflow as tf


hparams = tf.contrib.train.HParams(
    source='./data/sample.en',
    target='./data/sample.kr',
    bpe_num_symbols=128,

)
