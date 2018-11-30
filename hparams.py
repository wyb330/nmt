from tensorflow.contrib.training import HParams


params_scale = 'large'  # ['small', 'medium', 'large']

hparams = HParams(
    batch_size=16,
    bpe_num_symbols=128,
    num_units=64,
    num_layers_encoder=2,
    num_layers_decoder=2,
    src_max_len=200,
    tgt_max_len=200,
    infer_src_max_len=200,
    infer_tgt_max_len=200,
    optimizer='adam',
    learning_rate=0.0001,
    final_learning_rate=0.00002,
    warmup_steps=4000,
    bucket=0,
    use_bpe=False,
    attention_mechanism='scaled_luong',
    beam_width=1,
    encoder_type='bi',
    unit_type='gru',
    dropout=0.2,
    num_residual_layers=0,
    sos='<s>',
    eos='</s>',
    pad='<pad>',
    unk='<unk>',
)

if params_scale == 'medium':
    hparams.set_hparam('batch_size', 32)
    hparams.set_hparam('bpe_num_symbols', 24000)
    hparams.set_hparam('num_units', 256)
elif params_scale == 'large':
    hparams.set_hparam('batch_size', 64)
    hparams.set_hparam('bpe_num_symbols', 32000)
    hparams.set_hparam('num_units', 512)

