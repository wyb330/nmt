# Tensorflow Neural Machine Translation


### Training

1. **Preprocess the data**
```
python preprocess.py --source [source file] --target [target file] --out_dir /out/path
```

2. **Training**
```
python train.py --source [source file] --target [target file] --vocab_dir /vocab/path
```

3. **Predict**
```
python predict.py --model_path /model/path --source [source file]
```

4. **Eavluate**
```
python eval.py --model_path /model/path --source [source file]  --target [target file]
```
